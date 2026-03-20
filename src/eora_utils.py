# src/eora_utils.py
from __future__ import annotations

from typing import Dict, Any, Tuple, List
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def _is_gpt2_conv1d(m: torch.nn.Module) -> bool:
    """
    GPT-2 uses transformers.pytorch_utils.Conv1D for many projection layers.
    We detect by class name to avoid hard importing transformers internals.
    """
    return m.__class__.__name__ == "Conv1D"


@torch.no_grad()
def _svd_low_rank(delta: torch.Tensor, rank: int, svd_on_cpu: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a rank-r factorization delta ≈ B @ A, where:
      - B: [out, r]
      - A: [r, in]
    delta is expected to be in [out, in] orientation.

    Use SVD: delta = U S Vh
      B = U[:, :r] * S[:r]
      A = Vh[:r, :]
    """
    if delta.dtype not in (torch.float32, torch.float64):
        delta = delta.float()

    device = delta.device
    if svd_on_cpu:
        delta_ = delta.detach().cpu()
    else:
        delta_ = delta

    U, S, Vh = torch.linalg.svd(delta_, full_matrices=False)

    r = min(rank, S.numel())
    U_r = U[:, :r]                 # [out, r]
    S_r = S[:r]                    # [r]
    Vh_r = Vh[:r, :]               # [r, in]

    B = U_r * S_r.unsqueeze(0)     # [out, r]
    A = Vh_r                       # [r, in]

    if svd_on_cpu:
        B = B.to(device)
        A = A.to(device)

    return B, A


def _resolve_peft_task_type(config: Dict[str, Any]) -> TaskType:
    """
    Map config.task_type to PEFT TaskType.
    Accepts: classification / seq_cls / lm / causal_lm / seq2seq
    """
    t = (config.get("task_type") or "classification").lower()
    if t in ["lm", "causal_lm", "causallm", "causal-lm"]:
        return TaskType.CAUSAL_LM
    if t in ["seq2seq", "seq2seq_lm", "seq2seqlm"]:
        return TaskType.SEQ_2_SEQ_LM
    # default: classification
    return TaskType.SEQ_CLS


@torch.no_grad()
def apply_eora_to_base(
    base_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.nn.Module:
    """
    Build an EoRA-style adapter on top of base_model so that base+adapter ≈ optimized_model.

    Steps:
      1) Wrap base_model with PEFT LoRA layers on target_modules.
      2) For each target layer, compute ΔW = W_opt - W_base (in "effective" [out,in] orientation).
      3) Compute rank-r SVD low-rank approximation: ΔW ≈ B@A.
      4) Write (A,B) into LoRA A/B weights.

    Notes:
      - For GPT-2 Conv1D layers: stored weight is [in, out], effective [out, in] is W^T.
      - For classification models: copy the classification head (score) from optimized to base
        to avoid random head ruining metrics. For LM there is no score head, it will be skipped.
      - EoRA branch: NO training, freeze all params.
    """
    eora_cfg = config.get("eora", {})
    rank = int(eora_cfg.get("rank", 8))
    alpha = int(eora_cfg.get("alpha", rank))
    dropout = float(eora_cfg.get("dropout", 0.0))
    target_modules = eora_cfg.get("target_modules", ["c_attn", "c_proj"])
    svd_on_cpu = bool(eora_cfg.get("svd_on_cpu", True))

    # PEFT task type should match config.task_type (LM vs CLS)
    peft_task = _resolve_peft_task_type(config)

    # 1) Wrap base_model with LoRA
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=peft_task,
    )
    eora_model = get_peft_model(base_model, lora_cfg)

    # 2) Map optimized modules by name
    opt_modules = dict(optimized_model.named_modules())

    # 3) Fill LoRA weights from ΔW low-rank approximation
    prefixes = ["base_model.model.", "base_model.", "model."]
    adapter_name = "default"

    num_filled = 0
    for name, module in eora_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        # Align module name between PEFT-wrapped base and optimized model
        opt_name = name
        for pfx in prefixes:
            if opt_name.startswith(pfx):
                opt_name = opt_name[len(pfx):]
                break

        if opt_name not in opt_modules:
            continue

        opt_layer = opt_modules[opt_name]
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None or not hasattr(base_layer, "weight") or not hasattr(opt_layer, "weight"):
            continue

        Wb = base_layer.weight.data
        Wo = opt_layer.weight.data

        # Effective [out, in]
        if _is_gpt2_conv1d(base_layer):
            delta = (Wo.T - Wb.T).contiguous()
        else:
            delta = (Wo - Wb).contiguous()

        B, A = _svd_low_rank(delta, rank=rank, svd_on_cpu=svd_on_cpu)

        # PEFT expects:
        #   lora_A.weight: [r, in]
        #   lora_B.weight: [out, r]
        module.lora_A[adapter_name].weight.data.copy_(A)
        module.lora_B[adapter_name].weight.data.copy_(B)
        num_filled += 1

    # 4) Copy classification head if exists
    if hasattr(eora_model, "score") and hasattr(optimized_model, "score"):
        try:
            eora_model.score.weight.data.copy_(optimized_model.score.weight.data)
            if hasattr(eora_model.score, "bias") and eora_model.score.bias is not None:
                eora_model.score.bias.data.copy_(optimized_model.score.bias.data)
        except Exception:
            pass

    # 5) Freeze everything (NO training in EoRA branch)
    for p in eora_model.parameters():
        p.requires_grad_(False)
    eora_model.eval()

    scaling = alpha / max(rank, 1)
    print(f"[EoRA] Filled LoRA layers: {num_filled}")
    print(f"[EoRA] task={peft_task}, rank={rank}, alpha={alpha} (scaling={scaling:.3f}), svd_on_cpu={svd_on_cpu}")
    return eora_model


@torch.no_grad()
def save_eora_adapter(eora_model: torch.nn.Module, tokenizer, output_dir: str) -> str:
    """
    Save PEFT adapter and tokenizer to output_dir/adapter
    """
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    eora_model.save_pretrained(adapter_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(adapter_dir)

    return adapter_dir


@torch.no_grad()
def apply_eora_base_to_optimized(
    base_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.nn.Module:
    """
    Backward-compatible alias.
    """
    return apply_eora_to_base(base_model, optimized_model, config)


# ============================================================
# Quantized workflow (Exp3 / GPTQModel) - OPTIONAL
#   These functions are isolated so exp1/exp2 (no quant) won't
#   crash if gptqmodel isn't installed in the current env.
# ============================================================
def _build_calibration_texts(config: Dict[str, Any]) -> List[str]:
    """
    EoRA (GPTQModel.adapter.generate) needs a list of texts as calibration_dataset.

    Priority:
      1) eora.calibration_local_txt (local file, avoid HF SSL)
      2) fallback to HF dataset specified by data.dataset_name (legacy)
    """
    eora_cfg = config.get("eora", {})
    n = int(eora_cfg.get("calibration_num_samples", 512))

    local_txt = eora_cfg.get("calibration_local_txt", None)
    if local_txt:
        p = Path(local_txt)
        if not p.exists():
            raise FileNotFoundError(f"calibration_local_txt not found: {p}")
        texts: List[str] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)
                if len(texts) >= n:
                    break
        if not texts:
            raise ValueError(f"calibration_local_txt is empty: {p}")
        return texts

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "glue/sst2")

    if "/" in dataset_name:
        d0, d1 = dataset_name.split("/", 1)
        ds = load_dataset(d0, d1)
        text_key = "sentence"
    else:
        ds = load_dataset(dataset_name)
        text_key = "text"

    texts = ds["train"].select(range(n))[text_key]
    return list(texts)


def generate_eora_adapter_for_quantized(config: Dict[str, Any], save_dir: str):
    """
    Exp3: teacher=optimized(full precision) + student=quantized(GPTQ)
    Generate EoRA adapter using GPTQModel.adapter.generate(...) and save into save_dir.
    """
    # Import here to avoid hard dependency for non-quant experiments
    from gptqmodel import GPTQModel
    from gptqmodel.adapter.adapter import Lora

    eora_cfg = config.get("eora", {})
    rank = int(eora_cfg.get("rank", 16))

    optimized_model_dir = config.get("optimized_model_dir")
    quantized_model_dir = config.get("quantized_model_dir")

    if not optimized_model_dir:
        raise ValueError("config['optimized_model_dir'] is required.")
    if not quantized_model_dir:
        raise ValueError("config['quantized_model_dir'] is required.")

    os.makedirs(save_dir, exist_ok=True)
    calibration_texts = _build_calibration_texts(config)

    eora = Lora(path=save_dir, rank=rank)

    GPTQModel.adapter.generate(
        adapter=eora,
        model_id_or_path=optimized_model_dir,
        quantized_model_id_or_path=quantized_model_dir,
        calibration_dataset=calibration_texts,
        calibration_dataset_concat_size=0,
        auto_gc=False,
    )
    return eora


def load_quantized_with_eora(config: Dict[str, Any], eora):
    """
    Load quantized model with attached EoRA adapter (for inference/eval).
    """
    from gptqmodel import GPTQModel

    quantized_model_dir = config["quantized_model_dir"]
    model = GPTQModel.load(
        model_id_or_path=quantized_model_dir,
        adapter=eora,
    )
    return model
