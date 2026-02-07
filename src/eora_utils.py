from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from gptqmodel import GPTQModel
from gptqmodel.adapter.adapter import Lora


def _get_module_by_name(root: torch.nn.Module, name: str) -> torch.nn.Module:
    """
    Given a dotted module name, return the corresponding submodule.
    """
    cur = root
    for part in name.split("."):
        if not hasattr(cur, part):
            raise AttributeError(f"Module '{type(cur).__name__}' has no attribute '{part}' while resolving '{name}'.")
        cur = getattr(cur, part)
    return cur


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

    We use SVD: delta = U S Vh, then set:
      B = U[:, :r] * S[:r]
      A = Vh[:r, :]
    so that B@A = U_r diag(S_r) Vh_r is the best rank-r approximation (Frobenius).
    """
    if delta.dtype not in (torch.float32, torch.float64):
        delta = delta.float()

    device = delta.device
    if svd_on_cpu:
        delta_ = delta.detach().cpu()
    else:
        delta_ = delta

    # full_matrices=False is faster and uses less memory
    U, S, Vh = torch.linalg.svd(delta_, full_matrices=False)

    r = min(rank, S.numel())
    U_r = U[:, :r]                           # [out, r]
    S_r = S[:r]                              # [r]
    Vh_r = Vh[:r, :]                         # [r, in]

    B = U_r * S_r.unsqueeze(0)               # [out, r]
    A = Vh_r                                 # [r, in]

    if svd_on_cpu:
        B = B.to(device)
        A = A.to(device)

    return B, A


@torch.no_grad()
def apply_eora_to_base(
    base_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.nn.Module:
    """
    Build an EoRA-style adapter on top of base_model so that base+adapter ≈ optimized_model.

    Implementation:
      1) Wrap base_model with PEFT LoRA layers on target_modules.
      2) For each target layer, compute ΔW = W_opt - W_base (in "effective" [out,in] orientation).
      3) Compute rank-r SVD low-rank approximation: ΔW ≈ B@A.
      4) Write (A,B) into LoRA A/B weights, set alpha=r so scaling=1.

    Also:
      - Copy classification head weights (if exists) from optimized to base, since it's tiny and task-specific.
        This prevents "random head" from dominating results.
    """
    eora_cfg = config.get("eora", {})
    rank = int(eora_cfg.get("rank", 8))
    alpha = int(eora_cfg.get("alpha", rank))  # recommend alpha=rank => scaling=1
    dropout = float(eora_cfg.get("dropout", 0.0))
    target_modules = eora_cfg.get("target_modules", ["c_attn", "c_proj"])
    svd_on_cpu = bool(eora_cfg.get("svd_on_cpu", True))

    # 1) Build PEFT LoRA wrapper on base_model
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_CLS",  # fits SST-2 classification; doesn't affect weight injection
    )
    eora_model = get_peft_model(base_model, lora_cfg)

    # 2) Build name->module map for optimized model
    opt_modules = dict(optimized_model.named_modules())

    # 3) Fill LoRA weights using ΔW low-rank approximation
    # PEFT-wrapped model names are like: base_model.model.transformer.h.0.attn.c_attn
    # We strip "base_model.model." prefix to align with optimized model names.
    # 3) Fill LoRA weights using ΔW low-rank approximation
    prefixes = ["base_model.model.", "base_model.", "model."]
    adapter_name = "default"

    num_filled = 0
    for name, module in eora_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

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

        if _is_gpt2_conv1d(base_layer):
            delta = (Wo.T - Wb.T).contiguous()
        else:
            delta = (Wo - Wb).contiguous()

        B, A = _svd_low_rank(delta, rank=rank, svd_on_cpu=svd_on_cpu)

        module.lora_A[adapter_name].weight.data.copy_(A)
        module.lora_B[adapter_name].weight.data.copy_(B)

        num_filled += 1

    # 4) Copy classification head if exists (GPT2ForSequenceClassification uses 'score')
    # This is important: otherwise head stays random and ruins performance.
    if hasattr(eora_model, "score") and hasattr(optimized_model, "score"):
        try:
            eora_model.score.weight.data.copy_(optimized_model.score.weight.data)
            if hasattr(eora_model.score, "bias") and eora_model.score.bias is not None:
                eora_model.score.bias.data.copy_(optimized_model.score.bias.data)
        except Exception:
            pass

    # 5) Freeze everything (no training in EoRA branch)
    for p in eora_model.parameters():
        p.requires_grad_(False)
    eora_model.eval()

    print(f"[EoRA] Filled LoRA layers: {num_filled}")
    print(f"[EoRA] rank={rank}, alpha={alpha} (scaling={alpha/rank:.3f}), svd_on_cpu={svd_on_cpu}")
    return eora_model


@torch.no_grad()
def save_eora_adapter(eora_model: torch.nn.Module, tokenizer, output_dir: str):
    """
    Save PEFT adapter and tokenizer to output_dir/adapter
    """
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    # eora_model is a PEFT model now
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
    # just an alias wrapper for script compatibility
    return apply_eora_to_base(base_model, optimized_model, config)


def _build_calibration_texts(config: Dict[str, Any]) -> List[str]:
    """
    EoRA 需要一份“文本列表”作为 calibration_dataset。
    你可以用:
      - data/wikitext2_like/train.txt (推荐，稳定不依赖HF下载)
      - 或 datasets.load_dataset(...) 从 HF 拉
    """
    eora_cfg = config.get("eora", {})
    n = int(eora_cfg.get("calibration_num_samples", 512))

    # 1) 优先用本地文本（避免你遇到的 HF SSL 问题）
    local_txt = eora_cfg.get("calibration_local_txt", None)
    if local_txt:
        p = Path(local_txt)
        if not p.exists():
            raise FileNotFoundError(f"calibration_local_txt not found: {p}")
        texts = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)
                if len(texts) >= n:
                    break
        if len(texts) == 0:
            raise ValueError(f"calibration_local_txt is empty: {p}")
        return texts

    # 2) 否则用 HF dataset
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "glue/sst2")

    if "/" in dataset_name:
        d0, d1 = dataset_name.split("/", 1)
        ds = load_dataset(d0, d1)
        # SST-2 的文本字段
        text_key = "sentence"
    else:
        ds = load_dataset(dataset_name)
        text_key = "text"

    texts = ds["train"].select(range(n))[text_key]
    return list(texts)


def generate_eora_adapter_for_quantized(config: Dict[str, Any], save_dir: str) -> Lora:
    """
    实验3：teacher=optimized(full precision) + student=quantized(GPTQ)
    生成 EoRA adapter，并保存到 save_dir.
    """
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

    eora = Lora(
        # 注意：官方说明 path 在 generate 时是“保存路径”，load 时也是“读取路径”
        path=save_dir,
        rank=rank,
    )

    # 官方 EoRA workflow: GPTQModel.adapter.generate(...)
    GPTQModel.adapter.generate(
        adapter=eora,
        model_id_or_path=optimized_model_dir,          # teacher (FP)
        quantized_model_id_or_path=quantized_model_dir,# student (GPTQ)
        calibration_dataset=calibration_texts,
        calibration_dataset_concat_size=0,
        auto_gc=False,
    )

    return eora


def load_quantized_with_eora(config: Dict[str, Any], eora: Lora):
    """
    加载量化模型，并挂上 EoRA adapter（推理/评估用）
    """
    quantized_model_dir = config["quantized_model_dir"]
    model = GPTQModel.load(
        model_id_or_path=quantized_model_dir,
        adapter=eora,
    )
    return model
