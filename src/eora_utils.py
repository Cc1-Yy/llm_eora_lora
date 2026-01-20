from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import os

import torch
from peft import LoraConfig, get_peft_model


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
    prefix = "base_model.model."
    adapter_name = "default"

    num_filled = 0
    for name, module in eora_model.named_modules():
        # We only care modules that have LoRA params
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        # Resolve corresponding original layer name
        if name.startswith(prefix):
            opt_name = name[len(prefix):]
        else:
            opt_name = name

        if opt_name not in opt_modules:
            # Some wrappers introduce extra nodes; skip if not found
            continue

        opt_layer = opt_modules[opt_name]

        # PEFT LoRA layers keep the original layer in module.base_layer
        base_layer = getattr(module, "base_layer", None)
        if base_layer is None or not hasattr(base_layer, "weight") or not hasattr(opt_layer, "weight"):
            continue

        Wb = base_layer.weight.data
        Wo = opt_layer.weight.data

        # We want delta in "effective" [out, in] orientation for LoRA:
        # - For nn.Linear: weight is [out, in] already.
        # - For GPT2 Conv1D: weight is stored as [in, out], but PEFT sets fan_in_fan_out=True.
        #   Effective [out, in] is W^T.
        if _is_gpt2_conv1d(base_layer):
            delta = (Wo.T - Wb.T).contiguous()   # [out, in]
        else:
            delta = (Wo - Wb).contiguous()       # [out, in]

        # SVD low-rank: delta ≈ B @ A
        B, A = _svd_low_rank(delta, rank=rank, svd_on_cpu=svd_on_cpu)

        # PEFT uses nn.Linear for lora_A and lora_B; their weights are:
        #   lora_A.weight: [r, in]
        #   lora_B.weight: [out, r]
        # Effective delta in [out,in] is (lora_B.weight @ lora_A.weight)
        # BUT note: nn.Linear computes x @ W^T; PEFT already handles that internally.
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
