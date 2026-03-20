# scripts/core/22_exp2_eora_lm.py
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from typing import Dict, Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.eora_utils import apply_eora_base_to_optimized  # 需要你在 eora_utils.py 中提供


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_task_type(config: Dict[str, Any]) -> None:
    if str(config.get("task_type", "")).lower() == "lm":
        config["task_type"] = "causal_lm"


def maybe_set_speed_knobs():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_set_tokenizer_pad(tokenizer):
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


@torch.no_grad()
def eval_with_teacher_lm(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader,
    device: str,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate teacher-alignment metrics for causal LM:
      - masked KL(student || teacher target distribution) on valid next-token positions
      - masked MSE(logits) on valid next-token positions

    Valid positions are determined by labels[:, 1:] != -100.
    """
    student.eval()
    teacher.eval()

    T = max(float(temperature), 1e-6)

    total_valid_tokens = 0.0
    kl_sum = 0.0
    mse_sum = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("[Exp2 EoRA-LM] batch missing labels; cannot compute masked teacher-alignment metrics.")

        # Forward without labels for logits-only teacher/student alignment
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask", None),
        }

        out_s = student(**model_inputs)
        out_t = teacher(**model_inputs)

        logits_s = out_s.logits   # [B, L, V]
        logits_t = out_t.logits   # [B, L, V]

        # causal shift: predict token t+1 from position t
        s = logits_s[:, :-1, :]
        t = logits_t[:, :-1, :]
        y = labels[:, 1:]  # [B, L-1]

        valid = (y != -100)           # [B, L-1]
        valid_f = valid.float()
        n_valid = float(valid_f.sum().item())

        if n_valid <= 0:
            continue

        # KL( teacher || student )
        s_logp = torch.log_softmax(s / T, dim=-1)
        t_prob = torch.softmax(t / T, dim=-1)
        token_kl = torch.sum(t_prob * (torch.log(t_prob + 1e-12) - s_logp), dim=-1) * (T * T)  # [B, L-1]

        # MSE on logits
        token_mse = torch.mean((s - t) ** 2, dim=-1)  # [B, L-1]

        kl_sum += float((token_kl * valid_f).sum().item())
        mse_sum += float((token_mse * valid_f).sum().item())
        total_valid_tokens += n_valid

    denom = max(total_valid_tokens, 1.0)
    return {
        "kl_to_teacher": kl_sum / denom,
        "mse_logits_to_teacher": mse_sum / denom,
        "valid_tokens": total_valid_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    normalize_task_type(cfg)

    if cfg.get("task_type") != "causal_lm":
        raise ValueError(
            f"[22_exp2_eora_lm] This script is for causal LM only. "
            f"Set task_type: causal_lm (or lm). Got: {cfg.get('task_type')}"
        )

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_dir = cfg.get("output_dir", "outputs/lm/exp2/eora_match_lm")
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    maybe_set_speed_knobs()

    optimized_model_dir = cfg.get("optimized_model_dir")
    if not optimized_model_dir:
        raise ValueError(
            "[22_exp2_eora_lm] config must provide optimized_model_dir "
            "(path to fully trained teacher model)."
        )

    # 1) tokenizer + base model
    base_model, tokenizer = load_base_model_and_tokenizer(cfg)
    maybe_set_tokenizer_pad(tokenizer)
    base_model.to(device)

    if hasattr(base_model, "config") and getattr(base_model.config, "use_cache", None) is True:
        base_model.config.use_cache = False

    # 2) dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)
    try:
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    # 3) teacher
    teacher = AutoModelForCausalLM.from_pretrained(optimized_model_dir)
    teacher.to(device)
    teacher.eval()

    if hasattr(teacher, "config") and getattr(teacher.config, "use_cache", None) is True:
        teacher.config.use_cache = False

    # 4) apply EoRA: base -> optimized (closed-form)
    eora_model = apply_eora_base_to_optimized(base_model, teacher, cfg)
    eora_model.to(device)
    eora_model.eval()

    if hasattr(eora_model, "config") and getattr(eora_model.config, "use_cache", None) is True:
        eora_model.config.use_cache = False

    # 5) evaluate task metrics using the same evaluator as Exp1
    val_task = evaluate(eora_model, val_loader, cfg)
    test_task = evaluate(eora_model, test_loader, cfg)

    # 6) evaluate teacher-alignment metrics
    temperature = float(
        cfg.get("distill", {}).get(
            "temperature",
            cfg.get("kd", {}).get("T", 1.0)
        )
    )

    val_align = eval_with_teacher_lm(
        eora_model, teacher, val_loader, device=device, temperature=temperature
    )
    test_align = eval_with_teacher_lm(
        eora_model, teacher, test_loader, device=device, temperature=temperature
    )

    val_metrics = {**val_task, **val_align}
    test_metrics = {**test_task, **test_align}

    # 7) save meta + metrics
    meta = {
        "seed": seed,
        "model_name": cfg.get("model_name"),
        "task_type": cfg.get("task_type"),
        "optimized_model_dir": optimized_model_dir,
        "eora": cfg.get("eora", {}),
        "distill": cfg.get("distill", {}),
        "data": cfg.get("data", {}),
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = {
        "seed": seed,
        "model_name": cfg.get("model_name"),
        "task_type": cfg.get("task_type"),
        "optimized_model_dir": optimized_model_dir,
        "eora": cfg.get("eora", {}),
        "distill": cfg.get("distill", {}),
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp2 EoRA-LM(output-matching eval) done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()