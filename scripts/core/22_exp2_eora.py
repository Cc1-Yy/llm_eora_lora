from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eora_utils import apply_eora_base_to_optimized  # 你需要在 eora_utils.py 里提供这个函数


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def eval_with_teacher(student, teacher, dataloader, device, temperature: float = 1.0) -> Dict[str, float]:
    student.eval()
    teacher.eval()

    total = 0
    correct = 0
    ce_sum = 0.0
    kl_sum = 0.0
    mse_sum = 0.0

    for batch in dataloader:
        labels = batch.get("labels", None)
        batch_s = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out_s = student(**batch_s)
        logits_s = out_s.logits

        out_t = teacher(**batch_s)
        logits_t = out_t.logits

        if labels is not None:
            y = labels.to(device)
            ce = F.cross_entropy(logits_s, y)
            ce_sum += ce.item() * y.size(0)
            pred = torch.argmax(logits_s, dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        else:
            total += logits_s.size(0)

        s_logp = F.log_softmax(logits_s / temperature, dim=-1)
        t_prob = F.softmax(logits_t / temperature, dim=-1)
        kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature ** 2)
        mse = F.mse_loss(logits_s, logits_t)

        kl_sum += kl.item()
        mse_sum += mse.item()

    acc = (correct / total) if total > 0 else 0.0
    ce_avg = (ce_sum / total) if total > 0 else None
    kl_avg = kl_sum / max(len(dataloader), 1)
    mse_avg = mse_sum / max(len(dataloader), 1)

    return {
        "accuracy": float(acc),
        "ce_loss": float(ce_avg) if ce_avg is not None else None,
        "kl_to_teacher": float(kl_avg),
        "mse_logits_to_teacher": float(mse_avg),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg.get("output_dir", "outputs/exp2_eora_match")
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # tokenizer + dataloaders
    base_model, tokenizer = load_base_model_and_tokenizer(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)

    # teacher
    teacher_dir = cfg["optimized_model_dir"]
    teacher_cfg = dict(cfg)
    teacher_cfg["model_name"] = teacher_dir
    teacher, _ = load_base_model_and_tokenizer(teacher_cfg)

    # apply eora: base -> optimized (closed-form)
    eora_model = apply_eora_base_to_optimized(base_model, teacher, cfg)

    eora_model.to(device)
    teacher.to(device)

    temperature = float(cfg.get("distill", {}).get("temperature", 1.0))

    val_metrics = eval_with_teacher(eora_model, teacher, val_loader, device, temperature=temperature)
    test_metrics = eval_with_teacher(eora_model, teacher, test_loader, device, temperature=temperature)

    metrics = {
        "seed": cfg.get("seed", 42),
        "model_name": cfg.get("model_name"),
        "optimized_model_dir": cfg.get("optimized_model_dir"),
        "eora": cfg.get("eora", {}),
        "distill": cfg.get("distill", {}),
        "val": val_metrics,
        "test.py": test_metrics,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp2 EoRA(output-matching eval) done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
