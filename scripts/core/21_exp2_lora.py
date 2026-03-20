# scripts/core/21_exp2_lora.py
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from torch.optim import AdamW

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.lora_utils import add_lora_to_model, print_trainable_params


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


def _collect_score_leaf_modules(score_module: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Collect all possible leaf modules that may actually hold the classification head weights.

    Supports:
      1) plain nn.Linear-like module with .weight
      2) PEFT ModulesToSaveWrapper.original_module
      3) PEFT ModulesToSaveWrapper.modules_to_save[...]
    """
    leaves: List[torch.nn.Module] = []

    if score_module is None:
        return leaves

    # Case 1: plain module itself has weights
    if hasattr(score_module, "weight") and getattr(score_module, "weight", None) is not None:
        leaves.append(score_module)

    # Case 2: wrapped original module
    if hasattr(score_module, "original_module"):
        orig = getattr(score_module, "original_module")
        if orig is not None and hasattr(orig, "weight") and getattr(orig, "weight", None) is not None:
            leaves.append(orig)

    # Case 3: modules_to_save (often a ModuleDict)
    if hasattr(score_module, "modules_to_save"):
        mts = getattr(score_module, "modules_to_save")
        if mts is not None:
            if hasattr(mts, "values"):
                for mod in mts.values():
                    if hasattr(mod, "weight") and getattr(mod, "weight", None) is not None:
                        leaves.append(mod)
            elif isinstance(mts, dict):
                for mod in mts.values():
                    if hasattr(mod, "weight") and getattr(mod, "weight", None) is not None:
                        leaves.append(mod)

    # de-duplicate by object id
    uniq = []
    seen = set()
    for m in leaves:
        if id(m) not in seen:
            uniq.append(m)
            seen.add(id(m))
    return uniq


def maybe_copy_and_freeze_classification_head(student: torch.nn.Module, teacher: torch.nn.Module) -> bool:
    """
    For sequence classification Exp2, copy the trained task head from teacher
    and freeze it, so the comparison focuses on low-rank approximation rather
    than relearning a random classification head.
    """
    if not hasattr(student, "score") or not hasattr(teacher, "score"):
        print("[Info] score head not found on one side; skip head copy.")
        return False

    src_candidates = _collect_score_leaf_modules(teacher.score)
    dst_candidates = _collect_score_leaf_modules(student.score)

    if not src_candidates:
        print("[Warn] Teacher score head exists but no weight-bearing submodule was found.")
        return False
    if not dst_candidates:
        print("[Warn] Student score head exists but no weight-bearing submodule was found.")
        return False

    src = src_candidates[0]
    copied_any = False

    try:
        with torch.no_grad():
            for dst in dst_candidates:
                if hasattr(dst, "weight") and dst.weight is not None:
                    dst.weight.copy_(src.weight.detach().to(dst.weight.device, dtype=dst.weight.dtype))
                    copied_any = True

                src_bias = getattr(src, "bias", None)
                dst_bias = getattr(dst, "bias", None)
                if src_bias is not None and dst_bias is not None:
                    dst_bias.copy_(src_bias.detach().to(dst_bias.device, dtype=dst_bias.dtype))

        if not copied_any:
            print("[Warn] Head copy attempted but no destination weight was matched.")
            return False

        # Freeze the whole wrapped score head
        for p in student.score.parameters():
            p.requires_grad = False

        return True

    except Exception as e:
        print(f"[Warn] Failed to copy/freeze classification head from teacher: {e}")
        return False


@torch.no_grad()
def teacher_forward(teacher_model, batch, device):
    batch_t = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    out = teacher_model(**batch_t)
    return out.logits  # [B, C]


def distill_loss(student_logits, teacher_logits, temperature: float = 1.0, loss_type: str = "kl") -> torch.Tensor:
    """
    loss_type:
      - "kl": KL(student || teacher) using softmax at temperature
      - "mse": MSE(student_logits, teacher_logits)
      - "kl+mse": both
    """
    if loss_type not in {"kl", "mse", "kl+mse"}:
        raise ValueError(f"Unknown loss_type={loss_type}")

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    loss = 0.0

    if loss_type in {"kl", "kl+mse"}:
        s_logp = F.log_softmax(student_logits / temperature, dim=-1)
        t_prob = F.softmax(teacher_logits / temperature, dim=-1)
        kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature ** 2)
        loss = loss + kl

    if loss_type in {"mse", "kl+mse"}:
        mse = F.mse_loss(student_logits, teacher_logits)
        loss = loss + mse

    return loss


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

        logits_t = teacher_forward(teacher, batch, device)

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
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_dir = cfg.get("output_dir", "outputs/exp2_lora")
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) tokenizer + dataloaders
    base_model, tokenizer = load_base_model_and_tokenizer(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)

    # 2) teacher（optimized）
    teacher_dir = cfg["optimized_model_dir"]
    teacher_cfg = dict(cfg)
    teacher_cfg["model_name"] = teacher_dir
    teacher, _ = load_base_model_and_tokenizer(teacher_cfg)

    base_model.to(device)
    teacher.to(device)
    teacher.eval()

    # 3) student = base + LoRA
    student = add_lora_to_model(base_model, cfg)
    student.to(device)

    # 3.1) copy and freeze classification head for fair Exp2 comparison
    copied_head = maybe_copy_and_freeze_classification_head(student, teacher)
    if copied_head:
        print("[Exp2 LoRA] Copied classification head from teacher and froze it.")

    print_trainable_params(student)

    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    num_epochs = int(train_cfg.get("num_epochs", 3))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    distill_cfg = cfg.get("distill", {})
    temperature = float(distill_cfg.get("temperature", 1.0))
    loss_type = str(distill_cfg.get("loss_type", "kl"))

    optim_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=lr, weight_decay=weight_decay)

    best_metric = -1e9
    best_state = None

    # 4) training (match teacher)
    for epoch in range(1, num_epochs + 1):
        student.train()

        # keep frozen classification head in eval mode
        if copied_head and hasattr(student, "score"):
            student.score.eval()

        total_loss = 0.0
        total_examples = 0

        loop = tqdm(train_loader, desc=f"[Exp2 LoRA] Epoch {epoch}/{num_epochs}")
        for batch in loop:
            batch_dev = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                t_logits = teacher_forward(teacher, batch, device)

            out_s = student(**{k: v for k, v in batch_dev.items() if k != "labels"})
            s_logits = out_s.logits

            loss = distill_loss(s_logits, t_logits, temperature=temperature, loss_type=loss_type)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)
            optimizer.step()

            bs = s_logits.size(0)
            total_loss += loss.item() * bs
            total_examples += bs
            loop.set_postfix(loss=float(loss.item()))

        val_metrics = eval_with_teacher(student, teacher, val_loader, device, temperature=temperature)
        score = -val_metrics["kl_to_teacher"]  # minimize KL

        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        train_loss = total_loss / max(total_examples, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val={val_metrics}")

    if best_state is not None:
        student.load_state_dict(best_state)

    # 5) final eval
    val_metrics = eval_with_teacher(student, teacher, val_loader, device, temperature=temperature)
    test_metrics = eval_with_teacher(student, teacher, test_loader, device, temperature=temperature)

    metrics = {
        "seed": seed,
        "model_name": cfg.get("model_name"),
        "optimized_model_dir": cfg.get("optimized_model_dir"),
        "lora": cfg.get("lora", {}),
        "train": cfg.get("train", {}),
        "distill": cfg.get("distill", {}),
        "copied_classification_head": copied_head,
        "val": val_metrics,
        "test": test_metrics,
    }

    # 6) save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    ensure_dir(adapter_dir)
    student.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp2 LoRA done ===")
    print("Copied classification head:", copied_head)
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()