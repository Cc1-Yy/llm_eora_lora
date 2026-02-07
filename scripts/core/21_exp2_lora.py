from __future__ import annotations

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, Tuple

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
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def teacher_forward(teacher_model, batch, device):
    # teacher 不需要梯度
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
        # KLDivLoss expects input=log-prob, target=prob
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

        # task metrics（如果有 labels）
        if labels is not None:
            y = labels.to(device)
            ce = F.cross_entropy(logits_s, y)
            ce_sum += ce.item() * y.size(0)

            pred = torch.argmax(logits_s, dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        else:
            # 没 label 的情况也支持
            bs = logits_s.size(0)
            total += bs

        # gap metrics
        s_logp = F.log_softmax(logits_s / temperature, dim=-1)
        t_prob = F.softmax(logits_t / temperature, dim=-1)
        kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature ** 2)
        mse = F.mse_loss(logits_s, logits_t)

        # 这里按 batch mean 累加（再平均）
        kl_sum += kl.item()
        mse_sum += mse.item()

    acc = (correct / total) if total > 0 else 0.0
    ce_avg = (ce_sum / total) if total > 0 else None
    kl_avg = kl_sum / max(len(dataloader), 1)
    mse_avg = mse_sum / max(len(dataloader), 1)

    out = {
        "accuracy": float(acc),
        "ce_loss": float(ce_avg) if ce_avg is not None else None,
        "kl_to_teacher": float(kl_avg),
        "mse_logits_to_teacher": float(mse_avg),
    }
    return out


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

    # 1) tokenizer + dataloaders（同 Exp1）
    base_model, tokenizer = load_base_model_and_tokenizer(cfg)
    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)

    # 2) teacher（optimized）
    teacher_dir = cfg["optimized_model_dir"]
    teacher_cfg = dict(cfg)
    teacher_cfg["model_name"] = teacher_dir  # 让 model_utils 从本地目录加载
    teacher, _ = load_base_model_and_tokenizer(teacher_cfg)

    base_model.to(device)
    teacher.to(device)
    teacher.eval()

    # 3) student = base + LoRA
    student = add_lora_to_model(base_model, cfg)
    student.to(device)
    print_trainable_params(student)

    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    num_epochs = int(train_cfg.get("num_epochs", 3))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    distill_cfg = cfg.get("distill", {})
    temperature = float(distill_cfg.get("temperature", 1.0))
    loss_type = str(distill_cfg.get("loss_type", "kl"))  # kl / mse / kl+mse

    optim_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=lr, weight_decay=weight_decay)

    best_metric = -1e9
    best_state = None

    # 4) training (match teacher)
    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        total_examples = 0

        loop = tqdm(train_loader, desc=f"[Exp2 LoRA] Epoch {epoch}/{num_epochs}")
        for batch in loop:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            # teacher logits
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
            loop.set_postfix(loss=loss.item())

        # epoch end eval（以 val 的 kl_to_teacher 作为选最优：越小越好）
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
        "val": val_metrics,
        "test.py": test_metrics,
    }

    # 6) save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    ensure_dir(adapter_dir)
    student.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp2 LoRA done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
