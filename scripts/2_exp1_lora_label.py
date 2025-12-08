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
from torch.optim import AdamW
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    output_dir = config.get("output_dir", "outputs/exp1_lora_label")
    ensure_dir(output_dir)

    # 1) load base model & tokenizer
    base_model, tokenizer = load_base_model_and_tokenizer(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)

    # 2) dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # 3) add LoRA
    model = add_lora_to_model(base_model, config)
    model.to(device)

    print_trainable_params(model)

    # 4) optimizer (只优化 requires_grad 参数)
    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    num_epochs = int(train_cfg.get("num_epochs", 3))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=lr, weight_decay=weight_decay)

    # 5) train loop
    best_metric = -1e9
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        loop = tqdm(train_loader, desc=f"[Exp1 LoRA-Label] Epoch {epoch}/{num_epochs}")

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)

            optimizer.step()

            bs = batch["input_ids"].size(0)
            total_loss += loss.item() * bs
            total_examples += bs

            loop.set_postfix(loss=loss.item())

        # epoch end eval
        val_metrics = evaluate(model, val_loader, config)

        # 分类任务用 accuracy 选最优
        score = val_metrics.get("accuracy", -1.0)

        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        train_loss = total_loss / max(total_examples, 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val={val_metrics}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) final eval
    val_metrics = evaluate(model, val_loader, config)
    test_metrics = evaluate(model, test_loader, config)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "data": config.get("data", {}),
        "lora": config.get("lora", {}),
        "train": config.get("train", {}),
    }

    # 7) save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    ensure_dir(adapter_dir)

    # PEFT 标准保存 adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # 8) save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp1 LoRA(label) done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
