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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.train_utils import train_optimized


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

    output_dir = config.get("output_dir", "outputs/optimized_minimal")
    ensure_dir(output_dir)

    # 1) load model & tokenizer
    model, tokenizer = load_base_model_and_tokenizer(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2) dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # 3) train optimized
    model = train_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        evaluate_fn=evaluate,
    )

    # 4) final eval
    val_metrics = evaluate(model, val_loader, config)
    test_metrics = evaluate(model, test_loader, config)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "data": config.get("data", {}),
        "train": config.get("train", {}),
    }

    # 5) save model
    model_dir = os.path.join(output_dir, "model")
    ensure_dir(model_dir)

    # 用 HF 标准方式保存（之后 EoRA/LoRA 都好对接）
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # 6) save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Optimized training done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
