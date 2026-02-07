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
from transformers import AutoModelForSequenceClassification

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.eora_utils import apply_eora_to_base, save_eora_adapter


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

    output_dir = config.get("output_dir", "outputs/exp1_eora")
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    optimized_model_dir = config.get("optimized_model_dir")
    if not optimized_model_dir:
        raise ValueError("exp1_eora.yaml must set optimized_model_dir (e.g., outputs/optimized_sst2/model)")

    # 1) load base model & tokenizer (pretrained, NOT fine-tuned)
    base_model, tokenizer = load_base_model_and_tokenizer(config)
    base_model.to(device)

    # 2) load optimized model (full fine-tuned teacher)
    # Use same head type (SequenceClassification) for SST-2
    optimized_model = AutoModelForSequenceClassification.from_pretrained(
        optimized_model_dir,
        num_labels=int(config.get("num_labels", 2)),
    )
    optimized_model.to(device)
    optimized_model.eval()

    # 3) dataloaders (same split policy as LoRA branch)
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # 4) apply EoRA (closed-form low-rank adapter) on base to approximate optimized
    eora_model = apply_eora_to_base(base_model=base_model, optimized_model=optimized_model, config=config)
    eora_model.to(device)

    # 5) evaluate
    val_metrics = evaluate(eora_model, val_loader, config)
    test_metrics = evaluate(eora_model, test_loader, config)

    # 6) save adapter + tokenizer
    adapter_dir = save_eora_adapter(eora_model, tokenizer, output_dir)

    # 7) save metrics/meta
    meta = {
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "num_labels": config.get("num_labels"),
        "data": config.get("data", {}),
        "eora": config.get("eora", {}),
        "optimized_model_dir": optimized_model_dir,
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
    }

    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = {"val": val_metrics, "test.py": test_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp1 EoRA(base->optimized, no quant) done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
