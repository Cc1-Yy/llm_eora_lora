# scripts/core/32_exp3_eora.py
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
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.eora_utils import generate_eora_adapter_for_quantized, load_quantized_with_eora


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


class _GPTQModelWrapper(torch.nn.Module):
    """
    兼容你现有 evaluate(model, dataloader, config) 的最小包装：
    - 有的 GPTQModel 返回对象里模型在 .model
    - 有的直接可 forward
    """
    def __init__(self, gptq_obj):
        super().__init__()
        self.gptq_obj = gptq_obj
        self.inner = getattr(gptq_obj, "model", gptq_obj)

    def forward(self, **batch):
        return self.inner(**batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    output_root = config.get("output_dir", "outputs/exp3_eora")
    ensure_dir(output_root)

    # --- derive run subdir name: r{rank}_ar{alpha/r} ---
    eora_cfg = config.get("eora", {})
    rank = int(eora_cfg.get("rank", 16))
    alpha = int(eora_cfg.get("alpha", rank))
    ar = alpha / max(rank, 1)
    run_dir = os.path.join(output_root, f"r{rank}_ar{ar:g}")
    ensure_dir(run_dir)

    optimized_model_dir = config.get("optimized_model_dir")
    quantized_model_dir = config.get("quantized_model_dir")
    if not optimized_model_dir or not quantized_model_dir:
        raise ValueError("Need optimized_model_dir and quantized_model_dir for Exp3-EoRA.")

    # tokenizer：用 quantized 目录即可（你那里面 tokenizers 都齐）
    tok_src = quantized_model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # 1) baseline: quantized without adapter (可选但强烈建议记录)
    do_baseline = bool(config.get("do_quant_baseline_eval", True))
    baseline_metrics = None
    if do_baseline:
        from gptqmodel import GPTQModel
        q_obj = GPTQModel.load(model_id_or_path=quantized_model_dir)
        q_model = _GPTQModelWrapper(q_obj).to("cuda" if torch.cuda.is_available() else "cpu")
        baseline_metrics = evaluate(q_model, test_loader, config)

    # 2) generate EoRA adapter
    adapter_save_dir = os.path.join(run_dir, "adapter")
    eora = generate_eora_adapter_for_quantized(config, save_dir=adapter_save_dir)

    # 3) load quantized + EoRA
    eora_obj = load_quantized_with_eora(config, eora)
    eora_model = _GPTQModelWrapper(eora_obj).to("cuda" if torch.cuda.is_available() else "cpu")

    # 4) eval
    val_metrics = evaluate(eora_model, val_loader, config)
    test_metrics = evaluate(eora_model, test_loader, config)

    # save
    meta = {
        "seed": seed,
        "task_type": config.get("task_type"),
        "model_name": config.get("model_name"),
        "optimized_model_dir": optimized_model_dir,
        "quantized_model_dir": quantized_model_dir,
        "eora": config.get("eora", {}),
        "baseline_quant_test": baseline_metrics,
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = {"val": val_metrics, "test.py": test_metrics}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp3 EoRA(quant recovery) done ===")
    if baseline_metrics is not None:
        print("Quantized baseline test.py:", baseline_metrics)
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
