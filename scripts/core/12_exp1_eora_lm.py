# scripts/core/12_exp1_eora_lm.py
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


def _run_tag_from_cfg(config: Dict[str, Any]) -> str:
    eora_cfg = config.get("eora", {})
    r = int(eora_cfg.get("rank", 8))
    alpha = float(eora_cfg.get("alpha", r))
    ar = alpha / max(r, 1)
    # 统一格式，便于 summarize 扫描
    # e.g. r64_ar1.25
    return f"r{r}_ar{ar:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # 允许你写 task_type: lm
    if config.get("task_type") == "lm":
        config["task_type"] = "causal_lm"

    seed = int(config.get("seed", 42))
    set_seed(seed)

    base_output_dir = config.get("output_dir", "outputs/exp1_eora_lm")
    ensure_dir(base_output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    optimized_model_dir = config.get("optimized_model_dir")

    if not optimized_model_dir:
        raise ValueError("exp1_eora_lm.yaml must set optimized_model_dir (teacher), e.g. outputs/optimized_lm_small/model")

    # 每次跑一个子目录，避免覆盖
    run_tag = config.get("run_tag") or _run_tag_from_cfg(config)
    output_dir = os.path.join(base_output_dir, run_tag)
    ensure_dir(output_dir)

    # 1) load base model & tokenizer (pretrained, NOT fine-tuned)
    base_model, tokenizer = load_base_model_and_tokenizer(config)
    base_model.to(device)

    # 2) load optimized LM teacher (full fine-tuned)
    # 这里必须是 CausalLM
    optimized_model = AutoModelForCausalLM.from_pretrained(optimized_model_dir)
    optimized_model.to(device)
    optimized_model.eval()

    # 3) dataloaders (LM pipeline: train_corpus/eval_corpus/test_corpus)
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    try:
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    # 4) apply EoRA (closed-form low-rank adapter) on base to approximate optimized
    # 注意：EoRA 分支“不训练”，只是做一次 closed-form 权重注入
    eora_model = apply_eora_to_base(base_model=base_model, optimized_model=optimized_model, config=config)
    eora_model.to(device)

    # 5) evaluate (LM => loss + ppl)
    val_metrics = evaluate(eora_model, val_loader, config)
    test_metrics = evaluate(eora_model, test_loader, config)

    # 6) save adapter + tokenizer
    adapter_dir = save_eora_adapter(eora_model, tokenizer, output_dir)

    # 7) save meta + metrics
    meta = {
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "data": config.get("data", {}),
        "eora": config.get("eora", {}),
        "optimized_model_dir": optimized_model_dir,
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
        "run_tag": run_tag,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = {"val": val_metrics, "test": test_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp1 EoRA-LM(base->optimized, no quant) done ===")
    print("Run tag:", run_tag)
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
