# scripts/utils/compare_base_vs_optimized.py

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 保证能 import src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import get_dataloaders
from src.eval_utils import evaluate


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="比如 configs/optimized_sst2.yaml，用来确定数据和任务类型",
    )
    parser.add_argument(
        "--optimized_model_dir",
        type=str,
        default="outputs/optimized_sst2/model",
        help="已经 full fine-tune 好的老师模型目录",
    )
    args = parser.parse_args()

    # 1) 读取配置（用 optimized_sst2.yaml 就行）
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2) 加载 tokenizer（用 optimized 模型目录里的 tokenizer，确保一致）
    tok_src = args.optimized_model_dir or config.get("model_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) 构建 dataloaders（和你训练 optimized 时完全一致）
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    print(f"[Data] val batches:  {len(val_loader)}")
    print(f"[Data] test batches: {len(test_loader)}")

    # 4) 加载 Base model（预训练的 gpt2 分类头，未在 SST-2 上训练）
    base_model_name = config.get("model_name", "gpt2")
    print(f"\n== Loading BASE model: {base_model_name} ==")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=int(config.get("num_labels", 2)),
    )
    base_model.to(device)
    base_model.eval()

    print("Evaluating BASE model ...")
    base_val_metrics = evaluate(base_model, val_loader, config)
    base_test_metrics = evaluate(base_model, test_loader, config)
    print("BASE val:", base_val_metrics)
    print("BASE test:", base_test_metrics)

    # 5) 加载 Optimized model（你训练好的老师模型）
    print(f"\n== Loading OPTIMIZED model from: {args.optimized_model_dir} ==")
    opt_model = AutoModelForSequenceClassification.from_pretrained(
        args.optimized_model_dir
    )
    opt_model.to(device)
    opt_model.eval()

    print("Evaluating OPTIMIZED model ...")
    opt_val_metrics = evaluate(opt_model, val_loader, config)
    opt_test_metrics = evaluate(opt_model, test_loader, config)
    print("OPTIMIZED val:", opt_val_metrics)
    print("OPTIMIZED test:", opt_test_metrics)

    # 6) 汇总结果并保存成 json，方便后面画图/做表
    results = {
        "config_path": args.config,
        "optimized_model_dir": args.optimized_model_dir,
        "base": {
            "val": base_val_metrics,
            "test": base_test_metrics,
        },
        "optimized": {
            "val": opt_val_metrics,
            "test": opt_test_metrics,
        },
    }

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "compare_base_vs_optimized")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sst2_base_vs_optimized.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Comparison done. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
