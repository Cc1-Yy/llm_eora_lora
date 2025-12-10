# scripts/3_exp1_eora_calc.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
import random
from typing import Dict, Any
from transformers import AutoTokenizer

import numpy as np
import torch
import yaml

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.eora_utils import generate_eora_adapter, load_quant_with_eora


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

    output_dir = config.get("output_dir", "outputs/exp1_eora_calc")
    ensure_dir(output_dir)

    # ---- 0) 基本字段检查 ----
    optimized_model_dir = config.get("optimized_model_dir")
    quantized_model_dir = config.get("quantized_model_dir")
    if not optimized_model_dir:
        raise ValueError("config['optimized_model_dir'] is required.")
    if not quantized_model_dir:
        raise ValueError("config['quantized_model_dir'] is required.")

    # ---- 1) 仅加载 tokenizer ----
    # 优先用 optimized 模型目录的 tokenizer
    tok_src = config.get("optimized_model_dir") or config.get("model_name")
    if not tok_src:
        raise ValueError("Need optimized_model_dir or model_name to load tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2) dataloaders（用于评估）----
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # ---- 3) 生成 EoRA adapter（官方路径）----
    # 这一步会使用:
    #   optimized_model_dir (full precision)
    #   quantized_model_dir (gptq)
    #   calibration texts
    eora = generate_eora_adapter(config)

    # ---- 4) 加载“量化 + EoRA”模型 ----
    # GPTQModel.load(..., adapter=eora)
    eora_model = load_quant_with_eora(config, eora)

    # ---- 5) 评估 ----
    # 你现有 evaluate 假设是 transformers 风格的 forward 输出
    # 如果未来你发现 GPTQModel 对分类头不兼容，
    # 这一步可能需要你换成更贴合 GPTQModel 的生成/LM 评估方式。
    val_metrics = evaluate(eora_model, val_loader, config)
    test_metrics = evaluate(eora_model, test_loader, config)

    # ---- 6) 保存 meta 信息 ----
    meta = {
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "data": config.get("data", {}),
        "eora": config.get("eora", {}),
        "optimized_model_dir": optimized_model_dir,
        "quantized_model_dir": quantized_model_dir,
        "output_dir": output_dir,
        # 官方示例里 adapter 会保存在 quantized_model_dir 下
        "eora_adapter_path_hint": os.path.join(
            quantized_model_dir,
            f"eora_rank{int(config.get('eora', {}).get('rank', 8))}"
        ),
    }

    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 7) 保存 metrics ----
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp1 EoRA(calc) done (GPTQModel workflow) ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
