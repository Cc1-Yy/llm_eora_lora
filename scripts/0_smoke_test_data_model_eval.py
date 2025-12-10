import os
import sys
import json
from pathlib import Path

import torch

# ---- Make imports robust no matter where you run this from ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate


def main():
    # 1) 最小分类配置（强烈建议先用它）
    config = {
        "model_name": "gpt2",  # 你也可以换 roberta-base
        "task_type": "classification",
        "num_labels": 2,
        "data": {
            "dataset_name": "glue/sst2",
            "max_length": 128,
            "batch_size": 4,
            "num_workers": 0,  # Windows 先用 0，稳定后再尝试 2/4
        },
        "seed": 42,
    }

    print("== Loading model/tokenizer ==")
    model, tokenizer = load_base_model_and_tokenizer(config)
    print("model loaded:", model.__class__.__name__)
    print("pad_token_id:", tokenizer.pad_token_id)

    print("\n== Building dataloaders ==")
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    # 2) 看一个 batch 结构
    batch = next(iter(train_loader))
    print("\n== Batch keys ==")
    print(batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("labels shape:", batch["labels"].shape)
    print("labels dtype:", batch["labels"].dtype)

    # 3) 放到设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n== Device ==")
    print("Using device:", device)
    model.to(device)
    print("Model param device:", next(model.parameters()).device)

    print("\n== Running evaluate on val ==")
    metrics = evaluate(model, val_loader, config)
    print("metrics:", metrics)

    # 4) 保存一份输出（可选）
    out_path = PROJECT_ROOT / "outputs" / "smoke_test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Smoke test done. Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
