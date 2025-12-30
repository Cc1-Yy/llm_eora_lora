import os
import sys
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from gptqmodel import GPTQModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import get_dataloaders


def main():

    quant_model_dir = PROJECT_ROOT / "outputs" / "quantize_optimized_sst2"
    print("== Quantized model dir ==", quant_model_dir)

    config = {
        "model_name": "gpt2",
        "task_type": "classification",
        "num_labels": 2,
        "data": {
            "dataset_name": "glue/sst2",
            "max_length": 128,
            "batch_size": 8,
            "num_workers": 0,
        },
        "seed": 42,
    }

    print("\n== Loading tokenizer from quantized model dir ==")
    tokenizer = AutoTokenizer.from_pretrained(quant_model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("pad_token_id:", tokenizer.pad_token_id)

    print("\n== Building dataloaders (SST-2) ==")
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    batch = next(iter(val_loader))
    print("\n== Val batch sample ==")
    print("keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    if "labels" in batch:
        print("labels shape:", batch["labels"].shape)
        print("labels dtype:", batch["labels"].dtype)

    print("\n== Loading quantized model with GPTQModel ==")
    model = GPTQModel.load(str(quant_model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print("Using device:", device)

    print("\n== Running a few forward passes on val loader (no loss) ==")
    num_batches_to_test = 3
    tested = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"[Batch {tested+1}] logits shape:", logits.shape)

            tested += 1
            if tested >= num_batches_to_test:
                break

    out_dir = PROJECT_ROOT / "outputs" / "smoke_test_quantized_sst2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"

    result = {
        "quant_model_dir": str(quant_model_dir),
        "num_tested_batches": tested,
        "status": "forward_ok",
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Quantized smoke test done. Result saved to {out_path}")


if __name__ == "__main__":
    main()
