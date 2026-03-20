# scripts/utils/make_lora_teacher_lm.py
from __future__ import annotations

import os
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="gpt2")
    ap.add_argument("--adapter_dir", type=str, required=True, help="Path to LoRA adapter (e.g. .../adapter_best)")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to save merged teacher model")
    ap.add_argument("--offline", action="store_true", help="Set HF offline env to avoid hub timeouts")
    args = ap.parse_args()

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tokenizer.pad_token_id

    # Load adapter on top of base
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    # Merge LoRA into base weights -> a normal HF model
    merged = model.merge_and_unload()

    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"[OK] merged teacher saved to: {out_dir}")


if __name__ == "__main__":
    main()