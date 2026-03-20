# scripts/core/12_exp1_eora.py
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, List

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _collect_score_leaf_modules(score_module: torch.nn.Module) -> List[torch.nn.Module]:
    """
    Collect all possible leaf modules that may actually hold the classification head weights.

    Supports:
      1) plain nn.Linear-like module with .weight
      2) PEFT ModulesToSaveWrapper.original_module
      3) PEFT ModulesToSaveWrapper.modules_to_save[...]
    """
    leaves: List[torch.nn.Module] = []

    if score_module is None:
        return leaves

    # Case 1: plain module itself has weights
    if hasattr(score_module, "weight") and getattr(score_module, "weight", None) is not None:
        leaves.append(score_module)

    # Case 2: wrapped original module
    if hasattr(score_module, "original_module"):
        orig = getattr(score_module, "original_module")
        if orig is not None and hasattr(orig, "weight") and getattr(orig, "weight", None) is not None:
            leaves.append(orig)

    # Case 3: modules_to_save (often a ModuleDict)
    if hasattr(score_module, "modules_to_save"):
        mts = getattr(score_module, "modules_to_save")
        if mts is not None:
            if hasattr(mts, "values"):
                for mod in mts.values():
                    if hasattr(mod, "weight") and getattr(mod, "weight", None) is not None:
                        leaves.append(mod)
            elif isinstance(mts, dict):
                for mod in mts.values():
                    if hasattr(mod, "weight") and getattr(mod, "weight", None) is not None:
                        leaves.append(mod)

    # de-duplicate by object id
    uniq = []
    seen = set()
    for m in leaves:
        if id(m) not in seen:
            uniq.append(m)
            seen.add(id(m))
    return uniq


def maybe_copy_classification_head(eora_model: torch.nn.Module, optimized_model: torch.nn.Module) -> bool:
    """
    For sequence classification, copy the trained task head from the optimized teacher
    to avoid evaluating EoRA with a freshly-randomized classification head.

    Returns True if at least one destination head is successfully copied.
    """
    if not hasattr(eora_model, "score") or not hasattr(optimized_model, "score"):
        print("[Info] score head not found on one side; skip head copy.")
        return False

    src_candidates = _collect_score_leaf_modules(optimized_model.score)
    dst_candidates = _collect_score_leaf_modules(eora_model.score)

    if not src_candidates:
        print("[Warn] Teacher score head exists but no weight-bearing submodule was found.")
        return False
    if not dst_candidates:
        print("[Warn] EoRA score head exists but no weight-bearing submodule was found.")
        return False

    src = src_candidates[0]  # teacher should normally have a single usable source
    copied_any = False

    try:
        with torch.no_grad():
            for dst in dst_candidates:
                if hasattr(dst, "weight") and dst.weight is not None:
                    dst.weight.copy_(src.weight.detach().to(dst.weight.device, dtype=dst.weight.dtype))
                    copied_any = True

                src_bias = getattr(src, "bias", None)
                dst_bias = getattr(dst, "bias", None)
                if src_bias is not None and dst_bias is not None:
                    dst_bias.copy_(src_bias.detach().to(dst_bias.device, dtype=dst_bias.dtype))

        if copied_any:
            return True

        print("[Warn] Head copy attempted but no destination weight was matched.")
        return False

    except Exception as e:
        print(f"[Warn] Failed to copy classification head from teacher: {e}")
        return False


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
    optimized_model = AutoModelForSequenceClassification.from_pretrained(
        optimized_model_dir,
        num_labels=int(config.get("num_labels", 2)),
    )
    optimized_model.to(device)
    optimized_model.eval()

    # 3) dataloaders (same split policy as LoRA branch)
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # 4) apply EoRA (closed-form low-rank adapter) on base to approximate optimized
    eora_model = apply_eora_to_base(
        base_model=base_model,
        optimized_model=optimized_model,
        config=config,
    )
    eora_model.to(device)
    eora_model.eval()

    # 4.1) copy trained classification head from teacher for fair comparison
    copied_head = maybe_copy_classification_head(eora_model, optimized_model)
    if copied_head:
        print("[EoRA] Copied classification head from optimized model.")

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
        "copied_classification_head": copied_head,
    }

    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    metrics = {"val": val_metrics, "test": test_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== Exp1 EoRA(base->optimized, no quant) done ===")
    print("Copied classification head:", copied_head)
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()