# scripts/core/01_train_optimized_lm.py
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

# ---- Make imports robust no matter where you run this from ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.train_utils import train_optimized


# ============================================================
# Utils
# ============================================================

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


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_yaml(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def normalize_task_type(config: Dict[str, Any]) -> None:
    # Keep backwards-compatible alias:
    #   - "lm" -> "causal_lm"
    if str(config.get("task_type", "")).lower() == "lm":
        config["task_type"] = "causal_lm"


def maybe_set_speed_knobs():
    """
    Safe speed knobs for RTX 30xx / Ampere:
      - TF32 speeds up matmul (tiny numerical differences, usually fine)
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Determinism vs speed trade-off: keep benchmark False by default
    # torch.backends.cudnn.benchmark = False


def maybe_set_tokenizer_pad(tokenizer):
    """
    GPT-2 has no pad_token by default.
    For batching/padding, it's common to set pad_token = eos_token.
    """
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def build_run_name(config: Dict[str, Any]) -> str:
    """
    Build a descriptive run name so outputs don't overwrite and are comparable.
    Example:
      lm_gpt2_len256_bs4_acc2_lr3e-5_steps3000_seed42_20260208_021533
    """
    model_name = str(config.get("model_name", "model"))
    seed = _safe_int(config.get("seed", 42), 42)

    data_cfg = config.get("data", {})
    max_length = _safe_int(data_cfg.get("max_length", 256), 256)
    batch_size = _safe_int(data_cfg.get("batch_size", 4), 4)

    train_cfg = config.get("train", {})
    lr = _safe_float(train_cfg.get("lr", 0.0), 0.0)
    grad_accum = _safe_int(train_cfg.get("grad_accum_steps", 1), 1)
    max_steps = train_cfg.get("max_train_steps", None)
    max_steps = _safe_int(max_steps, -1) if max_steps is not None else -1

    # compact lr string
    lr_str = f"{lr:.2e}".replace("+", "")
    tag = _now_tag()

    return f"lm_{model_name}_len{max_length}_bs{batch_size}_acc{grad_accum}_lr{lr_str}_steps{max_steps}_seed{seed}_{tag}"


def load_best_state_if_exists(model: torch.nn.Module, run_dir: str) -> bool:
    """
    If train_utils saved best_state_dict.pt, load it back to ensure we save best HF weights.
    """
    best_pt = os.path.join(run_dir, "best_state_dict.pt")
    if not os.path.exists(best_pt):
        return False
    try:
        state = torch.load(best_pt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        return True
    except Exception as e:
        print(f"[WARN] Failed to load best_state_dict.pt: {e}")
        return False


def estimate_tokens_per_step(config: Dict[str, Any]) -> Optional[int]:
    """
    Rough estimate for logging only:
      tokens/step ~= batch_size * max_length * grad_accum_steps
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    try:
        bs = int(data_cfg.get("batch_size", 1))
        ml = int(data_cfg.get("max_length", 1))
        acc = int(train_cfg.get("grad_accum_steps", 1))
        return bs * ml * acc
    except Exception:
        return None


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_name", type=str, default=None, help="Optional custom run name (subdir under output_dir/runs)")
    args = ap.parse_args()

    config = load_config(args.config)
    normalize_task_type(config)

    # ---- seed ----
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # ---- base output dir ----
    base_output_dir = str(config.get("output_dir", "outputs/optimized_lm"))
    ensure_dir(base_output_dir)

    # ---- run dir (avoid overwrite) ----
    run_name = args.run_name or config.get("run_name", None) or build_run_name(config)
    run_dir = os.path.join(base_output_dir, "runs", run_name)
    ensure_dir(run_dir)

    # ---- speed knobs ----
    maybe_set_speed_knobs()

    # ---- load model/tokenizer ----
    model, tokenizer = load_base_model_and_tokenizer(config)
    maybe_set_tokenizer_pad(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device:", device.type)

    # Important for causal LM training: disable KV cache
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # ---- dataloaders ----
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # Print loader sizes (train may be iterable/unknown)
    try:
        print("train batches:", len(train_loader))
    except Exception:
        print("train batches: (iterable/unknown)")
    try:
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    # ---- run metadata ----
    train_cfg = config.get("train", {})
    run_info = {
        "run_name": run_name,
        "run_dir": run_dir,
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "device": device.type,
        "timestamp": _now_tag(),
        "train": train_cfg,
        "data": config.get("data", {}),
        "tokens_per_step_est": estimate_tokens_per_step(config),
        "notes": [
            "If streaming dataset is used, max_train_steps must be set for fair comparison.",
            "This script saves model_best/ if best_state_dict.pt exists.",
        ],
    }

    save_yaml(config, os.path.join(run_dir, "config_used.yaml"))
    save_json(run_info, os.path.join(run_dir, "run_info.json"))

    print(f"[Run] base_output_dir={base_output_dir}")
    print(f"[Run] run_name={run_name}")
    print(f"[Run] run_dir={run_dir}")
    if run_info["tokens_per_step_est"] is not None:
        print(f"[Run] tokens/step(est)={run_info['tokens_per_step_est']}")

    # ---- train ----
    model = train_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        evaluate_fn=evaluate,
        output_dir=run_dir,     # IMPORTANT: best_state_dict.pt saved here
    )

    # ---- ensure best weights loaded before saving HF ----
    loaded_best = load_best_state_if_exists(model, run_dir)
    print(f"[Run] best_state_dict.pt loaded: {loaded_best}")

    # ---- final eval (on best) ----
    val_metrics = evaluate(model, val_loader, config)
    test_metrics = evaluate(model, test_loader, config)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "run_name": run_name,
        "run_dir": run_dir,
    }

    save_json(metrics, os.path.join(run_dir, "metrics.json"))

    # ---- save HF model ----
    # Always save a standard "model/" for downstream use
    model_dir = os.path.join(run_dir, "model")
    ensure_dir(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Also save a best model dir explicitly (nice for clarity)
    model_best_dir = os.path.join(run_dir, "model_best")
    ensure_dir(model_best_dir)
    model.save_pretrained(model_best_dir)
    tokenizer.save_pretrained(model_best_dir)

    # Optional: export best model to a stable alias path for downstream configs
    export_best_model_dir = config.get("export_best_model_dir", None)
    if export_best_model_dir:
        ensure_dir(export_best_model_dir)
        model.save_pretrained(export_best_model_dir)
        tokenizer.save_pretrained(export_best_model_dir)
        print("HF exported best model to stable path:", export_best_model_dir)

    print("=== Optimized LM training done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to run_dir:", run_dir)
    print("HF model saved to:", model_dir)
    print("HF best model saved to:", model_best_dir)


if __name__ == "__main__":
    main()
