# scripts/core/11_exp1_lora_lm.py
from __future__ import annotations

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

# ---- Make imports robust no matter where you run this from ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.lora_utils import add_lora_to_model, print_trainable_params


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


def normalize_task_type(config: Dict[str, Any]) -> None:
    """
    Backwards-compatible alias:
      - "lm" -> "causal_lm"
    """
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


def maybe_set_tokenizer_pad(tokenizer):
    """
    GPT-2 has no pad_token by default. For padding/collation, set pad_token=eos_token.
    """
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


def _pretty_ar(alpha: float, rank: int) -> str:
    ar = float(alpha) / max(int(rank), 1)
    if abs(ar - round(ar)) < 1e-9:
        return f"{ar:.0f}"
    return f"{ar:.2f}".rstrip("0").rstrip(".")


def _get_run_name(config: Dict[str, Any]) -> str:
    """
    Default naming:
      r{rank}_ar{alpha/r}
    Example:
      rank=8, alpha=8 -> r8_ar1
      rank=16, alpha=20 -> r16_ar1.25
    """
    if config.get("run_name"):
        return str(config["run_name"])

    lcfg = config.get("lora", {})
    r = lcfg.get("rank", None)
    a = lcfg.get("alpha", None)
    if r is None or a is None:
        return "run"

    try:
        r = int(r)
        a = float(a)
        ar_str = _pretty_ar(a, r)
        return f"r{r}_ar{ar_str}"
    except Exception:
        return "run"


def _make_amp_helpers(device: torch.device, use_amp: bool):
    """
    Use the newer torch.amp API to avoid deprecation warnings.
    Returns: scaler, autocast_ctx (callable -> context manager)
    """
    use_amp = bool(use_amp)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def _autocast():
            return torch.amp.autocast("cuda", enabled=use_amp)

        return scaler, _autocast

    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def _autocast():
        return torch.autocast("cpu", enabled=False)

    return scaler, _autocast


def _estimate_tokens_per_step(config: Dict[str, Any]) -> int:
    """
    Rough token throughput estimate:
      tokens/optimizer_step ~= batch_size * max_length * grad_accum_steps
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    bs = int(data_cfg.get("batch_size", 1))
    max_len = int(data_cfg.get("max_length", 128))
    acc = int(train_cfg.get("grad_accum_steps", 1))
    acc = max(1, acc)
    return bs * max_len * acc


def _get_train_knobs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unify config keys:
      log_every_steps / log_every
      eval_every_steps / eval_every
    """
    train_cfg = config.get("train", {})
    out = dict(train_cfg) if isinstance(train_cfg, dict) else {}

    out["log_every_steps"] = int(train_cfg.get("log_every_steps", train_cfg.get("log_every", 50)))
    out["eval_every_steps"] = int(train_cfg.get("eval_every_steps", train_cfg.get("eval_every", 1000)))

    # standardize
    out["grad_accum_steps"] = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    out["num_epochs"] = max(1, int(train_cfg.get("num_epochs", 1)))

    # max_train_steps is optimizer steps (global_step)
    mts = train_cfg.get("max_train_steps", None)
    out["max_train_steps"] = int(mts) if mts is not None else None

    out["warmup_ratio"] = float(train_cfg.get("warmup_ratio", 0.0))
    out["use_amp"] = bool(train_cfg.get("use_amp", True))
    out["lr"] = float(train_cfg.get("lr", 2e-5))
    out["weight_decay"] = float(train_cfg.get("weight_decay", 0.0))
    out["grad_clip"] = float(train_cfg.get("grad_clip", 1.0))
    out["scheduler"] = str(train_cfg.get("scheduler", "linear")).lower()

    return out


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    normalize_task_type(config)

    if config.get("task_type") != "causal_lm":
        raise ValueError(
            f"[11_exp1_lora_lm] This script is for causal LM only. "
            f"Set task_type: causal_lm (or lm). Got: {config.get('task_type')}"
        )

    seed = int(config.get("seed", 42))
    set_seed(seed)

    base_output_dir = str(config.get("output_dir", "outputs/exp1_lora_lm"))
    ensure_dir(base_output_dir)

    # Runs folder for clean sweeps
    runs_dir = os.path.join(base_output_dir, "runs")
    ensure_dir(runs_dir)

    run_name = _get_run_name(config)
    run_dir = os.path.join(runs_dir, run_name)
    ensure_dir(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)
    maybe_set_speed_knobs()

    print(f"[Run] base_output_dir={base_output_dir}")
    print(f"[Run] run_name={run_name}")
    print(f"[Run] run_dir={run_dir}")
    print(f"[Run] tokens/step(est)={_estimate_tokens_per_step(config)}")

    # 1) load base model & tokenizer
    base_model, tokenizer = load_base_model_and_tokenizer(config)
    maybe_set_tokenizer_pad(tokenizer)

    base_model.to(device)

    # IMPORTANT: disable KV cache for causal LM training
    if hasattr(base_model, "config") and getattr(base_model.config, "use_cache", None) is True:
        base_model.config.use_cache = False

    # 2) dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # Useful prints
    try:
        print("train batches:", len(train_loader))
    except Exception:
        print("train batches: (iterable/unknown)")
    try:
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    # 3) add LoRA
    model = add_lora_to_model(base_model, config)
    model.to(device)
    print_trainable_params(model)

    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # -------------------------
    # Train config (unified keys)
    # -------------------------
    train_cfg = _get_train_knobs(config)

    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    num_epochs = int(train_cfg["num_epochs"])
    grad_clip = float(train_cfg["grad_clip"])

    use_amp = bool(train_cfg["use_amp"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])

    max_train_steps = train_cfg["max_train_steps"]  # optimizer steps
    warmup_ratio = float(train_cfg["warmup_ratio"])
    scheduler_type = str(train_cfg["scheduler"]).lower()

    log_every = int(train_cfg["log_every_steps"])
    eval_every = int(train_cfg["eval_every_steps"])

    # optimizer only on trainable (LoRA) parameters
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=lr, weight_decay=weight_decay)

    # ---- scheduler ----
    # total_steps counts optimizer steps (global_step)
    if max_train_steps is not None:
        total_steps = int(max_train_steps)
        stop_mode = f"step-based stop at max_train_steps={max_train_steps}"
    else:
        # fall back: epoch-based (map-style only)
        try:
            steps_per_epoch = len(train_loader)
        except Exception:
            raise ValueError(
                "[11_exp1_lora_lm] train_loader has no len (likely streaming). "
                "Please set train.max_train_steps in config."
            )
        opt_steps_per_epoch = (steps_per_epoch + grad_accum_steps - 1) // grad_accum_steps
        total_steps = int(opt_steps_per_epoch * max(num_epochs, 1))
        stop_mode = f"epoch-based stop (total_steps~{total_steps})"

    total_steps = max(1, int(total_steps))
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    if scheduler_type not in ["linear", ""]:
        print(f"[Warn] scheduler='{scheduler_type}' not implemented; using linear warmup-decay.")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler, autocast = _make_amp_helpers(device, use_amp)

    print(
        f"[Train] task=causal_lm {stop_mode} "
        f"total_steps={total_steps} warmup_steps={warmup_steps} "
        f"grad_accum={grad_accum_steps} lr={lr:g} wd={weight_decay:g}"
    )

    # -------------------------
    # Train loop (best by val loss)
    # Key: global_step == optimizer steps
    #      micro_step  == backward steps (micro-batches)
    # Must be able to run past one epoch to reach max_train_steps.
    # -------------------------
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    global_step = 0
    micro_step = 0
    stopped_by_steps = False

    # Also save best adapter on disk (adapter-only, lightweight)
    adapter_dir = os.path.join(run_dir, "adapter")
    adapter_best_dir = os.path.join(run_dir, "adapter_best")
    ensure_dir(adapter_dir)
    ensure_dir(adapter_best_dir)

    optimizer.zero_grad(set_to_none=True)

    # Helper: do an eval and maybe update best
    def _eval_and_maybe_best(tag: str):
        nonlocal best_val_loss, best_state
        val_metrics = evaluate(model, val_loader, config)
        val_loss = float(val_metrics.get("loss", 1e9))
        print(f"\n[{tag}] val={val_metrics}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # save adapter best immediately (safer for long runs)
            try:
                model.save_pretrained(adapter_best_dir)
                tokenizer.save_pretrained(adapter_best_dir)
                print(f"[Best] adapter_best saved. best_val_loss={best_val_loss:.6f}")
            except Exception as e:
                print(f"[Best] save adapter_best failed: {e}")
        return val_metrics

    # We still keep epochs for logging readability,
    # but if max_train_steps is set, we will auto-continue epochs until reaching it.
    epoch = 0
    while True:
        epoch += 1
        model.train()

        # If user set num_epochs AND no max_train_steps, obey epoch limit.
        if (max_train_steps is None) and (epoch > num_epochs):
            break

        loop = tqdm(train_loader, desc=f"[Exp1 LoRA-LM] Epoch {epoch}", leave=True)

        # stats per epoch (micro-batch weighted)
        ep_loss_sum = 0.0
        ep_count = 0

        for batch in loop:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                loss_to_backward = loss / float(grad_accum_steps)

            scaler.scale(loss_to_backward).backward()
            micro_step += 1

            bs = int(batch["input_ids"].size(0))
            ep_loss_sum += float(loss.item()) * bs
            ep_count += bs

            if micro_step % grad_accum_steps == 0:
                # grad clip
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip,
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1

                # logging
                if log_every > 0 and (global_step % log_every == 0):
                    lr_now = float(scheduler.get_last_lr()[0])
                    loop.set_postfix(loss=float(loss.item()), lr=lr_now)

                # periodic eval
                if eval_every > 0 and (global_step % eval_every == 0):
                    _eval_and_maybe_best(f"Eval@step {global_step}")

                # step-based stop
                if (max_train_steps is not None) and (global_step >= max_train_steps):
                    stopped_by_steps = True
                    break

        # epoch end summary
        ep_train_loss = ep_loss_sum / max(ep_count, 1)
        print(f"Epoch {epoch}: train_loss={ep_train_loss:.4f} (global_step={global_step})")

        # epoch end eval (useful even if eval_every==0)
        _eval_and_maybe_best(f"EpochEnd {epoch}")

        if stopped_by_steps:
            break

        # If user provided num_epochs AND max_train_steps, we ignore num_epochs (to guarantee step budget).
        # But to avoid infinite loops when max_train_steps is None:
        if (max_train_steps is None) and (epoch >= num_epochs):
            break

        # Safety: if train_loader is empty (shouldn't), prevent infinite loop
        if ep_count == 0:
            print("[Warn] Empty epoch (no batches). Stopping.")
            break

    # restore best (if any)
    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    val_metrics = evaluate(model, val_loader, config)
    test_metrics = evaluate(model, test_loader, config)

    # save final adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "data": config.get("data", {}),
        "lora": config.get("lora", {}),
        "train": config.get("train", {}),
        "run_name": run_name,
        "run_dir": run_dir,
        "best_val_loss": best_val_loss,
        "global_step": global_step,                  # optimizer steps
        "total_scheduler_steps": total_steps,
        "warmup_steps": warmup_steps,
        "grad_accum_steps": grad_accum_steps,
        "stopped_by_max_train_steps": bool(stopped_by_steps),
        "adapter_dir": adapter_dir,
        "adapter_best_dir": adapter_best_dir,
    }

    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    save_json(config, os.path.join(run_dir, "config_used.json"))

    print("=== Exp1 LoRA(LM) done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to run_dir:", run_dir)
    print("Adapter saved to:", adapter_dir)
    print("Best adapter saved to:", adapter_best_dir)


if __name__ == "__main__":
    main()
