# src/train_utils.py
from __future__ import annotations

from typing import Dict, Any, Optional, Callable

import os
import time

import torch
from torch.optim import AdamW
from tqdm import tqdm

try:
    from transformers import get_linear_schedule_with_warmup
except Exception:
    get_linear_schedule_with_warmup = None


def _is_iterable_dataloader(train_loader) -> bool:
    ds = getattr(train_loader, "dataset", None)
    if ds is None:
        return False
    return ds.__class__.__name__ == "IterableDataset"


def _normalize_task_type(task_type: str) -> str:
    task_type = (task_type or "classification").lower()
    if task_type == "lm":
        task_type = "causal_lm"
    return task_type


def _get_amp_handles(device: torch.device, use_amp: bool):
    """
    Prefer the newer torch.amp API when available, fallback to torch.cuda.amp.
    Returns (scaler, autocast_ctx_factory).
    """
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
            return scaler, autocast
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)
            return scaler, autocast

    # CPU: no AMP
    try:
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        autocast = lambda: torch.autocast("cpu", enabled=False)
        return scaler, autocast
    except Exception:
        class _DummyScaler:
            def is_enabled(self): return False
            def scale(self, x): return x
            def unscale_(self, _): pass
            def step(self, opt): opt.step()
            def update(self): pass
        scaler = _DummyScaler()
        autocast = lambda: torch.autocast("cpu", enabled=False)
        return scaler, autocast


def train_optimized(
    model,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    evaluate_fn: Callable,
    output_dir: Optional[str] = None,
):
    """
    Unified training loop for:
      - classification (accuracy-focused)
      - causal_lm (loss/ppl-focused)

    Key features:
      - AMP safe on Windows
      - grad_accum_steps
      - max_train_steps (strongly recommended for streaming/iterable; also good for fair comparisons)
      - linear warmup scheduler
      - periodic eval & best checkpoint retention (by acc or -loss)
      - accepts both YAML keys:
          log_every_steps / log_every
          eval_every_steps / eval_every
    """
    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    num_epochs = int(train_cfg.get("num_epochs", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    use_amp = bool(train_cfg.get("use_amp", True))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    grad_accum_steps = max(1, grad_accum_steps)

    # Step-based controls
    max_train_steps = train_cfg.get("max_train_steps", None)
    max_train_steps = int(max_train_steps) if max_train_steps is not None else None

    # ✅ compatibility: support both "*_steps" and legacy names
    log_every = int(train_cfg.get("log_every_steps", train_cfg.get("log_every", 50)))
    eval_every = int(train_cfg.get("eval_every_steps", train_cfg.get("eval_every", 1000)))

    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    warmup_steps = train_cfg.get("warmup_steps", None)
    warmup_steps = int(warmup_steps) if warmup_steps is not None else None

    task_type = _normalize_task_type(config.get("task_type", "classification"))

    device = next(model.parameters()).device

    # LM: disable cache for training stability
    if task_type == "causal_lm":
        if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
            model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # total steps estimation
    is_iterable = _is_iterable_dataloader(train_loader)

    if max_train_steps is None:
        if is_iterable:
            raise ValueError("[train_optimized] train.max_train_steps is required when using streaming/IterableDataset.")
        total_steps = (len(train_loader) * max(num_epochs, 1)) // grad_accum_steps
    else:
        total_steps = max_train_steps

    total_steps = max(1, int(total_steps))

    if warmup_steps is None:
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(int(warmup_steps), total_steps))

    scheduler = None
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    scaler, autocast = _get_amp_handles(device, use_amp)

    print(
        f"[train_optimized] task={task_type} iterable={is_iterable} "
        f"total_steps={total_steps} warmup_steps={warmup_steps} "
        f"grad_accum={grad_accum_steps} lr={lr} wd={weight_decay}"
    )
    if max_train_steps is None:
        print(f"[train_optimized] epoch-based: epochs={num_epochs}, steps/epoch={len(train_loader)//grad_accum_steps}")
    else:
        print(f"[train_optimized] step-based stop at max_train_steps={max_train_steps}")

    # best metric tracking
    best_metric = -1e9
    best_state = None

    global_step = 0          # optimizer steps
    micro_step = 0           # micro-batches
    running_loss = 0.0       # sum(loss * bs)
    running_count = 0        # sum(bs)
    t0 = time.time()

    def _score_from_val(val_metrics: Dict[str, float]) -> float:
        if task_type == "classification":
            return float(val_metrics.get("accuracy", -1.0))
        return -float(val_metrics.get("loss", 1e9))

    def _maybe_save_best(tag: str):
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        torch.save(best_state, os.path.join(output_dir, f"{tag}.pt"))

    epoch = 0
    stop_training = False

    while (not stop_training) and (epoch < max(num_epochs, 1)):
        epoch += 1
        model.train()

        loop = tqdm(train_loader, desc=f"[Optimized] Epoch {epoch}/{max(num_epochs,1)}", leave=True)
        optimizer.zero_grad(set_to_none=True)

        for batch in loop:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                loss_to_backward = loss / float(grad_accum_steps)

            if hasattr(scaler, "scale"):
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            micro_step += 1

            bs = int(batch["input_ids"].size(0)) if "input_ids" in batch else 1
            running_loss += float(loss.item()) * bs
            running_count += bs

            if micro_step % grad_accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    try:
                        scaler.unscale_(optimizer)
                    except Exception:
                        pass
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if hasattr(scaler, "step"):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                if log_every > 0 and (global_step % log_every == 0):
                    dt = time.time() - t0
                    avg_loss = running_loss / max(running_count, 1)
                    it_s = global_step / max(dt, 1e-9)
                    lr_now = optimizer.param_groups[0]["lr"]
                    loop.set_postfix(loss=float(avg_loss), lr=float(lr_now), it_s=float(it_s))

                if eval_every and eval_every > 0 and (global_step % eval_every == 0):
                    val_metrics = evaluate_fn(model, val_loader, config)
                    score = _score_from_val(val_metrics)
                    if score > best_metric:
                        best_metric = score
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        _maybe_save_best("best_state_dict")
                    print(f"[Step {global_step}] val={val_metrics}")

                if (max_train_steps is not None) and (global_step >= max_train_steps):
                    stop_training = True
                    break

        # epoch end eval
        val_metrics = evaluate_fn(model, val_loader, config)
        score = _score_from_val(val_metrics)
        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _maybe_save_best("best_state_dict")

        avg_epoch_loss = running_loss / max(running_count, 1)
        print(f"Epoch {epoch}: train_loss={avg_epoch_loss:.4f}, val={val_metrics}")

        running_loss = 0.0
        running_count = 0

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
