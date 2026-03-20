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

from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM

# ---- Make imports robust ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.eval_utils import evaluate
from src.lora_utils import add_lora_to_model, print_trainable_params


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


def maybe_set_speed_knobs():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_set_tokenizer_pad(tokenizer):
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


def _make_amp_helpers(device: torch.device, use_amp: bool):
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


def _get_train_knobs(config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = config.get("train", {})
    out = dict(train_cfg) if isinstance(train_cfg, dict) else {}

    out["log_every_steps"] = int(train_cfg.get("log_every_steps", train_cfg.get("log_every", 50)))
    out["eval_every_steps"] = int(train_cfg.get("eval_every_steps", train_cfg.get("eval_every", 1000)))

    out["grad_accum_steps"] = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    out["num_epochs"] = max(1, int(train_cfg.get("num_epochs", 1)))

    mts = train_cfg.get("max_train_steps", None)
    out["max_train_steps"] = int(mts) if mts is not None else None

    out["warmup_ratio"] = float(train_cfg.get("warmup_ratio", 0.0))
    out["use_amp"] = bool(train_cfg.get("use_amp", True))
    out["lr"] = float(train_cfg.get("lr", 2e-5))
    out["weight_decay"] = float(train_cfg.get("weight_decay", 0.0))
    out["grad_clip"] = float(train_cfg.get("grad_clip", 1.0))

    # KD knobs
    kd_cfg = config.get("kd", {}) if isinstance(config.get("kd", {}), dict) else {}
    out["kd_T"] = float(kd_cfg.get("T", 1.0))
    out["kd_lambda"] = float(kd_cfg.get("lambda", 1.0))       # weight on KD loss
    out["kd_loss"] = str(kd_cfg.get("loss", "kl")).lower()    # "kl" or "mse"
    out["sup_lambda"] = float(kd_cfg.get("sup_lambda", 0.0))  # 0.0 => pure output-matching

    return out


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    T: float,
    mode: str = "kl",
) -> torch.Tensor:
    """
    student_logits, teacher_logits: [B, L, V]
    labels: [B, L]

    For causal LM, compare logits on positions [:, :-1, :]
    against labels[:, 1:].

    Only valid positions with labels != -100 are used.
    """
    s = student_logits[:, :-1, :]   # [B, L-1, V]
    t = teacher_logits[:, :-1, :]   # [B, L-1, V]
    y = labels[:, 1:]               # [B, L-1]

    valid = (y != -100)             # [B, L-1]

    if mode == "mse":
        token_loss = torch.mean((s - t) ** 2, dim=-1)  # [B, L-1]
    else:
        T = max(float(T), 1e-6)
        s_logp = torch.log_softmax(s / T, dim=-1)
        t_p = torch.softmax(t / T, dim=-1)
        token_loss = torch.sum(
            t_p * (torch.log(t_p + 1e-12) - s_logp), dim=-1
        ) * (T * T)  # [B, L-1]

    valid_f = valid.float()
    denom = valid_f.sum().clamp_min(1.0)
    return (token_loss * valid_f).sum() / denom


@torch.no_grad()
def eval_with_teacher_lm(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Teacher-alignment metrics for causal LM:
      - masked KL(teacher || student) on valid next-token positions
      - masked MSE(logits) on valid next-token positions
    """
    student.eval()
    teacher.eval()

    T = max(float(temperature), 1e-6)

    total_valid_tokens = 0.0
    kl_sum = 0.0
    mse_sum = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("[Exp2 LoRA-KD] batch missing labels; cannot compute teacher-alignment metrics.")

        model_inputs = {k: v for k, v in batch.items() if k != "labels"}

        out_s = student(**model_inputs)
        out_t = teacher(**model_inputs)

        logits_s = out_s.logits   # [B, L, V]
        logits_t = out_t.logits   # [B, L, V]

        s = logits_s[:, :-1, :]
        t = logits_t[:, :-1, :]
        y = labels[:, 1:]

        valid = (y != -100)
        valid_f = valid.float()
        n_valid = float(valid_f.sum().item())

        if n_valid <= 0:
            continue

        s_logp = torch.log_softmax(s / T, dim=-1)
        t_prob = torch.softmax(t / T, dim=-1)
        token_kl = torch.sum(
            t_prob * (torch.log(t_prob + 1e-12) - s_logp), dim=-1
        ) * (T * T)

        token_mse = torch.mean((s - t) ** 2, dim=-1)

        kl_sum += float((token_kl * valid_f).sum().item())
        mse_sum += float((token_mse * valid_f).sum().item())
        total_valid_tokens += n_valid

    denom = max(total_valid_tokens, 1.0)
    return {
        "kl_to_teacher": kl_sum / denom,
        "mse_logits_to_teacher": mse_sum / denom,
        "valid_tokens": total_valid_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    if str(config.get("task_type", "")).lower() == "lm":
        config["task_type"] = "causal_lm"
    if config.get("task_type") != "causal_lm":
        raise ValueError("This script is for causal_lm only.")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    base_output_dir = str(config.get("output_dir", "outputs/exp2_lora_lm_kd"))
    ensure_dir(base_output_dir)
    runs_dir = os.path.join(base_output_dir, "runs")
    ensure_dir(runs_dir)

    run_name = str(config.get("run_name", "run"))
    run_dir = os.path.join(runs_dir, run_name)
    ensure_dir(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)
    maybe_set_speed_knobs()

    # student base + LoRA
    base_model, tokenizer = load_base_model_and_tokenizer(config)
    maybe_set_tokenizer_pad(tokenizer)
    base_model.to(device)

    if hasattr(base_model, "config") and getattr(base_model.config, "use_cache", None) is True:
        base_model.config.use_cache = False

    model = add_lora_to_model(base_model, config)
    model.to(device)
    print_trainable_params(model)

    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # teacher
    teacher_dir = config.get("teacher_model_dir") or config.get("optimized_model_dir")
    if not teacher_dir:
        raise ValueError("Exp2 KD requires teacher_model_dir (path to merged teacher).")

    teacher = AutoModelForCausalLM.from_pretrained(teacher_dir)
    teacher.to(device)
    teacher.eval()

    if hasattr(teacher, "config") and getattr(teacher.config, "use_cache", None) is True:
        teacher.config.use_cache = False

    # data
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    try:
        print("train batches:", len(train_loader))
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    train_cfg = _get_train_knobs(config)
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    grad_clip = float(train_cfg["grad_clip"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])
    max_train_steps = train_cfg["max_train_steps"]
    warmup_ratio = float(train_cfg["warmup_ratio"])
    use_amp = bool(train_cfg["use_amp"])

    kd_T = float(train_cfg["kd_T"])
    kd_lambda = float(train_cfg["kd_lambda"])
    kd_mode = str(train_cfg["kd_loss"])
    sup_lambda = float(train_cfg["sup_lambda"])

    log_every = int(train_cfg["log_every_steps"])
    eval_every = int(train_cfg["eval_every_steps"])

    if max_train_steps is None:
        raise ValueError("[Exp2 KD] Please set train.max_train_steps for fair comparisons.")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    total_steps = int(max_train_steps)
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler, autocast = _make_amp_helpers(device, use_amp)

    print(f"[Run] run_dir={run_dir}")
    print(
        f"[KD] teacher={teacher_dir}  T={kd_T}  lambda={kd_lambda}  "
        f"sup_lambda={sup_lambda}  loss={kd_mode}"
    )
    print(f"[Train] steps={total_steps} warmup={warmup_steps} lr={lr:g} grad_accum={grad_accum_steps}")

    select_best_by = str(config.get("select_best_by", "loss")).lower()

    best_score = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    global_step = 0
    micro_step = 0

    optimizer.zero_grad(set_to_none=True)

    def _eval_and_maybe_best(tag: str):
        nonlocal best_score, best_state

        val_task = evaluate(model, val_loader, config)
        val_align = eval_with_teacher_lm(
            model, teacher, val_loader, device=device, temperature=kd_T
        )
        val_metrics = {**val_task, **val_align}

        if select_best_by == "kl_to_teacher":
            score = float(val_metrics.get("kl_to_teacher", 1e9))
        else:
            score = float(val_metrics.get("loss", 1e9))

        print(f"\n[{tag}] val={val_metrics}")

        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            try:
                best_dir = os.path.join(run_dir, "adapter_best")
                ensure_dir(best_dir)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"[Best] adapter_best saved. best_{select_best_by}={best_score:.6f}")
            except Exception as e:
                print(f"[Best] save adapter_best failed: {e}")

        return val_metrics

    epoch = 0
    stopped_by_steps = False

    while global_step < total_steps:
        epoch += 1
        model.train()

        loop = tqdm(train_loader, desc=f"[Exp2 LoRA-KD] Epoch {epoch}", leave=True)

        ep_loss_sum = 0.0
        ep_count = 0

        for batch in loop:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.get("labels", None)
            if labels is None:
                raise ValueError("[Exp2 KD] batch missing labels. Ensure DataCollatorForLanguageModeling provides labels.")

            teacher_inputs = {k: v for k, v in batch.items() if k != "labels"}

            with torch.no_grad():
                teacher_outputs = teacher(**teacher_inputs)
                teacher_logits = teacher_outputs.logits

            with autocast():
                student_outputs = model(**batch)
                student_logits = student_outputs.logits

                # HF causal LM supervised loss (already handles labels == -100)
                sup_loss = student_outputs.loss

                # masked KD loss on valid next-token positions only
                kd_loss = kd_loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    T=kd_T,
                    mode=kd_mode,
                )

                # Exp2 main: set sup_lambda=0.0 for pure output-matching
                loss = sup_lambda * sup_loss + kd_lambda * kd_loss
                loss_to_backward = loss / float(grad_accum_steps)

            scaler.scale(loss_to_backward).backward()
            micro_step += 1

            bs = int(batch["input_ids"].size(0))
            ep_loss_sum += float(loss.item()) * bs
            ep_count += bs

            if micro_step % grad_accum_steps == 0:
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

                if log_every > 0 and (global_step % log_every == 0):
                    lr_now = float(scheduler.get_last_lr()[0])
                    loop.set_postfix(
                        loss=float(loss.item()),
                        sup=float(sup_loss.item()),
                        kd=float(kd_loss.item()),
                        lr=lr_now,
                    )

                if eval_every > 0 and (global_step % eval_every == 0):
                    _eval_and_maybe_best(f"Eval@step {global_step}")

                if global_step >= total_steps:
                    stopped_by_steps = True
                    break

        ep_train_loss = ep_loss_sum / max(ep_count, 1)
        print(f"Epoch {epoch}: train_loss={ep_train_loss:.4f} (global_step={global_step})")

        _eval_and_maybe_best(f"EpochEnd {epoch}")

        if stopped_by_steps:
            break

        if ep_count == 0:
            print("[Warn] Empty epoch (no batches). Stopping.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    val_task = evaluate(model, val_loader, config)
    test_task = evaluate(model, test_loader, config)

    val_align = eval_with_teacher_lm(
        model, teacher, val_loader, device=device, temperature=kd_T
    )
    test_align = eval_with_teacher_lm(
        model, teacher, test_loader, device=device, temperature=kd_T
    )

    val_metrics = {**val_task, **val_align}
    test_metrics = {**test_task, **test_align}

    # save final adapter
    adapter_dir = os.path.join(run_dir, "adapter")
    ensure_dir(adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # meta
    meta = {
        "seed": seed,
        "teacher_model_dir": teacher_dir,
        "kd": {
            "T": kd_T,
            "lambda": kd_lambda,
            "sup_lambda": sup_lambda,
            "loss": kd_mode,
        },
        "select_best_by": select_best_by,
        "lora": config.get("lora", {}),
        "train": config.get("train", {}),
        "data": config.get("data", {}),
        "best_score": best_score,
        "best_metric": select_best_by,
        "global_step": global_step,
        "run_name": run_name,
        "run_dir": run_dir,
    }
    save_json(meta, os.path.join(run_dir, "meta.json"))

    metrics = {"val": val_metrics, "test": test_metrics}
    save_json(metrics, os.path.join(run_dir, "metrics.json"))

    print("=== Exp2 LoRA-KD done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()