# scripts/core/31_exp3_lora_lm.py
from __future__ import annotations

import os
import sys
import json
import math
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
from src.lora_utils import add_lora_to_model, print_trainable_params


# ============================================================
# Basic utils
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
    if str(config.get("task_type", "")).lower() == "lm":
        config["task_type"] = "causal_lm"


def maybe_set_speed_knobs():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_set_tokenizer_pad(tokenizer):
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


def _pretty_ar(alpha: float, rank: int) -> str:
    ar = float(alpha) / max(int(rank), 1)
    if abs(ar - round(ar)) < 1e-9:
        return f"{ar:.0f}"
    return f"{ar:.2f}".rstrip("0").rstrip(".")


def _get_run_name(config: Dict[str, Any]) -> str:
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
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    bs = int(data_cfg.get("batch_size", 1))
    max_len = int(data_cfg.get("max_length", 128))
    acc = int(train_cfg.get("grad_accum_steps", 1))
    acc = max(1, acc)
    return bs * max_len * acc


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
    out["scheduler"] = str(train_cfg.get("scheduler", "linear")).lower()

    return out


# ============================================================
# Quantized / wrapper helpers
# ============================================================

def _is_gptq_quantized_dir(path: str) -> bool:
    if not isinstance(path, str):
        return False
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "quantize_config.json")):
        return True
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        qcfg = obj.get("quantization_config", {})
        return isinstance(qcfg, dict) and qcfg.get("quant_method") == "gptq"
    except Exception:
        return False


def _resolve_quantized_model_dir(config: Dict[str, Any]) -> Optional[str]:
    qdir = config.get("quantized_model_dir", None)
    if qdir:
        return str(qdir)
    model_name = config.get("model_name", None)
    if model_name and _is_gptq_quantized_dir(model_name):
        return str(model_name)
    return None


def _unwrap_trainable_backbone(model_or_wrapper):
    """
    load_base_model_and_tokenizer() may return:
      - a normal HF nn.Module
      - a GPTQ wrapper with .model
    For PEFT/LoRA training, prefer the inner nn.Module when available.
    """
    if isinstance(model_or_wrapper, torch.nn.Module):
        inner = getattr(model_or_wrapper, "model", None)
        if isinstance(inner, torch.nn.Module):
            print("[Load] Using inner `.model` as LoRA trainable backbone.")
            return inner
        return model_or_wrapper

    inner = getattr(model_or_wrapper, "model", None)
    if isinstance(inner, torch.nn.Module):
        print("[Load] Using wrapper `.model` as LoRA trainable backbone.")
        return inner

    raise TypeError(
        f"[31_exp3_lora_lm] Loaded model is not a trainable nn.Module. "
        f"type={type(model_or_wrapper)}"
    )


def _disable_use_cache(model):
    for obj in [model, getattr(model, "model", None)]:
        if obj is None:
            continue
        if hasattr(obj, "config") and getattr(obj.config, "use_cache", None) is True:
            obj.config.use_cache = False


# ============================================================
# Eval helpers (token-weighted LM + teacher alignment)
# ============================================================

@torch.no_grad()
def evaluate_causal_lm(model: torch.nn.Module, dataloader, device: torch.device) -> Dict[str, float]:
    """
    Token-weighted LM eval.
    outputs.loss is averaged over valid target tokens in a batch, so we
    reweight by the number of valid target tokens for a more correct corpus loss.
    """
    model.eval()

    total_loss = 0.0
    total_valid_tokens = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("[31_exp3_lora_lm] LM eval requires labels in the batch.")

        outputs = model(**batch)
        loss = outputs.loss

        valid = (labels[:, 1:] != -100).float()
        n_valid = float(valid.sum().item())
        if n_valid <= 0:
            continue

        total_loss += float(loss.item()) * n_valid
        total_valid_tokens += n_valid

    avg_loss = total_loss / max(total_valid_tokens, 1.0)
    ppl = float(math.exp(min(avg_loss, 20.0)))

    return {
        "loss": float(avg_loss),
        "ppl": ppl,
        "valid_tokens": float(total_valid_tokens),
    }


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
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError(
                "[31_exp3_lora_lm] batch missing labels; cannot compute teacher-alignment metrics."
            )

        model_inputs = {k: v for k, v in batch.items() if k != "labels"}

        out_s = student(**model_inputs)
        out_t = teacher(**model_inputs)

        logits_s = out_s.logits
        logits_t = out_t.logits

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
            f"[31_exp3_lora_lm] This script is for causal LM only. "
            f"Got: {config.get('task_type')}"
        )

    seed = int(config.get("seed", 42))
    set_seed(seed)
    maybe_set_speed_knobs()

    quantized_model_dir = _resolve_quantized_model_dir(config)
    optimized_model_dir = config.get("optimized_model_dir") or config.get("teacher_model_dir")
    if not optimized_model_dir:
        raise ValueError("[31_exp3_lora_lm] Config must provide optimized_model_dir (teacher).")

    base_output_dir = str(config.get("output_dir", "outputs/lm/exp3/lora_recover"))
    ensure_dir(base_output_dir)

    runs_dir = os.path.join(base_output_dir, "runs")
    ensure_dir(runs_dir)

    run_name = _get_run_name(config)
    run_dir = os.path.join(runs_dir, run_name)
    ensure_dir(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)
    print(f"[Run] base_output_dir={base_output_dir}")
    print(f"[Run] run_name={run_name}")
    print(f"[Run] run_dir={run_dir}")
    print(f"[Run] tokens/step(est)={_estimate_tokens_per_step(config)}")
    print(f"[Run] quantized_model_dir={quantized_model_dir}")
    print(f"[Run] optimized_model_dir={optimized_model_dir}")

    # --------------------------------------------------------
    # 1) Load quantized backbone + tokenizer
    # --------------------------------------------------------
    loaded_model, tokenizer = load_base_model_and_tokenizer(config)
    maybe_set_tokenizer_pad(tokenizer)

    base_model = _unwrap_trainable_backbone(loaded_model)
    _disable_use_cache(base_model)

    # For non-quantized fallback runs, moving to device is fine.
    # For GPTQ quantized backbones, weights are usually already placed correctly.
    is_quantized_backbone = quantized_model_dir is not None
    if not is_quantized_backbone:
        base_model.to(device)

    # --------------------------------------------------------
    # 2) Dataloaders
    # --------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    try:
        print("train batches:", len(train_loader))
    except Exception:
        print("train batches: (iterable/unknown)")
    try:
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    # --------------------------------------------------------
    # 3) Add LoRA on top of quantized backbone
    # --------------------------------------------------------
    model = add_lora_to_model(base_model, config)
    _disable_use_cache(model)

    print_trainable_params(model)

    # Newly created LoRA params are typically created on the target module device.
    # For non-quantized fallback, .to(device) is safe.
    if not is_quantized_backbone:
        model.to(device)

    # --------------------------------------------------------
    # 4) Teacher for alignment eval
    # --------------------------------------------------------
    teacher = AutoModelForCausalLM.from_pretrained(optimized_model_dir)
    teacher.to(device)
    teacher.eval()
    _disable_use_cache(teacher)

    # --------------------------------------------------------
    # 5) Train config
    # --------------------------------------------------------
    train_cfg = _get_train_knobs(config)

    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    num_epochs = int(train_cfg["num_epochs"])
    grad_clip = float(train_cfg["grad_clip"])

    use_amp = bool(train_cfg["use_amp"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])

    max_train_steps = train_cfg["max_train_steps"]
    warmup_ratio = float(train_cfg["warmup_ratio"])
    scheduler_type = str(train_cfg["scheduler"]).lower()

    log_every = int(train_cfg["log_every_steps"])
    eval_every = int(train_cfg["eval_every_steps"])

    temperature = float(
        config.get("distill", {}).get(
            "temperature",
            config.get("kd", {}).get("T", 1.0)
        )
    )
    select_best_by = str(config.get("select_best_by", "loss")).lower()

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if len(optim_params) == 0:
        raise ValueError("[31_exp3_lora_lm] No trainable parameters found after adding LoRA.")
    optimizer = AdamW(optim_params, lr=lr, weight_decay=weight_decay)

    if max_train_steps is not None:
        total_steps = int(max_train_steps)
        stop_mode = f"step-based stop at max_train_steps={max_train_steps}"
    else:
        try:
            steps_per_epoch = len(train_loader)
        except Exception:
            raise ValueError(
                "[31_exp3_lora_lm] train_loader has no len (likely streaming). "
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
    print(f"[SelectBest] metric={select_best_by} temperature={temperature:g}")

    # --------------------------------------------------------
    # 6) Train loop
    # --------------------------------------------------------
    best_score = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    global_step = 0
    micro_step = 0
    stopped_by_steps = False

    adapter_dir = os.path.join(run_dir, "adapter")
    adapter_best_dir = os.path.join(run_dir, "adapter_best")
    ensure_dir(adapter_dir)
    ensure_dir(adapter_best_dir)

    optimizer.zero_grad(set_to_none=True)

    def _eval_and_maybe_best(tag: str):
        nonlocal best_score, best_state

        val_task = evaluate_causal_lm(model, val_loader, device=device)
        val_align = eval_with_teacher_lm(
            model, teacher, val_loader, device=device, temperature=temperature
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
                model.save_pretrained(adapter_best_dir)
                tokenizer.save_pretrained(adapter_best_dir)
                print(f"[Best] adapter_best saved. best_{select_best_by}={best_score:.6f}")
            except Exception as e:
                print(f"[Best] save adapter_best failed: {e}")

        return val_metrics

    epoch = 0
    while True:
        epoch += 1
        model.train()

        if (max_train_steps is None) and (epoch > num_epochs):
            break

        loop = tqdm(train_loader, desc=f"[Exp3 LoRA-LM] Epoch {epoch}", leave=True)

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
                    loop.set_postfix(loss=float(loss.item()), lr=lr_now)

                if eval_every > 0 and (global_step % eval_every == 0):
                    _eval_and_maybe_best(f"Eval@step {global_step}")

                if (max_train_steps is not None) and (global_step >= max_train_steps):
                    stopped_by_steps = True
                    break

        ep_train_loss = ep_loss_sum / max(ep_count, 1)
        print(f"Epoch {epoch}: train_loss={ep_train_loss:.4f} (global_step={global_step})")

        _eval_and_maybe_best(f"EpochEnd {epoch}")

        if stopped_by_steps:
            break

        if (max_train_steps is None) and (epoch >= num_epochs):
            break

        if ep_count == 0:
            print("[Warn] Empty epoch (no batches). Stopping.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------------------------------------------------
    # 7) Final eval
    # --------------------------------------------------------
    val_task = evaluate_causal_lm(model, val_loader, device=device)
    test_task = evaluate_causal_lm(model, test_loader, device=device)

    val_align = eval_with_teacher_lm(
        model, teacher, val_loader, device=device, temperature=temperature
    )
    test_align = eval_with_teacher_lm(
        model, teacher, test_loader, device=device, temperature=temperature
    )

    val_metrics = {**val_task, **val_align}
    test_metrics = {**test_task, **test_align}

    # save final adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    meta = {
        "seed": seed,
        "task_type": config.get("task_type"),
        "model_name": config.get("model_name"),
        "tokenizer_name": config.get("tokenizer_name"),
        "optimized_model_dir": optimized_model_dir,
        "quantized_model_dir": quantized_model_dir,
        "is_quantized_backbone": bool(is_quantized_backbone),
        "lora": config.get("lora", {}),
        "train": config.get("train", {}),
        "data": config.get("data", {}),
        "distill": config.get("distill", {}),
        "select_best_by": select_best_by,
        "best_score": best_score,
        "best_metric": select_best_by,
        "global_step": global_step,
        "total_scheduler_steps": total_steps,
        "warmup_steps": warmup_steps,
        "grad_accum_steps": grad_accum_steps,
        "stopped_by_max_train_steps": bool(stopped_by_steps),
        "run_name": run_name,
        "run_dir": run_dir,
        "adapter_dir": adapter_dir,
        "adapter_best_dir": adapter_best_dir,
        "notes": [
            "This run trains LoRA on top of a quantized GPTQ backbone when quantized_model_dir/model_name points to a GPTQ directory.",
            "Task metrics are token-weighted loss/ppl; alignment metrics are KL and logits-MSE to the optimized teacher.",
            "For first verification, use a small max_train_steps smoke test before launching full runs.",
        ],
    }
    save_json(meta, os.path.join(run_dir, "meta.json"))

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    save_json(config, os.path.join(run_dir, "config_used.json"))

    print("=== Exp3 LoRA-LM done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", run_dir)
    print("Adapter saved to:", adapter_dir)
    print("Best adapter saved to:", adapter_best_dir)


if __name__ == "__main__":
    main()