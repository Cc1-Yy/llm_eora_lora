from __future__ import annotations

import os
import sys
import json
import math
import argparse
from typing import Dict, Any

import torch
import yaml
from transformers import AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders


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


def maybe_set_tokenizer_pad(tokenizer):
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


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


def _unwrap_forward_model(model_or_wrapper):
    inner = getattr(model_or_wrapper, "model", None)
    if inner is not None:
        return inner
    return model_or_wrapper


def _set_eval_mode(model_or_wrapper):
    if hasattr(model_or_wrapper, "eval"):
        try:
            model_or_wrapper.eval()
        except Exception:
            pass

    inner = getattr(model_or_wrapper, "model", None)
    if inner is not None and hasattr(inner, "eval"):
        try:
            inner.eval()
        except Exception:
            pass


@torch.no_grad()
def evaluate_causal_lm(model_or_wrapper, dataloader, device: str):
    model = _unwrap_forward_model(model_or_wrapper)
    _set_eval_mode(model_or_wrapper)

    total_loss = 0.0
    total_valid_tokens = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("LM evaluation requires labels.")

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
def eval_with_teacher_lm(student_or_wrapper, teacher, dataloader, device: str, temperature: float = 1.0):
    student = _unwrap_forward_model(student_or_wrapper)
    _set_eval_mode(student_or_wrapper)
    teacher.eval()

    T = max(float(temperature), 1e-6)

    total_valid_tokens = 0.0
    kl_sum = 0.0
    mse_sum = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("batch missing labels")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    normalize_task_type(cfg)

    if cfg.get("task_type") != "causal_lm":
        raise ValueError("This script is for causal_lm only.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = str(cfg.get("output_dir", "outputs/lm/baseline_eval"))
    ensure_dir(output_dir)

    model, tokenizer = load_base_model_and_tokenizer(cfg)
    maybe_set_tokenizer_pad(tokenizer)

    model_name = cfg.get("model_name") or cfg.get("base_model_ckpt")
    is_quantized = _is_gptq_quantized_dir(model_name)

    if not is_quantized:
        inner = _unwrap_forward_model(model)
        if isinstance(inner, torch.nn.Module):
            inner.to(device)

    for obj in [model, getattr(model, "model", None)]:
        if obj is None:
            continue
        if hasattr(obj, "config") and getattr(obj.config, "use_cache", None) is True:
            obj.config.use_cache = False

    teacher_dir = cfg.get("optimized_model_dir", None)
    teacher = None
    if teacher_dir:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_dir)
        teacher.to(device)
        teacher.eval()
        if hasattr(teacher, "config") and getattr(teacher.config, "use_cache", None) is True:
            teacher.config.use_cache = False

    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)

    val_task = evaluate_causal_lm(model, val_loader, device=device)
    test_task = evaluate_causal_lm(model, test_loader, device=device)

    metrics = {
        "val": val_task,
        "test": test_task,
    }

    if teacher is not None:
        temperature = float(
            cfg.get("distill", {}).get(
                "temperature",
                cfg.get("kd", {}).get("T", 1.0)
            )
        )
        val_align = eval_with_teacher_lm(model, teacher, val_loader, device=device, temperature=temperature)
        test_align = eval_with_teacher_lm(model, teacher, test_loader, device=device, temperature=temperature)
        metrics["val"].update(val_align)
        metrics["test"].update(test_align)

    save_json(metrics, os.path.join(output_dir, "metrics.json"))

    print("=== Baseline LM eval done ===")
    print("Val metrics:", metrics["val"])
    print("Test metrics:", metrics["test"])
    print("Saved to:", os.path.join(output_dir, "metrics.json"))


if __name__ == "__main__":
    main()