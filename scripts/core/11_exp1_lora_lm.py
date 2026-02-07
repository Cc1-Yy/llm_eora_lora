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

# ---- Make imports robust no matter where you run this from ----
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
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _get_run_name(config: Dict[str, Any]) -> str:
    """
    Optional helper: if you want auto subfolder naming like r8_ar1.25 etc.
    If config["run_name"] exists -> use it.
    Else build from LoRA rank & alpha/r.
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
        ar = a / max(r, 1)
        # keep 2 decimals if needed
        if abs(ar - round(ar)) < 1e-9:
            ar_str = f"{ar:.0f}"
        else:
            ar_str = f"{ar:.2f}".rstrip("0").rstrip(".")
        return f"r{r}_ar{ar_str}"
    except Exception:
        return "run"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ---- normalize task_type alias ----
    if config.get("task_type") == "lm":
        config["task_type"] = "causal_lm"

    if config.get("task_type") != "causal_lm":
        raise ValueError(
            f"[11_exp1_lora_lm] This script is for LM only. "
            f"Please set task_type: causal_lm (or lm). Got: {config.get('task_type')}"
        )

    seed = int(config.get("seed", 42))
    set_seed(seed)

    base_output_dir = config.get("output_dir", "outputs/exp1_lora_lm")
    ensure_dir(base_output_dir)

    # Optional: create a subfolder per run to avoid overwrite
    run_name = _get_run_name(config)
    output_dir = os.path.join(base_output_dir, run_name)
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) load base LM model & tokenizer
    base_model, tokenizer = load_base_model_and_tokenizer(config)
    base_model.to(device)

    # 2) dataloaders (LM pipeline should return batches with input_ids/attention_mask/labels)
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # 3) add LoRA
    model = add_lora_to_model(base_model, config)
    model.to(device)
    print_trainable_params(model)

    # 4) optimizer (only train requires_grad params)
    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    num_epochs = int(train_cfg.get("num_epochs", 1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # Optional speed/memory helpers
    use_amp = bool(train_cfg.get("use_amp", True))  # safe default on RTX 3060
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=lr, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    # 5) train loop (select best by val loss, smaller is better)
    best_val_loss = float("inf")
    best_state = None

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens_or_examples = 0

        loop = tqdm(train_loader, desc=f"[Exp1 LoRA-LM] Epoch {epoch}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loop, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                outputs = model(**batch)
                loss = outputs.loss
                # gradient accumulation
                loss_to_backward = loss / max(grad_accum_steps, 1)

            scaler.scale(loss_to_backward).backward()

            # bookkeeping (use batch size as "examples" proxy; LM ppl uses loss anyway)
            bs = batch["input_ids"].size(0)
            total_loss += float(loss.item()) * bs
            total_tokens_or_examples += bs

            if step % grad_accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optim_params, grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            loop.set_postfix(loss=float(loss.item()))

        train_loss = total_loss / max(total_tokens_or_examples, 1)

        # epoch end eval
        val_metrics = evaluate(model, val_loader, config)
        val_loss = float(val_metrics.get("loss", 1e9))
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val={val_metrics}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) final eval
    val_metrics = evaluate(model, val_loader, config)
    test_metrics = evaluate(model, test_loader, config)

    # 7) save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    ensure_dir(adapter_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # 8) save metrics/meta
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "seed": seed,
        "model_name": config.get("model_name"),
        "task_type": config.get("task_type"),
        "data": config.get("data", {}),
        "lora": config.get("lora", {}),
        "train": config.get("train", {}),
        "optimized_model_dir": config.get("optimized_model_dir", None),
        "run_name": run_name,
        "output_dir": output_dir,
    }

    save_json(metrics, os.path.join(output_dir, "metrics.json"))
    save_json(config, os.path.join(output_dir, "config_used.json"))

    print("=== Exp1 LoRA(LM) done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
