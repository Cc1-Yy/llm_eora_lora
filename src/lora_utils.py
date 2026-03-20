# src/lora_utils.py
from __future__ import annotations
from typing import Dict, Any

from peft import LoraConfig, get_peft_model
from peft import TaskType


def _infer_peft_task_type(task_type: str) -> TaskType:
    """
    Map your config task_type -> PEFT TaskType.
    """
    t = (task_type or "classification").lower()
    if t == "lm":
        t = "causal_lm"

    if t == "classification":
        return TaskType.SEQ_CLS
    if t == "causal_lm":
        return TaskType.CAUSAL_LM
    if t == "seq2seq":
        return TaskType.SEQ_2_SEQ_LM

    # fallback: most safe default for GPT-like models
    return TaskType.CAUSAL_LM


def add_lora_to_model(model, config: Dict[str, Any]):
    lora_cfg = config.get("lora", {})
    r = int(lora_cfg.get("rank", 8))
    alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg.get("target_modules", None)

    if not target_modules:
        raise ValueError("config['lora']['target_modules'] is required.")

    task_type = _infer_peft_task_type(config.get("task_type", "classification"))

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)
    return model


def print_trainable_params(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    ratio = trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({ratio:.2%})")
