from __future__ import annotations
from typing import Dict, Any

from peft import LoraConfig, get_peft_model, TaskType


def _map_task_type_for_peft(task_type: str) -> TaskType:
    """
    Map your config task_type -> PEFT TaskType
    """
    t = (task_type or "").lower()

    # allow aliases
    if t in ["lm", "causal_lm", "causallm", "language_modeling"]:
        return TaskType.CAUSAL_LM
    if t in ["classification", "seq_cls", "sequence_classification", "cls"]:
        return TaskType.SEQ_CLS
    if t in ["seq2seq", "seq_2_seq", "seq2seq_lm"]:
        return TaskType.SEQ_2_SEQ_LM

    raise ValueError(f"Unknown task_type for PEFT: {task_type}")


def add_lora_to_model(model, config: Dict[str, Any]):
    lora_cfg = config.get("lora", {})
    r = int(lora_cfg.get("rank", 8))
    alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg.get("target_modules", None)

    if not target_modules:
        raise ValueError("config['lora']['target_modules'] is required.")

    task_type = config.get("task_type", "classification")
    peft_task_type = _map_task_type_for_peft(task_type)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=peft_task_type,
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
