from __future__ import annotations

from typing import Tuple, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)


def load_base_model_and_tokenizer(config: Dict[str, Any]):
    """
    期望 config 示例：
    config = {
        "model_name": "gpt2",
        "task_type": "classification",  # or "causal_lm" or "seq2seq"
        "num_labels": 2,               # 分类任务需要
    }
    """
    model_name = config.get("model_name") or config.get("base_model_ckpt")
    if not model_name:
        raise ValueError("config['model_name'] (or base_model_ckpt) is required.")

    task_type = config.get("task_type", "classification")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 处理一些模型没有 pad_token 的情况（如 GPT2）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if task_type == "classification":
        num_labels = int(config.get("num_labels", 2))
        hf_cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=hf_cfg)
    elif task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif task_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # 对齐 pad token id（尤其是 GPT2 分类时）
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
