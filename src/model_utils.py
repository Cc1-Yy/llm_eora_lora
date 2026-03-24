# src/model_utils.py
from __future__ import annotations

import json
import os
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)


def _is_gptq_quantized_dir(model_name: str) -> bool:
    """
    Heuristic detection for a local GPTQ quantized checkpoint directory.
    """
    if not isinstance(model_name, str):
        return False
    if not os.path.isdir(model_name):
        return False

    quant_cfg_path = os.path.join(model_name, "quantize_config.json")
    config_path = os.path.join(model_name, "config.json")

    if os.path.exists(quant_cfg_path):
        return True

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            qcfg = cfg.get("quantization_config", {})
            if isinstance(qcfg, dict) and qcfg.get("quant_method") == "gptq":
                return True
        except Exception:
            pass

    return False


def load_base_model_and_tokenizer(config: Dict[str, Any]):
    """
    Load model + tokenizer for classification / causal LM / seq2seq.

    Supported config keys:
      - model_name
      - base_model_ckpt   (fallback alias for model_name)
      - tokenizer_name    (optional; useful when model dir lacks tokenizer files,
                           e.g. quantized checkpoints)
      - task_type         ("classification", "causal_lm", "lm", "seq2seq")
      - num_labels        (for classification)
    """
    model_name = config.get("model_name") or config.get("base_model_ckpt")
    if not model_name:
        raise ValueError("config['model_name'] (or base_model_ckpt) is required.")

    task_type = config.get("task_type", "classification")
    if task_type == "lm":
        task_type = "causal_lm"

    tokenizer_name = config.get("tokenizer_name", model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if task_type == "classification":
        num_labels = int(config.get("num_labels", 2))
        hf_cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=hf_cfg,
        )

    elif task_type == "causal_lm":
        # For LoRA / PEFT training on GPTQ checkpoints, prefer the Transformers
        # loading path instead of GPTQModel.from_quantized().
        # This is closer to the PEFT-documented GPTQ workflow.
        model = AutoModelForCausalLM.from_pretrained(model_name)

    elif task_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Set pad token on underlying config when available.
    if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Only resize embeddings for normal HF models.
    try:
        if not _is_gptq_quantized_dir(model_name):
            model_vocab_size = getattr(model.get_input_embeddings(), "num_embeddings", None)
            if model_vocab_size is not None and len(tokenizer) != model_vocab_size:
                model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    return model, tokenizer