from __future__ import annotations

from typing import Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import DataCollatorWithPadding


def _get_text_and_label_keys(dataset_name: str) -> Dict[str, str]:
    """
    针对常见分类数据集，返回文本字段名和标签字段名。
    你之后加数据集就往这里补一行。
    """
    name = dataset_name.lower()
    if name in ["sst2", "glue/sst2", "glue"]:
        return {"text": "sentence", "label": "label"}
    if name in ["ag_news"]:
        return {"text": "text", "label": "label"}
    if name in ["yelp_polarity"]:
        return {"text": "text", "label": "label"}
    # fallback（尽量不崩）
    return {"text": "text", "label": "label"}


def build_tokenize_fn(tokenizer, text_key: str, max_length: int):
    def fn(batch):
        return tokenizer(
            batch[text_key],
            truncation=True,
            max_length=max_length,
        )

    return fn


def get_dataloaders(config: Dict[str, Any], tokenizer):
    """
    期望 config 结构示例：
    config = {
        "task_type": "classification",
        "data": {
            "dataset_name": "sst2",
            "dataset_config_name": None,   # 可选
            "max_length": 128,
            "batch_size": 8,
            "num_workers": 2,
        },
        "seed": 42,
    }
    """
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name")
    dataset_config_name = data_cfg.get("dataset_config_name", None)
    max_length = int(data_cfg.get("max_length", 128))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))

    if dataset_name is None:
        raise ValueError("config['data']['dataset_name'] is required.")

    # 1) load dataset
    # 支持 "glue/sst2" 这种写法，也支持 dataset_config_name 分离写法
    if "/" in dataset_name and dataset_config_name is None:
        # 例如 "glue/sst2"
        parts = dataset_name.split("/")
        ds = load_dataset(parts[0], parts[1])
        canonical_name = dataset_name
    else:
        ds = load_dataset(dataset_name, dataset_config_name) if dataset_config_name else load_dataset(dataset_name)
        canonical_name = dataset_name

    # 2) infer keys
    keys = _get_text_and_label_keys(canonical_name)
    text_key, label_key = keys["text"], keys["label"]

    # 3) tokenize
    tokenize_fn = build_tokenize_fn(tokenizer, text_key, max_length)

    # 提前确认 split 名
    # 常见是 train/validation/test，有些是 train/test 需要自己切 val
    has_val = "validation" in ds
    has_test = "test" in ds

    # 4) map
    remove_cols = [c for c in ds["train"].column_names if c not in [text_key, label_key]]
    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    # 5) rename label -> labels (HF Trainer & 自己写循环都更统一)
    def rename_label(batch):
        batch["labels"] = batch[label_key]
        return batch

    ds_tok = ds_tok.map(rename_label, batched=False)

    # 6) set format torch
    cols = ["input_ids", "attention_mask", "labels"]
    for split in ds_tok.keys():
        keep = [c for c in cols if c in ds_tok[split].column_names]
        ds_tok[split].set_format(type="torch", columns=keep)

    # 7) collator
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8) choose splits
    train_ds = ds_tok["train"]
    val_ds = ds_tok["validation"] if has_val else None
    test_ds = ds_tok["test"] if has_test else None

    # 若没有 validation，就从 train 切一点（最小可行）
    if val_ds is None:
        split = train_ds.train_test_split(test_size=0.1, seed=int(config.get("seed", 42)))
        train_ds, val_ds = split["train"], split["test"]

    # 若没有 test，就把 validation 当 test 先顶上（最小可行）
    if test_ds is None:
        test_ds = val_ds

    try:
        if "labels" in test_ds.column_names:
            sample = test_ds.select(range(min(200, len(test_ds))))
            labels = sample["labels"]
            # labels 可能是 list 或 tensor
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            if all(l == -1 for l in labels):
                test_ds = val_ds
    except Exception:
        pass

    # 9) dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return train_loader, val_loader, test_loader
