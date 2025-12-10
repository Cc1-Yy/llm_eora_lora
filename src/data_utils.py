from __future__ import annotations

from typing import Dict, Any

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling


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
    支持:
      - classification: 需要 label 字段 -> 会 rename_label
      - causal_lm: 不需要 label -> 用 LM collator 自动生成 labels

    期望 config 结构示例：
    config = {
        "task_type": "classification" | "causal_lm",
        "data": {
            "dataset_name": "glue/sst2" / "wikitext",
            "dataset_config_name": None,
            "max_length": 128,
            "batch_size": 8,
            "num_workers": 2,
            # 可选：当你用很不标准的数据集时手动指定文本字段
            # "text_key": "text"
        },
        "seed": 42,
    }
    """
    task_type = config.get("task_type", "classification")

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name")
    dataset_config_name = data_cfg.get("dataset_config_name", None)
    max_length = int(data_cfg.get("max_length", 128))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    text_key_override = data_cfg.get("text_key", None)

    if dataset_name is None:
        raise ValueError("config['data']['dataset_name'] is required.")

    # 1) load dataset
    # 支持 "glue/sst2" 这种写法，也支持 dataset_config_name 分离写法
    if "/" in dataset_name and dataset_config_name is None:
        parts = dataset_name.split("/")
        ds = load_dataset(parts[0], parts[1])
        canonical_name = dataset_name
    else:
        ds = load_dataset(dataset_name, dataset_config_name) if dataset_config_name else load_dataset(dataset_name)
        canonical_name = dataset_name

    # 2) infer keys
    if task_type == "classification":
        keys = _get_text_and_label_keys(canonical_name)
        text_key, label_key = keys["text"], keys["label"]
    else:
        # causal_lm: 不需要 label
        text_key = text_key_override or "text"
        label_key = None

    # 3) tokenize
    tokenize_fn = build_tokenize_fn(tokenizer, text_key, max_length)

    # 提前确认 split 名
    has_val = "validation" in ds
    has_test = "test" in ds

    # 4) map
    # remove_cols: 只保留 tokenize 需要的原始字段
    if task_type == "classification":
        keep_raw = [text_key, label_key]
    else:
        keep_raw = [text_key]

    remove_cols = [c for c in ds["train"].column_names if c not in keep_raw]
    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)

    # 5) rename label -> labels (仅分类)
    if task_type == "classification":
        def rename_label(batch):
            batch["labels"] = batch[label_key]
            return batch

        ds_tok = ds_tok.map(rename_label, batched=False)

    # 6) set format torch
    # classification 要显式包含 labels
    cols = ["input_ids", "attention_mask"]
    if task_type == "classification":
        cols.append("labels")

    for split in ds_tok.keys():
        keep = [c for c in cols if c in ds_tok[split].column_names]
        ds_tok[split].set_format(type="torch", columns=keep)

    # 7) collator
    if task_type == "classification":
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        # causal_lm: 让 collator 自动构造 labels = input_ids
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

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

    # 你原来的保护逻辑：只对“分类任务 + 原生无 test”的情况做 labels==-1 检查
    if task_type == "classification" and not has_test:
        try:
            if "labels" in test_ds.column_names:
                sample = test_ds.select(range(min(200, len(test_ds))))
                labels = sample["labels"]
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
