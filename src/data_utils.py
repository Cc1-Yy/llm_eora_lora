from __future__ import annotations

from typing import Dict, Any

from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)


def _get_text_and_label_keys(dataset_name: str) -> Dict[str, str]:
    """
    Return text/label field names for common CLASSIFICATION datasets.
    Extend this mapping as you add more datasets.
    """
    name = dataset_name.lower()
    if name in ["sst2", "glue/sst2", "glue"]:
        return {"text": "sentence", "label": "label"}
    if name in ["ag_news", "yelp_polarity"]:
        return {"text": "text", "label": "label"}

    return {"text": "text", "label": "label"}


def build_tokenize_fn(tokenizer, text_key: str, max_length: int):
    def fn(batch):
        return tokenizer(
            batch[text_key],
            truncation=True,
            max_length=max_length,
        )
    return fn


def _maybe_build_internal_test_for_sst2(
    ds: DatasetDict,
    config: Dict[str, Any],
) -> DatasetDict:
    data_cfg = config.get("data", {})
    use_internal_test = bool(data_cfg.get("use_internal_test", False))
    internal_test_size = float(data_cfg.get("internal_test_size", 0.1))
    seed = int(config.get("seed", 42))

    if not use_internal_test:
        return ds

    split = ds["train"].train_test_split(
        test_size=internal_test_size,
        seed=seed,
        shuffle=True,
    )

    new_ds = DatasetDict()
    new_ds["train"] = split["train"]
    if "validation" in ds:
        new_ds["validation"] = ds["validation"]
    else:
        val_split = split["train"].train_test_split(test_size=0.1, seed=seed)
        new_ds["train"] = val_split["train"]
        new_ds["validation"] = val_split["test"]

    new_ds["test"] = split["test"]

    return new_ds


def get_dataloaders(config: Dict[str, Any], tokenizer):
    """
    Supports:
      - classification
      - causal_lm

    Expected config structure:
    config = {
        "task_type": "classification" | "causal_lm",
        "num_labels": 2,
        "data": {
            "dataset_name": "glue/sst2" | "wikitext" | ...,
            "dataset_config_name": None,
            "max_length": 128,
            "batch_size": 8,
            "num_workers": 0,
            "text_key": optional,
            "use_internal_test": optional,
            "internal_test_size": optional,
        },
        "seed": 42,
    }
    """
    task_type = config.get("task_type", "classification")
    num_labels = int(config.get("num_labels", 2))

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
    canonical_name = dataset_name
    if dataset_name.lower() in ["glue/sst2", "sst2"]:
        ds = load_dataset("glue", "sst2")
        canonical_name = "glue/sst2"

        if task_type == "classification":
            ds = _maybe_build_internal_test_for_sst2(ds, config)

    else:
        if "/" in dataset_name and dataset_config_name is None:
            parts = dataset_name.split("/")
            ds = load_dataset(parts[0], parts[1])
            canonical_name = dataset_name
        else:
            ds = (
                load_dataset(dataset_name, dataset_config_name)
                if dataset_config_name
                else load_dataset(dataset_name)
            )
            canonical_name = dataset_name

    if "train" in ds:
        print(f"Loaded dataset columns: {ds['train'].column_names}")

    # 2) infer keys
    if task_type == "classification":
        keys = _get_text_and_label_keys(canonical_name)
        text_key, label_key = keys["text"], keys["label"]
    else:
        text_key = text_key_override or _get_text_and_label_keys(canonical_name)["text"]
        label_key = None

    # 3) tokenize
    tokenize_fn = build_tokenize_fn(tokenizer, text_key, max_length)

    # 4) check splits
    has_val = "validation" in ds
    has_test = "test" in ds

    # 5) map (remove unused raw columns)
    if task_type == "classification":
        keep_raw = [text_key, label_key]
    else:
        keep_raw = [text_key]

    ds_tok = DatasetDict()
    for split_name in ds.keys():
        split_ds = ds[split_name]
        remove_cols = [c for c in split_ds.column_names if c not in keep_raw]
        ds_tok[split_name] = split_ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=remove_cols,
        )

    # 6) filter empty examples for LM datasets
    if task_type != "classification":
        for split in list(ds_tok.keys()):
            ds_tok[split] = ds_tok[split].filter(
                lambda ex: len(ex.get("input_ids", [])) > 0
            )

    # 7) rename label -> labels (classification only)
    if task_type == "classification":
        def rename_label(example):
            example["labels"] = example[label_key]
            return example

        for sp in list(ds_tok.keys()):
            ds_tok[sp] = ds_tok[sp].map(rename_label, batched=False)

        # 8) label sanity check（覆盖所有存在的 split）
        def check_labels(example):
            v = example.get("labels", -1)
            if v == -1:
                return example
            if v < 0 or v >= num_labels:
                raise ValueError(
                    f"Invalid label value {v} detected. "
                    f"Labels must be in [0, {num_labels-1}]."
                )
            return example

        for sp in list(ds_tok.keys()):
            ds_tok[sp] = ds_tok[sp].map(check_labels, batched=False)

        for sp in list(ds_tok.keys()):
            if "labels" in ds_tok[sp].column_names:
                ds_tok[sp] = ds_tok[sp].filter(lambda ex: ex["labels"] != -1)

    # 9) set format torch
    cols = ["input_ids", "attention_mask"]
    if task_type == "classification":
        cols.append("labels")

    for split in ds_tok.keys():
        keep = [c for c in cols if c in ds_tok[split].column_names]
        ds_tok[split].set_format(type="torch", columns=keep)

    # 10) collator
    if task_type == "classification":
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

    # 11) choose splits
    train_ds = ds_tok["train"]
    val_ds = ds_tok["validation"] if has_val else None
    test_ds = ds_tok["test"] if has_test else None

    if val_ds is None:
        split = train_ds.train_test_split(
            test_size=0.1,
            seed=int(config.get("seed", 42))
        )
        train_ds, val_ds = split["train"], split["test"]

    if test_ds is None:
        test_ds = val_ds

    # 12) dataloaders
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

    try:
        print(f"[Data] train size: {len(train_ds)}")
        print(f"[Data] val size:   {len(val_ds)}")
        print(f"[Data] test size:  {len(test_ds)}")
        print(f"[Data] test batches: {len(test_loader)}")
    except Exception:
        pass

    return train_loader, val_loader, test_loader
