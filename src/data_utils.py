# src/data_utils.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling


# ----------------------------
# Helpers
# ----------------------------
def _get_text_and_label_keys(dataset_name: str) -> Dict[str, str]:
    """
    Return text/label field names for common CLASSIFICATION datasets.
    Extend this mapping as you add more datasets.
    """
    name = (dataset_name or "").lower()
    if name in ["sst2", "glue/sst2", "glue"]:
        return {"text": "sentence", "label": "label"}
    if name in ["ag_news", "yelp_polarity"]:
        return {"text": "text", "label": "label"}
    return {"text": "text", "label": "label"}


def build_tokenize_fn(tokenizer, text_key: str, max_length: int):
    """
    Tokenize a batch of texts.
    NOTE: For LM we still do truncation to max_length. Padding is handled by collator.
    """
    def fn(batch):
        return tokenizer(
            batch[text_key],
            truncation=True,
            max_length=max_length,
        )
    return fn


def _is_iterable_dataset(ds) -> bool:
    # HF datasets streaming returns datasets.iterable_dataset.IterableDataset
    return ds.__class__.__name__ == "IterableDataset"


def _safe_column_names(ds) -> list:
    """
    HF datasets:
      - map-style Dataset: ds.column_names is a list
      - streaming IterableDataset: ds.column_names can be None
    """
    cols = getattr(ds, "column_names", None)
    if cols is not None:
        return list(cols)
    feats = getattr(ds, "features", None)
    if feats is not None:
        return list(feats.keys())
    return []


def _maybe_build_internal_test_for_sst2(ds: DatasetDict, config: Dict[str, Any]) -> DatasetDict:
    """
    If use_internal_test=true, split ds["train"] into (train,test.py) locally,
    keep original validation as validation if exists.
    """
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
        # Fallback: create validation from train
        val_split = split["train"].train_test_split(test_size=0.1, seed=seed)
        new_ds["train"] = val_split["train"]
        new_ds["validation"] = val_split["test.py"]

    new_ds["test.py"] = split["test.py"]
    return new_ds


def _wrap_drop_non_token_keys_for_lm(collator):
    """
    HF DataCollatorForLanguageModeling will call tokenizer.pad(features).
    If features contain raw strings like {"text": "..."} it will crash.
    This wrapper force-drops non-token keys for both streaming and map-style datasets.
    """
    keep = {"input_ids", "attention_mask", "labels"}  # LM collator may create labels
    def wrapped(features):
        new_feats = []
        for f in features:
            nf = {k: v for k, v in f.items() if k in keep}
            if "input_ids" not in nf:
                raise ValueError(f"[LM] Feature missing input_ids. keys={list(f.keys())}")
            new_feats.append(nf)
        return collator(new_feats)
    return wrapped


def _tokenize_lm_dataset(ds, tokenizer, text_key: str, max_length: int):
    """
    Tokenize an LM dataset and ENSURE raw text columns are removed (when possible).
    For streaming datasets where column_names may be None, we can't reliably remove;
    in that case collator wrapper will still protect us.
    """
    tok_fn = build_tokenize_fn(tokenizer, text_key=text_key, max_length=max_length)

    colnames = getattr(ds, "column_names", None)
    if colnames is None:
        # streaming: sometimes no column_names -> map without remove_columns
        ds_tok = ds.map(tok_fn, batched=True)
    else:
        # IMPORTANT: remove ALL original columns including text_key
        ds_tok = ds.map(tok_fn, batched=True, remove_columns=list(colnames))

    # Filter empty
    if hasattr(ds_tok, "filter"):
        ds_tok = ds_tok.filter(lambda ex: len(ex.get("input_ids", [])) > 0)

    return ds_tok


def _format_for_torch(ds, columns: list):
    """
    For map-style Dataset: set_format works well.
    For IterableDataset: do NOT force format conversion (may not support columns, may keep raw strings).
    """
    if _is_iterable_dataset(ds):
        return ds  # let collator tensorize
    if hasattr(ds, "set_format"):
        keep = [c for c in columns if c in _safe_column_names(ds)]
        ds.set_format(type="torch", columns=keep)
    return ds


# ----------------------------
# Main API
# ----------------------------
def get_dataloaders(config: Dict[str, Any], tokenizer):
    """
    Supports:
      - classification (HF datasets)
      - causal_lm / lm

    LM supports 2 ways:
      A) NEW (recommended for big LM): data.train_corpus / data.eval_corpus / data.test_corpus
      B) LEGACY: data.dataset_name (+ dataset_config_name)
    """
    task_type = config.get("task_type", "classification")
    if task_type == "lm":
        task_type = "causal_lm"

    num_labels = int(config.get("num_labels", 2))
    seed = int(config.get("seed", 42))

    data_cfg = config.get("data", {})

    # ============================================================
    # A) NEW LM pipeline: train_corpus / eval_corpus / test_corpus
    # ============================================================
    if task_type == "causal_lm" and "train_corpus" in data_cfg:
        train_spec = data_cfg["train_corpus"]
        eval_spec = data_cfg.get("eval_corpus", None)
        test_spec = data_cfg.get("test_corpus", None)

        if eval_spec is None:
            raise ValueError("For LM with data.train_corpus, you must provide data.eval_corpus.")

        max_length = int(data_cfg.get("max_length", 1024))
        batch_size = int(data_cfg.get("batch_size", 2))
        num_workers = int(data_cfg.get("num_workers", 0))

        def _load_corpus(spec: Dict[str, Any]):
            # Local cache
            if spec.get("local_disk_path"):
                ds_disk = load_from_disk(spec["local_disk_path"])
                split = spec.get("split", "validation")
                if split not in ds_disk:
                    raise ValueError(
                        f"Split '{split}' not found in load_from_disk({spec['local_disk_path']}). "
                        f"Available: {list(ds_disk.keys())}"
                    )
                return ds_disk[split]

            # HF dataset
            name = spec["name"]
            cfg = spec.get("config", None)
            split = spec.get("split", "train")
            streaming = bool(spec.get("streaming", False))

            if cfg is not None:
                ds_hf = load_dataset(name, cfg, split=split, streaming=streaming)
            else:
                ds_hf = load_dataset(name, split=split, streaming=streaming)

            if streaming:
                ds_hf = ds_hf.shuffle(
                    buffer_size=int(spec.get("shuffle_buffer", 10000)),
                    seed=seed,
                )

            max_n = spec.get("max_train_samples", None)
            if max_n is not None:
                max_n = int(max_n)
                if streaming:
                    ds_hf = ds_hf.take(max_n)
                else:
                    ds_hf = ds_hf.select(range(min(max_n, len(ds_hf))))

            return ds_hf

        # Load corpora
        train_raw = _load_corpus(train_spec)
        val_raw = _load_corpus(eval_spec)
        test_raw = _load_corpus(test_spec) if test_spec else val_raw

        text_key_train = train_spec.get("text_key", "text")
        text_key_val = eval_spec.get("text_key", "text")
        text_key_test = (test_spec.get("text_key", "text") if test_spec else text_key_val)

        # Tokenize and drop raw columns (when possible)
        train_ds = _tokenize_lm_dataset(train_raw, tokenizer, text_key_train, max_length)
        val_ds = _tokenize_lm_dataset(val_raw, tokenizer, text_key_val, max_length)
        test_ds = _tokenize_lm_dataset(test_raw, tokenizer, text_key_test, max_length)

        # Torch format only for map-style datasets
        cols = ["input_ids", "attention_mask"]
        train_ds = _format_for_torch(train_ds, cols)
        val_ds = _format_for_torch(val_ds, cols)
        test_ds = _format_for_torch(test_ds, cols)

        # Collator with strict key filtering (prevents 'text' crash)
        base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        collator = _wrap_drop_non_token_keys_for_lm(base_collator)

        # NOTE: streaming IterableDataset cannot do shuffle=True in DataLoader
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
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

        # Logging
        try:
            print(f"[LM-Data] train size: {len(train_ds)}")
        except Exception:
            print("[LM-Data] train size: (iterable/unknown)")
        try:
            print(f"[LM-Data] val size:   {len(val_ds)}")
            print(f"[LM-Data] test.py size:  {len(test_ds)}")
        except Exception:
            pass

        return train_loader, val_loader, test_loader

    # ==========================================
    # B) LEGACY pipeline: data.dataset_name ...
    # ==========================================
    dataset_name = data_cfg.get("dataset_name")
    dataset_config_name = data_cfg.get("dataset_config_name", None)
    max_length = int(data_cfg.get("max_length", 128))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    text_key_override = data_cfg.get("text_key", None)

    if dataset_name is None:
        raise ValueError("config['data']['dataset_name'] is required (or use data.train_corpus for LM).")

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
            ds = load_dataset(dataset_name, dataset_config_name) if dataset_config_name else load_dataset(dataset_name)
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

    # 4) splits
    has_val = "validation" in ds
    has_test = "test.py" in ds

    ds_tok = DatasetDict()

    for split_name in ds.keys():
        split_ds = ds[split_name]

        if task_type == "classification":
            # keep only raw text + label prior to tokenize removal
            keep_raw = [text_key, label_key]
            remove_cols = [c for c in split_ds.column_names if c not in keep_raw]
            ds_tok[split_name] = split_ds.map(
                tokenize_fn,
                batched=True,
                remove_columns=remove_cols,
            )
        else:
            # LM: remove ALL original cols including text, keep only tokenizer outputs
            ds_tok[split_name] = split_ds.map(
                tokenize_fn,
                batched=True,
                remove_columns=list(split_ds.column_names),
            )
            ds_tok[split_name] = ds_tok[split_name].filter(lambda ex: len(ex.get("input_ids", [])) > 0)

    # 5) classification: rename label -> labels + sanity check
    if task_type == "classification":
        def rename_label(example):
            example["labels"] = example[label_key]
            return example

        for sp in list(ds_tok.keys()):
            ds_tok[sp] = ds_tok[sp].map(rename_label, batched=False)

        def check_labels(example):
            v = example.get("labels", -1)
            if v == -1:
                return example
            if v < 0 or v >= num_labels:
                raise ValueError(
                    f"Invalid label value {v} detected. Labels must be in [0, {num_labels-1}]."
                )
            return example

        for sp in list(ds_tok.keys()):
            ds_tok[sp] = ds_tok[sp].map(check_labels, batched=False)
            if "labels" in ds_tok[sp].column_names:
                ds_tok[sp] = ds_tok[sp].filter(lambda ex: ex["labels"] != -1)

    # 6) choose splits
    train_ds = ds_tok["train"]
    val_ds = ds_tok["validation"] if has_val else None
    test_ds = ds_tok["test.py"] if has_test else None

    if val_ds is None:
        split = train_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds, val_ds = split["train"], split["test.py"]
    if test_ds is None:
        test_ds = val_ds

    # 7) format + collator
    if task_type == "classification":
        cols = ["input_ids", "attention_mask", "labels"]
        train_ds = _format_for_torch(train_ds, cols)
        val_ds = _format_for_torch(val_ds, cols)
        test_ds = _format_for_torch(test_ds, cols)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        cols = ["input_ids", "attention_mask"]
        train_ds = _format_for_torch(train_ds, cols)
        val_ds = _format_for_torch(val_ds, cols)
        test_ds = _format_for_torch(test_ds, cols)
        base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        collator = _wrap_drop_non_token_keys_for_lm(base_collator)

    # 8) dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(task_type == "classification"),  # LM shuffle is typically dataset-level; keep False here
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
        print(f"[Data] test.py size:  {len(test_ds)}")
        print(f"[Data] test.py batches: {len(test_loader)}")
    except Exception:
        pass

    return train_loader, val_loader, test_loader
