# src/data_utils.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

from torch.utils.data import DataLoader

from datasets import (
    load_dataset,
    load_from_disk,
    DatasetDict,
)

from transformers import (
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)

# ============================================================
# TOP-LEVEL helpers (Windows-friendly: no lambda / no closures)
# ============================================================

def _get_text_and_label_keys(dataset_name: str) -> Dict[str, str]:
    name = (dataset_name or "").lower()
    if name in ["sst2", "glue/sst2", "glue"]:
        return {"text": "sentence", "label": "label"}
    if name in ["ag_news", "yelp_polarity"]:
        return {"text": "text", "label": "label"}
    return {"text": "text", "label": "label"}


def _is_iterable_dataset(ds) -> bool:
    return ds.__class__.__name__ == "IterableDataset"


def _safe_column_names(ds) -> List[str]:
    cols = getattr(ds, "column_names", None)
    if cols is not None:
        return list(cols)
    feats = getattr(ds, "features", None)
    if feats is not None:
        return list(feats.keys())
    return []


def _filter_nonempty_input_ids(ex: Dict[str, Any]) -> bool:
    ids = ex.get("input_ids", None)
    return isinstance(ids, list) and len(ids) > 0


def _label_not_minus_one(ex: Dict[str, Any]) -> bool:
    return ex.get("labels", -1) != -1


def _tokenize_batch_cls(
    batch: Dict[str, Any],
    tokenizer,
    text_key: str,
    max_length: int,
) -> Dict[str, Any]:
    return tokenizer(
        batch[text_key],
        truncation=True,
        max_length=max_length,
    )


def _tokenize_batch_lm(
    batch: Dict[str, Any],
    tokenizer,
    text_key: str,
    max_length: int,
) -> Dict[str, Any]:
    # causal LM: truncation to max_length
    # labels will be created by DataCollatorForLanguageModeling(mlm=False)
    return tokenizer(
        batch[text_key],
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )


def _format_for_torch(ds, columns: List[str]):
    """
    For non-iterable HF Dataset, set torch format so DataLoader gives tensors.
    For streaming/IterableDataset: do nothing.
    """
    if _is_iterable_dataset(ds):
        return ds
    if hasattr(ds, "set_format"):
        keep = [c for c in columns if c in _safe_column_names(ds)]
        ds.set_format(type="torch", columns=keep)
    return ds


def _dl_kwargs_from_config(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    persistent_workers = bool(data_cfg.get("persistent_workers", True)) if num_workers > 0 else False
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    kw = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        kw["prefetch_factor"] = prefetch_factor
    return kw


def _get_split_seed(config: Dict[str, Any]) -> int:
    """
    Separate seed used only for dataset splitting / partitioning.
    Falls back to top-level seed for backward compatibility.
    """
    data_cfg = config.get("data", {})
    return int(data_cfg.get("split_seed", config.get("seed", 42)))


def _maybe_build_internal_test_for_sst2(ds: DatasetDict, config: Dict[str, Any]) -> DatasetDict:
    data_cfg = config.get("data", {})
    use_internal_test = bool(data_cfg.get("use_internal_test", False))
    internal_test_size = float(data_cfg.get("internal_test_size", 0.1))
    split_seed = _get_split_seed(config)

    if not use_internal_test:
        return ds

    split = ds["train"].train_test_split(
        test_size=internal_test_size,
        seed=split_seed,
        shuffle=True,
    )

    new_ds = DatasetDict()
    new_ds["train"] = split["train"]

    if "validation" in ds:
        new_ds["validation"] = ds["validation"]
    else:
        val_split = split["train"].train_test_split(test_size=0.1, seed=split_seed)
        new_ds["train"] = val_split["train"]
        new_ds["validation"] = val_split["test"]

    new_ds["test"] = split["test"]
    return new_ds


# ============================================================
# Collators (TOP-LEVEL classes => picklable for num_workers>0)
# ============================================================

class LMKeepKeysCollator:
    """
    Wrap a HF LM collator but drop extra keys robustly.
    Must be picklable on Windows => use a class, not a closure.
    """
    def __init__(self, base_collator):
        self.base_collator = base_collator
        self.keep = {"input_ids", "attention_mask", "labels"}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        new_feats = []
        for f in features:
            nf = {k: v for k, v in f.items() if k in self.keep}
            if "input_ids" not in nf:
                raise ValueError(f"[LM] Feature missing input_ids. keys={list(f.keys())}")
            new_feats.append(nf)
        return self.base_collator(new_feats)


# ============================================================
# LM corpus loader (local / online / streaming)
# ============================================================

def _load_corpus_from_spec(spec: Dict[str, Any], *, seed: int, default_split: str = "train"):
    """
    spec supports:
      - local_disk_path: load_from_disk, then select split
      OR
      - name / config / split / streaming
    optional:
      - max_train_samples (works for both streaming and non-streaming)
      - shuffle_buffer (streaming only)
    """
    if spec is None:
        raise ValueError("corpus spec is None")

    # local dataset on disk
    if spec.get("local_disk_path"):
        ds_disk = load_from_disk(spec["local_disk_path"])
        split = spec.get("split", default_split)
        if isinstance(ds_disk, DatasetDict):
            if split not in ds_disk:
                raise ValueError(
                    f"Split '{split}' not found in load_from_disk({spec['local_disk_path']}). "
                    f"Available: {list(ds_disk.keys())}"
                )
            ds = ds_disk[split]
        else:
            ds = ds_disk

        max_n = spec.get("max_train_samples", None)
        if max_n is not None:
            max_n = int(max_n)
            ds = ds.select(range(min(max_n, len(ds))))
        return ds

    # online dataset (HF)
    name = spec.get("name", None)
    if not name:
        raise ValueError("corpus spec must have local_disk_path OR name")

    cfg = spec.get("config", None)
    split = spec.get("split", default_split)
    streaming = bool(spec.get("streaming", False))

    if cfg is not None:
        ds = load_dataset(name, cfg, split=split, streaming=streaming)
    else:
        ds = load_dataset(name, split=split, streaming=streaming)

    # streaming shuffle (only for iterable)
    if streaming:
        buf = int(spec.get("shuffle_buffer", 0) or 0)
        if buf > 0:
            ds = ds.shuffle(buffer_size=buf, seed=seed)

    # optional limit
    max_n = spec.get("max_train_samples", None)
    if max_n is not None:
        max_n = int(max_n)
        if streaming:
            ds = ds.take(max_n)
        else:
            ds = ds.select(range(min(max_n, len(ds))))

    return ds


# ============================================================
# Main API
# ============================================================

def get_dataloaders(config: Dict[str, Any], tokenizer):
    task_type = (config.get("task_type", "classification") or "classification").lower()
    if task_type == "lm":
        task_type = "causal_lm"

    num_labels = int(config.get("num_labels", 2))
    seed = int(config.get("seed", 42))
    split_seed = _get_split_seed(config)
    data_cfg = config.get("data", {})
    dl_kwargs = _dl_kwargs_from_config(data_cfg)

    # ============================================================
    # A) NEW LM pipeline: data.train_corpus / eval_corpus / test_corpus
    # ============================================================
    if task_type == "causal_lm" and "train_corpus" in data_cfg:
        train_spec = data_cfg["train_corpus"]
        eval_spec = data_cfg.get("eval_corpus", None)
        test_spec = data_cfg.get("test_corpus", None)
        if eval_spec is None:
            raise ValueError("For LM with data.train_corpus, you must provide data.eval_corpus.")

        max_length = int(data_cfg.get("max_length", 1024))
        batch_size = int(data_cfg.get("batch_size", 2))

        pad_to_multiple_of = data_cfg.get("pad_to_multiple_of", 8)
        pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of is not None else None

        # load raw corpora
        train_raw = _load_corpus_from_spec(train_spec, seed=seed, default_split="train")
        val_raw = _load_corpus_from_spec(eval_spec, seed=seed, default_split="validation")
        test_raw = _load_corpus_from_spec(test_spec, seed=seed, default_split="test") if test_spec else val_raw

        text_key_train = train_spec.get("text_key", "text")
        text_key_val = eval_spec.get("text_key", "text")
        text_key_test = (test_spec.get("text_key", "text") if test_spec else text_key_val)

        def _tok_remove_cols(ds, text_key: str, desc: str = ""):
            """
            Tokenize LM dataset with HuggingFace datasets (Windows-friendly):
              - no lambda
              - no nested functions used by DataLoader workers
              - supports map-style and streaming
            """
            is_streaming = _is_iterable_dataset(ds)

            if is_streaming:
                # IterableDataset.map() does not always accept remove_columns/desc consistently
                ds_tok = ds.map(
                    _tokenize_batch_lm,
                    batched=True,
                    fn_kwargs={"tokenizer": tokenizer, "text_key": text_key, "max_length": max_length},
                )
            else:
                colnames = getattr(ds, "column_names", None)
                remove_cols = list(colnames) if colnames is not None else None
                ds_tok = ds.map(
                    _tokenize_batch_lm,
                    batched=True,
                    fn_kwargs={"tokenizer": tokenizer, "text_key": text_key, "max_length": max_length},
                    remove_columns=remove_cols,
                    desc=desc if desc else None,
                )

            if hasattr(ds_tok, "filter"):
                ds_tok = ds_tok.filter(_filter_nonempty_input_ids)

            return ds_tok

        train_ds = _tok_remove_cols(train_raw, text_key_train, desc="Tokenize LM train")
        val_ds = _tok_remove_cols(val_raw, text_key_val, desc="Tokenize LM val")
        test_ds = _tok_remove_cols(test_raw, text_key_test, desc="Tokenize LM test")

        # torch format
        cols = ["input_ids", "attention_mask"]
        train_ds = _format_for_torch(train_ds, cols)
        val_ds = _format_for_torch(val_ds, cols)
        test_ds = _format_for_torch(test_ds, cols)

        base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        collator = LMKeepKeysCollator(base_collator)

        # shuffle:
        # - streaming: must be False (use dataset.shuffle(...) instead)
        # - map-style: True is better
        shuffle_train = not _is_iterable_dataset(train_ds)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collator,
            **dl_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            **dl_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            **dl_kwargs,
        )

        # logging sizes
        try:
            print(f"[LM-Data] train size: {len(train_ds)}")
        except Exception:
            print("[LM-Data] train size: (iterable/unknown)")
        try:
            print(f"[LM-Data] val size:   {len(val_ds)}")
            print(f"[LM-Data] test size:  {len(test_ds)}")
        except Exception:
            pass

        return train_loader, val_loader, test_loader

    # ============================================================
    # B) LEGACY pipeline: data.dataset_name ...
    # ============================================================
    dataset_name = data_cfg.get("dataset_name")
    dataset_config_name = data_cfg.get("dataset_config_name", None)
    max_length = int(data_cfg.get("max_length", 128))
    batch_size = int(data_cfg.get("batch_size", 8))
    text_key_override = data_cfg.get("text_key", None)

    if dataset_name is None:
        raise ValueError("config['data']['dataset_name'] is required (or use data.train_corpus for LM).")

    canonical_name = dataset_name

    # load dataset
    if dataset_name.lower() in ["glue/sst2", "sst2"]:
        ds = load_dataset("glue", "sst2")
        canonical_name = "glue/sst2"
        if task_type == "classification":
            ds = _maybe_build_internal_test_for_sst2(ds, config)
    else:
        if "/" in dataset_name and dataset_config_name is None:
            parts = dataset_name.split("/", 1)
            ds = load_dataset(parts[0], parts[1])
            canonical_name = dataset_name
        else:
            ds = load_dataset(dataset_name, dataset_config_name) if dataset_config_name else load_dataset(dataset_name)
            canonical_name = dataset_name

    if "train" in ds:
        try:
            print(f"Loaded dataset columns: {ds['train'].column_names}")
        except Exception:
            pass

    # determine keys
    if task_type == "classification":
        keys = _get_text_and_label_keys(canonical_name)
        text_key, label_key = keys["text"], keys["label"]
    else:
        text_key = text_key_override or _get_text_and_label_keys(canonical_name)["text"]
        label_key = None

    has_val = "validation" in ds
    has_test = ("test" in ds) or ("test.py" in ds)  # backward-compat

    ds_tok = DatasetDict()

    # tokenize each split
    for split_name in ds.keys():
        split_ds = ds[split_name]

        if task_type == "classification":
            keep_raw = [text_key, label_key]
            remove_cols = [c for c in split_ds.column_names if c not in keep_raw]
            ds_tok[split_name] = split_ds.map(
                _tokenize_batch_cls,
                batched=True,
                fn_kwargs={"tokenizer": tokenizer, "text_key": text_key, "max_length": max_length},
                remove_columns=remove_cols,
                desc=f"Tokenize {split_name}",
            )
        else:
            ds_tok[split_name] = split_ds.map(
                _tokenize_batch_lm,
                batched=True,
                fn_kwargs={"tokenizer": tokenizer, "text_key": text_key, "max_length": max_length},
                remove_columns=list(split_ds.column_names),
                desc=f"Tokenize {split_name}",
            )
            if hasattr(ds_tok[split_name], "filter"):
                ds_tok[split_name] = ds_tok[split_name].filter(
                    _filter_nonempty_input_ids,
                    desc=f"Filter {split_name} nonempty",
                )

    # classification: add "labels"
    if task_type == "classification":
        def _rename_label(example: Dict[str, Any]) -> Dict[str, Any]:
            example["labels"] = example[label_key]
            return example

        for sp in list(ds_tok.keys()):
            ds_tok[sp] = ds_tok[sp].map(_rename_label, batched=False, desc=f"Set labels {sp}")

        def _check_labels(example: Dict[str, Any]) -> Dict[str, Any]:
            v = example.get("labels", -1)
            if v == -1:
                return example
            if v < 0 or v >= num_labels:
                raise ValueError(f"Invalid label value {v} detected. Labels must be in [0, {num_labels - 1}].")
            return example

        for sp in list(ds_tok.keys()):
            ds_tok[sp] = ds_tok[sp].map(_check_labels, batched=False, desc=f"Check labels {sp}")
            if "labels" in ds_tok[sp].column_names and hasattr(ds_tok[sp], "filter"):
                ds_tok[sp] = ds_tok[sp].filter(_label_not_minus_one, desc=f"Filter labels!=-1 {sp}")

    # choose splits
    train_ds = ds_tok["train"]
    val_ds = ds_tok["validation"] if has_val else None
    if "test" in ds_tok:
        test_ds = ds_tok["test"]
    elif "test.py" in ds_tok:
        test_ds = ds_tok["test.py"]
    else:
        test_ds = None

    if val_ds is None:
        split = train_ds.train_test_split(test_size=0.1, seed=split_seed)
        train_ds, val_ds = split["train"], split["test"]
    if test_ds is None:
        test_ds = val_ds

    # collator + torch format
    if task_type == "classification":
        cols = ["input_ids", "attention_mask", "labels"]
        train_ds = _format_for_torch(train_ds, cols)
        val_ds = _format_for_torch(val_ds, cols)
        test_ds = _format_for_torch(test_ds, cols)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        shuffle_train = True
    else:
        cols = ["input_ids", "attention_mask"]
        train_ds = _format_for_torch(train_ds, cols)
        val_ds = _format_for_torch(val_ds, cols)
        test_ds = _format_for_torch(test_ds, cols)
        pad_to_multiple_of = data_cfg.get("pad_to_multiple_of", 8)
        pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of is not None else None
        base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        collator = LMKeepKeysCollator(base_collator)
        shuffle_train = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collator,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        **dl_kwargs,
    )

    try:
        print(f"[Data] split_seed:  {split_seed}")
        print(f"[Data] train size:   {len(train_ds)}")
        print(f"[Data] val size:     {len(val_ds)}")
        print(f"[Data] test size:    {len(test_ds)}")
        print(f"[Data] test batches: {len(test_loader)}")
    except Exception:
        pass

    return train_loader, val_loader, test_loader