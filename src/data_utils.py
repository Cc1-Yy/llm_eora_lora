# src/data_utils.py
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

def load_wikitext2(tokenizer: PreTrainedTokenizerBase, block_size: int = 512):
    """
    加载并处理 WikiText-2，返回 tokenized 的 train/validation/test 数据集。
    """
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=False,
            return_attention_mask=False,
            truncation=False,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing WikiText-2",
    )

    # 按 block_size 拼接成 language modeling 的连续块
    def group_texts(examples):
        # 把多个样本的 input_ids 接在一起
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)

        total_length = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_length]

        result = {
            "input_ids": [
                concatenated[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }
        # labels 就是 shifted language modeling 的目标，直接复制 input_ids
        result["labels"] = list(result["input_ids"])
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc=f"Grouping tokens into blocks of {block_size}",
    )

    return lm_datasets["train"], lm_datasets["validation"], lm_datasets["test"]
