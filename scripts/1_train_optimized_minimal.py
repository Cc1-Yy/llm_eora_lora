import os
from dataclasses import dataclass, field
from typing import List

import math
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
)


@dataclass
class TrainConfig:
    model_name: str = "gpt2-medium"
    train_path: str = "data/wikitext2_like/train.txt"
    valid_path: str = "data/wikitext2_like/valid.txt"
    output_dir: str = "outputs/optimized/gpt2-medium-wt2-full-minimal"

    block_size: int = 256
    num_epochs: int = 1          # 先跑通，确认没问题再加大
    batch_size: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 4
    seed: int = 42


class TextBlockDataset(Dataset):
    """
    非 HuggingFace datasets 版本：
    - 读一个纯文本文件
    - 用 tokenizer 编码成一长串 token
    - 切成 block_size 的小块，每块作为一个训练样本
    """

    def __init__(self, file_path: str, tokenizer, block_size: int = 256):
        assert os.path.exists(file_path), f"File not found: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 把整个文本编码成一条长长的 token 序列
        encodings = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_ids: List[int] = encodings["input_ids"]

        # 切成整块
        n_blocks = len(input_ids) // block_size
        input_ids = input_ids[: n_blocks * block_size]

        self.examples = []
        for i in range(n_blocks):
            block = input_ids[i * block_size : (i + 1) * block_size]
            self.examples.append(torch.tensor(block, dtype=torch.long))

        self.block_size = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        # 因果语言模型：labels 就是 input_ids 本身
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = TrainConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 构造 Dataset & DataLoader（不依赖 datasets / pyarrow / Trainer）
    train_dataset = TextBlockDataset(cfg.train_path, tokenizer, cfg.block_size)
    valid_dataset = TextBlockDataset(cfg.valid_path, tokenizer, cfg.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"# train batches: {len(train_loader)}, # valid batches: {len(valid_loader)}")

    # 3. 加载 Base Model（预训练的 gpt2-medium）
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))  # 确保 embedding 大小匹配
    model.to(device)

    # 4. 优化器
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    global_step = 0

    # 5. 训练循环
    model.train()
    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"  step {step + 1}, train loss: {avg_loss:.4f}")
                running_loss = 0.0

        # 每个 epoch 结束后做一次简单验证
        val_loss = evaluate(model, valid_loader, device)
        try:
            ppl = math.exp(val_loss)
        except OverflowError:
            ppl = float("inf")

        print(f"  >> validation loss: {val_loss:.4f}, ppl: {ppl:.2f}")

    # 6. 保存 Optimized Model
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Finished training optimized model. Saved to: {cfg.output_dir}")


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


if __name__ == "__main__":
    main()
