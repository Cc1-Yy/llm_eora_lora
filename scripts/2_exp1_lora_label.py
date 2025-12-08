# scripts/2_exp1_lora_label.py
import os
from dataclasses import dataclass
from typing import List

import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


@dataclass
class LoraTrainConfig:
    # 模型与数据路径
    model_name: str = "gpt2-medium"  # Base model
    train_path: str = "data/wikitext2_like/train.txt"
    valid_path: str = "data/wikitext2_like/valid.txt"

    # LoRA 实验 1 的输出目录
    output_dir: str = "outputs/exp1/lora_label_r8"

    # 训练相关
    block_size: int = 256
    num_epochs: int = 1         # 先跑通，之后你可以改大一点
    batch_size: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    seed: int = 42

    # LoRA 配置（你可以之后 sweep r=4,8,16...）
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # GPT-2 里常用的 target modules：c_attn, c_proj
    target_modules: tuple = ("c_attn", "c_proj")


class TextBlockDataset(Dataset):
    """
    跟 1_train_optimized_minimal 里的逻辑一样：
    - 读一个纯文本文件
    - tokenizer 编码成一条长序列
    - 切成 block_size 大小的训练样本
    """

    def __init__(self, file_path: str, tokenizer, block_size: int = 256):
        assert os.path.exists(file_path), f"File not found: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        encodings = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        input_ids: List[int] = encodings["input_ids"]

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
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # causal LM: 预测下一个 token
        }


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    if n_batches == 0:
        return float("nan")
    return total_loss / n_batches


def main():
    cfg = LoraTrainConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载 tokenizer（跟 Optimized 一致）
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 构造数据集 / dataloader
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

    # 3. 加载 Base Model
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(device)

    # 4. 应用 LoRA（label-supervised 微调）
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 5. 优化器（只对 LoRA 可训练参数等）
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    global_step = 0

    # 6. 训练循环（label-supervised：目标就是语言模型的下一个 token）
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

        # 每个 epoch 做一次验证
        val_loss = evaluate(model, valid_loader, device)
        if not math.isnan(val_loss):
            try:
                ppl = math.exp(val_loss)
            except OverflowError:
                ppl = float("inf")
            print(f"  >> validation loss: {val_loss:.4f}, ppl: {ppl:.2f}")
        else:
            print("  >> no valid batches, skip validation metrics")

    # 7. 只保存 LoRA adapter（PEFT 的 save_pretrained）
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Finished LoRA label-supervised training for Exp1.")
    print(f"LoRA adapter saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
