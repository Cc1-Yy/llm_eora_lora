from __future__ import annotations
from typing import Dict, Any

import torch
from torch.optim import AdamW
from tqdm import tqdm


def train_optimized(model, train_loader, val_loader, config: Dict[str, Any], evaluate_fn):
    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    num_epochs = int(train_cfg.get("num_epochs", 3))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    device = next(model.parameters()).device

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = -1e9
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        loop = tqdm(train_loader, desc=f"[Optimized] Epoch {epoch}/{num_epochs}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            bs = batch["input_ids"].size(0)
            total_loss += loss.item() * bs
            total_examples += bs

            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / max(total_examples, 1)

        # 每个 epoch 评估一次
        val_metrics = evaluate_fn(model, val_loader, config)

        # 分类任务用 accuracy 作为 best 指标，否则用 -loss
        if config.get("task_type", "classification") == "classification":
            score = val_metrics.get("accuracy", -1.0)
        else:
            score = -val_metrics.get("loss", 1e9)

        if score > best_metric:
            best_metric = score
            # 存一份最优权重（内存版）
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val={val_metrics}")

    # 训练完恢复最优权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return model
