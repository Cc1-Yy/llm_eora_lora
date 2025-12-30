from __future__ import annotations

from typing import Dict, Any

import torch


@torch.no_grad()
def evaluate(model, dataloader, config: Dict[str, Any]) -> Dict[str, float]:

    task_type = config.get("task_type", "classification")
    device = next(model.parameters()).device

    model.eval()

    total_loss = 0.0
    total_examples = 0

    correct = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        # 1) accumulate loss
        loss = outputs.loss
        bs = batch["input_ids"].size(0)

        total_loss += loss.item() * bs
        total_examples += bs

        # 2) accuracy for classification
        if task_type == "classification":
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            labels = batch["labels"]
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / max(total_examples, 1)

    if task_type == "classification":
        accuracy = correct / max(total_examples, 1)
        return {"loss": float(avg_loss), "accuracy": float(accuracy)}

    return {"loss": float(avg_loss)}
