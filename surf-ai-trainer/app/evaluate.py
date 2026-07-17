import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn

from app.categories import CATEGORIES, labels_to_vector


def holdout_split(samples: list[dict], holdout_fraction: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1 - holdout_fraction)))
    return shuffled[:split_at], shuffled[split_at:]


def metrics(module: nn.Module, embedder, holdout: list[dict]) -> dict[str, float]:
    texts = [s["text"] for s in holdout]
    targets = np.array([labels_to_vector(s["labels"]) for s in holdout], dtype=np.float32)

    embeddings = np.asarray(embedder.embed(texts), dtype=np.float32)
    module.eval()
    with torch.no_grad():
        logits = module(torch.from_numpy(embeddings)).numpy()
    predictions = (logits > 0).astype(np.float32)  # sigmoid(logit) > 0.5 <=> logit > 0

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average=None, labels=range(len(CATEGORIES)), zero_division=0
    )

    result: dict[str, float] = {}
    for i, category in enumerate(CATEGORIES):
        result[f"precision_{category}"] = float(precision[i])
        result[f"recall_{category}"] = float(recall[i])
        result[f"f1_{category}"] = float(f1[i])

    result["macro_f1"] = float(f1.mean())
    return result
