from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn, optim

from app.categories import labels_to_vector
from app.head import HeadNet


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def train_head(
    samples: list[dict],
    embedder: Embedder,
    epochs: int = 100,
    lr: float = 1e-2,
) -> tuple[nn.Module, dict]:
    texts = [s["text"] for s in samples]
    embeddings = np.asarray(embedder.embed(texts), dtype=np.float32)
    targets = np.array([labels_to_vector(s["labels"]) for s in samples], dtype=np.float32)

    x = torch.from_numpy(embeddings)
    y = torch.from_numpy(targets)

    pos_counts = y.sum(dim=0)
    neg_counts = (len(samples) - pos_counts).clamp(min=1)
    pos_weight = neg_counts / pos_counts.clamp(min=1)

    module = HeadNet(input_dim=x.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(module.parameters(), lr=lr)

    module.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(module(x), y)
        loss.backward()
        opt.step()

    module.eval()
    with torch.no_grad():
        final_loss = criterion(module(x), y).item()

    metrics = {"train_loss": final_loss, "train_samples": len(samples)}
    return module, metrics


def export_head_onnx(module: nn.Module, dest: Path) -> None:
    module.eval()
    input_dim = module.net[0].in_features
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        module,
        dummy,
        str(dest),
        input_names=["embedding"],
        output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the dynamo exporter writes weights to a separate *.onnx.data file by
        # default; ModelStorage only uploads/downloads the single .onnx key, so external data
        # would silently produce an unloadable model in the S3 registry.
    )
