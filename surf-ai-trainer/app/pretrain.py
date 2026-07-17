from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from app.head import HeadNet


def _load_chunks(chunk_paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []
    for path in chunk_paths:
        data = np.load(path)
        embeddings.append(data["embeddings"])
        labels.append(data["labels"])
    return (
        torch.from_numpy(np.concatenate(embeddings, axis=0)),
        torch.from_numpy(np.concatenate(labels, axis=0)),
    )


def pretrain_head(
    embedding_chunk_paths: list[Path],
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
) -> tuple[HeadNet, dict]:
    x, y = _load_chunks(embedding_chunk_paths)

    pos_counts = y.sum(dim=0)
    neg_counts = (len(y) - pos_counts).clamp(min=1)
    pos_weight = neg_counts / pos_counts.clamp(min=1)

    module = HeadNet(input_dim=x.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(module.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    module.train()
    final_loss = 0.0
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            opt.zero_grad()
            loss = criterion(module(batch_x), batch_y)
            loss.backward()
            opt.step()
            final_loss = loss.item()

    module.eval()
    with torch.no_grad():
        full_loss = criterion(module(x), y).item()

    metrics = {"train_loss": full_loss, "train_samples": len(x)}
    return module, metrics


def save_checkpoint(module: HeadNet, dest: Path) -> None:
    torch.save(module.state_dict(), dest)


def load_checkpoint(path: Path, input_dim: int = 384) -> HeadNet:
    module = HeadNet(input_dim=input_dim)
    module.load_state_dict(torch.load(path, weights_only=True))
    return module


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="One-time bootstrap-corpus head pretraining. NOT run by pytest/CI.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("cache/bootstrap_embeddings"))
    parser.add_argument("--output", type=Path, default=Path("cache/base-checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    chunk_paths = sorted(args.embeddings_dir.glob("*.npz"))
    if not chunk_paths:
        parser.error(f"no .npz chunks found in {args.embeddings_dir}")

    module, metrics = pretrain_head(chunk_paths, args.epochs, args.batch_size, args.lr)
    save_checkpoint(module, args.output)
    print(f"pretrained on {metrics['train_samples']} rows, train_loss={metrics['train_loss']:.4f} -> {args.output}")


if __name__ == "__main__":
    _main()
