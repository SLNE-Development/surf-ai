import numpy as np
import torch

from app.head import HeadNet
from app.pretrain import load_checkpoint, pretrain_head, save_checkpoint


def _write_chunk(path, n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    labels = (rng.random((n, 6)) > 0.7).astype(np.float32)
    np.savez(path, embeddings=embeddings, labels=labels)


def test_pretrain_head_reduces_loss_over_epochs(tmp_path):
    chunk_path = tmp_path / "part-00000.npz"
    _write_chunk(chunk_path, n=64, dim=8)

    module, metrics_after_1 = pretrain_head([chunk_path], epochs=1, batch_size=16, lr=1e-2)
    module2, metrics_after_20 = pretrain_head([chunk_path], epochs=20, batch_size=16, lr=1e-2)

    assert metrics_after_20["train_loss"] < metrics_after_1["train_loss"]
    assert metrics_after_20["train_samples"] == 64


def test_pretrain_head_returns_module_matching_head_architecture(tmp_path):
    chunk_path = tmp_path / "part-00000.npz"
    _write_chunk(chunk_path, n=32, dim=8)

    module, _ = pretrain_head([chunk_path], epochs=1, batch_size=8)

    assert isinstance(module, HeadNet)
    module.eval()
    with torch.no_grad():
        out = module(torch.zeros(1, 8))
    assert out.shape == (1, 6)


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    chunk_path = tmp_path / "part-00000.npz"
    _write_chunk(chunk_path, n=32, dim=8)
    module, _ = pretrain_head([chunk_path], epochs=3, batch_size=8)

    dest = tmp_path / "base-checkpoint.pt"
    save_checkpoint(module, dest)
    loaded = load_checkpoint(dest, input_dim=8)

    module.eval()
    loaded.eval()
    with torch.no_grad():
        x = torch.randn(5, 8)
        assert torch.allclose(module(x), loaded(x))
