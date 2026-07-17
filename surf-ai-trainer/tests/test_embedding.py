from pathlib import Path

import numpy as np
import pytest

from app.embedding import Embedder, mean_pool_normalize


def test_mean_pool_normalize_matches_kotlin_reference():
    hidden = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)  # 2 tokens, 2 dims
    mask = np.array([1, 1])
    v = mean_pool_normalize(hidden, mask)
    assert v[0] == pytest.approx(1.0, abs=1e-5)  # mean=(2,0) -> normalize -> (1,0)
    assert v[1] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.slow
def test_export_then_embed_is_deterministic_and_normalized(tmp_path: Path):
    from app.export_embedding import export

    export(tmp_path)
    onnx_path = next(tmp_path.glob("*.onnx"))
    tokenizer_path = tmp_path / "tokenizer.json"

    embedder = Embedder(onnx_path, tokenizer_path, prefix="query: ")
    vectors = embedder.embed(["hello", "hello"])

    assert vectors.shape == (2, 384)
    assert np.allclose(vectors[0], vectors[1])
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-3)
