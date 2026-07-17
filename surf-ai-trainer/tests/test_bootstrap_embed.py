import json

import numpy as np

from app.bootstrap_embed import embed_corpus_chunks
from app.categories import CATEGORIES


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return np.array([[float(len(t))] * 4 for t in texts], dtype=np.float32)


def _write_chunk(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_embed_corpus_chunks_writes_npz_per_input_chunk(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_chunk(corpus_dir / "part-00000.jsonl", [
        {"text": "hi", "labels": ["HARASSMENT"], "language": "en", "source": "test"},
        {"text": "hello there", "labels": [], "language": "en", "source": "test"},
    ])
    output_dir = tmp_path / "embeddings"

    written = embed_corpus_chunks(corpus_dir, FakeEmbedder(), output_dir, batch_size=1)

    assert written == 1
    data = np.load(output_dir / "part-00000.npz")
    assert data["embeddings"].shape == (2, 4)
    assert data["labels"].shape == (2, len(CATEGORIES))
    assert data["labels"][0][CATEGORIES.index("HARASSMENT")] == 1.0
    assert data["labels"][1].sum() == 0.0


def test_embed_corpus_chunks_batches_embed_calls(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_chunk(corpus_dir / "part-00000.jsonl", [
        {"text": "a", "labels": [], "language": "en", "source": "test"},
        {"text": "bb", "labels": [], "language": "en", "source": "test"},
        {"text": "ccc", "labels": [], "language": "en", "source": "test"},
    ])
    output_dir = tmp_path / "embeddings"
    embedder = FakeEmbedder()

    embed_corpus_chunks(corpus_dir, embedder, output_dir, batch_size=2)

    assert len(embedder.calls) == 2
    assert embedder.calls[0] == ["a", "bb"]
    assert embedder.calls[1] == ["ccc"]


def test_embed_corpus_chunks_skips_existing_output(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _write_chunk(corpus_dir / "part-00000.jsonl", [{"text": "hi", "labels": [], "language": "en", "source": "test"}])
    output_dir = tmp_path / "embeddings"
    output_dir.mkdir()
    np.savez(output_dir / "part-00000.npz", embeddings=np.zeros((1, 4), dtype=np.float32), labels=np.zeros((1, 6), dtype=np.float32))

    embedder = FakeEmbedder()
    written = embed_corpus_chunks(corpus_dir, embedder, output_dir, batch_size=1)

    assert written == 0
    assert embedder.calls == []
