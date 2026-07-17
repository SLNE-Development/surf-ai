import hashlib

import numpy as np
import onnxruntime as ort

from app.categories import CATEGORIES
from app.train import export_head_onnx, train_head


class DeterministicFakeEmbedder:
    """Deterministic per-text pseudo-embedding - no network/model needed for this test."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(384).astype(np.float32)
            vectors.append(vec / np.linalg.norm(vec))
        return np.stack(vectors)


SAMPLES = [
    {"text": "kys", "labels": ["SELF_HARM"]},
    {"text": "geh dich doch einfach umbringen", "labels": ["SELF_HARM"]},
    {"text": "du hurensohn", "labels": ["HARASSMENT"]},
    {"text": "du bist so dumm", "labels": ["HARASSMENT"]},
    {"text": "penis", "labels": ["SEXUAL"]},
    {"text": "zeig mir deine titten", "labels": ["SEXUAL"]},
    {"text": "ich schlag dich tot", "labels": ["THREAT"]},
    {"text": "ich finde dich und mach dich fertig", "labels": ["THREAT"]},
    {"text": "kill the ender dragon", "labels": []},
    {"text": "let's go raid the nether fortress", "labels": []},
]


def test_train_and_export_onnx_roundtrip(tmp_path):
    module, metrics = train_head(SAMPLES, DeterministicFakeEmbedder(), epochs=200)
    assert "train_loss" in metrics

    dest = tmp_path / "head-test.onnx"
    export_head_onnx(module, dest)
    assert dest.exists()

    session = ort.InferenceSession(str(dest))
    embedder = DeterministicFakeEmbedder()
    kys_embedding = embedder.embed(["kys"])
    (logits,) = session.run(None, {"embedding": kys_embedding})

    assert logits.shape == (1, 6)
    predicted_category = CATEGORIES[int(np.argmax(logits[0]))]
    assert predicted_category == "SELF_HARM"
