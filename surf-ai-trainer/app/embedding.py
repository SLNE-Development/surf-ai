from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def mean_pool_normalize(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """hidden: (seq_len, dims) float32, mask: (seq_len,) - mirrors the Kotlin EmbeddingModel."""
    mask = mask.astype(np.float32)
    weighted = hidden * mask[:, None]
    count = mask.sum()
    pooled = weighted.sum(axis=0) / count if count > 0 else weighted.sum(axis=0)
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0 else pooled


class Embedder:
    def __init__(self, onnx_path: Path, tokenizer_path: Path, prefix: str):
        self.prefix = prefix
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(onnx_path))

    def embed(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch([self.prefix + t for t in texts])
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        token_type_ids = np.zeros_like(ids)
        outputs = self.session.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": token_type_ids},
        )
        hidden_state = outputs[0]  # (batch, seq_len, dims)

        return np.stack([
            mean_pool_normalize(hidden_state[i], mask[i]) for i in range(len(texts))
        ]).astype(np.float32)
