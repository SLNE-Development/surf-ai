import json
from pathlib import Path

import numpy as np

from app.categories import labels_to_vector


def embed_corpus_chunks(corpus_dir: Path, embedder, output_dir: Path, batch_size: int = 512) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for chunk_path in sorted(corpus_dir.glob("part-*.jsonl")):
        dest = output_dir / f"{chunk_path.stem}.npz"
        if dest.exists():
            continue

        rows = []
        with chunk_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        embeddings_batches = []
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            embeddings_batches.append(np.asarray(embedder.embed([r["text"] for r in batch]), dtype=np.float32))

        embeddings = np.concatenate(embeddings_batches, axis=0) if embeddings_batches else np.zeros((0, 0), dtype=np.float32)
        labels = np.array([labels_to_vector(r["labels"]) for r in rows], dtype=np.float32)

        np.savez(dest, embeddings=embeddings, labels=labels)
        written += 1

    return written
