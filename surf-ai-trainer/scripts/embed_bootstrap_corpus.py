"""One-time, manual script: embeds the chunked bootstrap corpus (built by
build_bootstrap_corpus.py) using the same embedding model/preprocessing as the rest of
the trainer, with GPU acceleration if available. NOT run by pytest or CI.

Requires the embedding model + tokenizer already exported locally (see
surf-ai-trainer/README.md, "Embedding model fixture" section) at
cache/embedding/model.onnx and cache/embedding/tokenizer.json.
"""
import argparse
from pathlib import Path

import onnxruntime as ort

from app.bootstrap_embed import embed_corpus_chunks
from app.config import Settings
from app.embedding import Embedder


def _providers() -> list[str]:
    available = ort.get_available_providers()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=Path("cache/bootstrap_corpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache/bootstrap_embeddings"))
    parser.add_argument("--onnx-path", type=Path, default=Path("cache/embedding/model.onnx"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("cache/embedding/tokenizer.json"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    settings = Settings()
    embedder = Embedder(args.onnx_path, args.tokenizer_path, settings.embedding_prefix)
    embedder.session.set_providers(_providers())

    written = embed_corpus_chunks(args.corpus_dir, embedder, args.output_dir, args.batch_size)
    print(f"embedded {written} new chunk(s) into {args.output_dir} (providers: {_providers()})")


if __name__ == "__main__":
    main()
