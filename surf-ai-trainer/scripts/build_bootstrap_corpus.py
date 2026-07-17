"""One-time, manual script: builds the bootstrap training corpus from local CSV exports
of public toxicity/hate-speech/self-harm datasets. NOT run by pytest or CI.

Expected inputs (download manually, see surf-ai-trainer/README.md for source links and
exact column schemas to verify against what you downloaded):
  --jigsaw-toxic-csv           Jigsaw Toxic Comment Classification Challenge train.csv
  --jigsaw-unintended-bias-csv Jigsaw Unintended Bias in Toxicity Classification train.csv
  --suicidewatch-csv           Reddit SuicideWatch / "Suicide and Depression Detection" csv
  --germeval-csv                GermEval 2018/2019 offensive-language csv

Output: chunked JSONL under --output-dir (default: cache/bootstrap_corpus/), each row
shaped {"text": str, "labels": list[str], "language": str, "source": str}.
"""
import argparse
import itertools
from pathlib import Path

from app.bootstrap_corpus import rows_from_csv, write_corpus_chunks
from app.bootstrap_mapping import (
    map_germeval_row,
    map_jigsaw_toxic_row,
    map_jigsaw_unintended_bias_row,
    map_suicidewatch_row,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jigsaw-toxic-csv", type=Path)
    parser.add_argument("--jigsaw-unintended-bias-csv", type=Path)
    parser.add_argument("--suicidewatch-csv", type=Path)
    parser.add_argument("--germeval-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/bootstrap_corpus"))
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    sources = []
    if args.jigsaw_toxic_csv:
        sources.append(rows_from_csv(args.jigsaw_toxic_csv, "comment_text", map_jigsaw_toxic_row, "en", "jigsaw_toxic"))
    if args.jigsaw_unintended_bias_csv:
        sources.append(rows_from_csv(args.jigsaw_unintended_bias_csv, "comment_text", map_jigsaw_unintended_bias_row, "en", "jigsaw_unintended_bias"))
    if args.suicidewatch_csv:
        sources.append(rows_from_csv(args.suicidewatch_csv, "text", map_suicidewatch_row, "en", "suicidewatch"))
    if args.germeval_csv:
        sources.append(rows_from_csv(args.germeval_csv, "text", map_germeval_row, "de", "germeval"))

    if not sources:
        parser.error("at least one --*-csv source is required")

    total = write_corpus_chunks(itertools.chain(*sources), args.output_dir, args.chunk_size)
    print(f"wrote {total} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
