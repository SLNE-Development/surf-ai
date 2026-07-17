import csv
import json
from pathlib import Path
from typing import Callable, Iterator


def rows_from_csv(
    csv_path: Path,
    text_column: str,
    mapper: Callable[[dict], list[str]],
    language: str,
    source: str,
) -> Iterator[dict]:
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row[text_column].strip()
            if not text:
                continue
            yield {"text": text, "labels": mapper(row), "language": language, "source": source}


def write_corpus_chunks(rows: Iterator[dict], output_dir: Path, chunk_size: int = 200_000) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    chunk_index = 0
    buffer: list[dict] = []

    def flush():
        nonlocal chunk_index
        if not buffer:
            return
        dest = output_dir / f"part-{chunk_index:05d}.jsonl"
        with dest.open("w", encoding="utf-8") as f:
            for row in buffer:
                f.write(json.dumps(row) + "\n")
        chunk_index += 1
        buffer.clear()

    for row in rows:
        buffer.append(row)
        total += 1
        if len(buffer) >= chunk_size:
            flush()
    flush()
    return total
