import csv
import json

from app.bootstrap_corpus import rows_from_csv, write_corpus_chunks
from app.bootstrap_mapping import map_jigsaw_toxic_row


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_rows_from_csv_applies_mapper_and_shape(tmp_path):
    csv_path = tmp_path / "jigsaw.csv"
    _write_csv(
        csv_path,
        [
            {"comment_text": "i will kill you", "toxic": "1", "severe_toxic": "0", "obscene": "0", "threat": "1", "insult": "0", "identity_hate": "0"},
            {"comment_text": "", "toxic": "0", "severe_toxic": "0", "obscene": "0", "threat": "0", "insult": "0", "identity_hate": "0"},
            {"comment_text": "nice build!", "toxic": "0", "severe_toxic": "0", "obscene": "0", "threat": "0", "insult": "0", "identity_hate": "0"},
        ],
        fieldnames=["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
    )

    rows = list(rows_from_csv(csv_path, text_column="comment_text", mapper=map_jigsaw_toxic_row, language="en", source="jigsaw_toxic"))

    assert rows == [
        {"text": "i will kill you", "labels": ["THREAT"], "language": "en", "source": "jigsaw_toxic"},
        {"text": "nice build!", "labels": [], "language": "en", "source": "jigsaw_toxic"},
    ]


def test_write_corpus_chunks_splits_and_counts(tmp_path):
    rows = ({"text": f"row {i}", "labels": [], "language": "en", "source": "test"} for i in range(5))

    total = write_corpus_chunks(rows, tmp_path, chunk_size=2)

    assert total == 5
    parts = sorted(tmp_path.glob("part-*.jsonl"))
    assert len(parts) == 3
    assert [json.loads(line) for line in parts[0].open(encoding="utf-8")] == [
        {"text": "row 0", "labels": [], "language": "en", "source": "test"},
        {"text": "row 1", "labels": [], "language": "en", "source": "test"},
    ]
    assert [json.loads(line) for line in parts[2].open(encoding="utf-8")] == [
        {"text": "row 4", "labels": [], "language": "en", "source": "test"},
    ]
