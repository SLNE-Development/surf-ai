import json
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"
SEED_FILES = ["seed_de.jsonl", "seed_en.jsonl", "seed_minecraft.jsonl"]


def load_seed() -> list[dict]:
    rows = []
    for filename in SEED_FILES:
        path = SEED_DIR / filename
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def bootstrap_into_db(conn) -> int:
    """Inserts the seed corpus into ai_seed_sample if the table is currently empty.

    Returns the number of rows inserted (0 if the table already had data).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ai_seed_sample")
        (count,) = cur.fetchone()
        if count > 0:
            return 0

        rows = load_seed()
        for row in rows:
            cur.execute(
                "INSERT INTO ai_seed_sample (text, labels, language, source) VALUES (%s, %s, %s, %s)",
                (row["text"], json.dumps(row["labels"]), row["language"], row["source"]),
            )
        conn.commit()
        return len(rows)
