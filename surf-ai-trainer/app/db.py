import json

import psycopg

from app.config import Settings


def connect(settings: Settings | None = None) -> psycopg.Connection:
    settings = settings or Settings()
    return psycopg.connect(settings.postgres_dsn)


def read_seed(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT text, labels, language, source FROM ai_seed_sample")
        return [
            {"text": text, "labels": json.loads(labels), "language": language, "source": source}
            for text, labels, language, source in cur.fetchall()
        ]


def read_labeled(conn: psycopg.Connection, non_quarantined: bool = True) -> list[dict]:
    query = "SELECT text, labels, feedback_type, source, weight, origin_model_version FROM ai_labeled_sample"
    if non_quarantined:
        query += " WHERE quarantined = false"
    with conn.cursor() as cur:
        cur.execute(query)
        return [
            {
                "text": text,
                "labels": json.loads(labels),
                "feedback_type": feedback_type,
                "source": source,
                "weight": weight,
                "origin_model_version": origin_model_version,
            }
            for text, labels, feedback_type, source, weight, origin_model_version in cur.fetchall()
        ]


def insert_version(
    conn: psycopg.Connection,
    version: int,
    s3_key: str,
    embedding_model_id: str,
    metrics: dict[str, float],
    training_data_hash: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ai_model_version (version, s3_key, embedding_model_id, metrics, training_data_hash)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (version, s3_key, embedding_model_id, json.dumps(metrics), training_data_hash),
        )
    conn.commit()


def set_active(conn: psycopg.Connection, version: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE ai_model_version SET active = false")
        cur.execute("UPDATE ai_model_version SET active = true WHERE version = %s", (version,))
    conn.commit()


def active_version(conn: psycopg.Connection) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM ai_model_version WHERE active = true")
        row = cur.fetchone()
        return row[0] if row else None


def next_version(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM ai_model_version")
        (version,) = cur.fetchone()
        return version
