import hashlib

import numpy as np
import pytest

from app import db, seed
from app.config import Settings
from app.pipeline import run_retrain

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS ai_seed_sample (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    text TEXT NOT NULL,
    labels TEXT NOT NULL,
    language VARCHAR(8) NOT NULL,
    source VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_labeled_sample (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    text TEXT NOT NULL,
    labels TEXT NOT NULL,
    feedback_type VARCHAR(32) NOT NULL,
    source VARCHAR(64) NOT NULL,
    quarantined BOOLEAN NOT NULL DEFAULT false,
    weight REAL NOT NULL DEFAULT 1.0,
    origin_model_version INTEGER
);
CREATE TABLE IF NOT EXISTS ai_model_version (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    version INTEGER NOT NULL UNIQUE,
    s3_key VARCHAR(256) NOT NULL,
    embedding_model_id VARCHAR(128) NOT NULL,
    metrics TEXT NOT NULL,
    training_data_hash VARCHAR(128) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT false
);
"""


class DeterministicFakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()[:4]
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            vec = rng.standard_normal(384).astype(np.float32)
            vectors.append(vec / np.linalg.norm(vec))
        return np.stack(vectors)


@pytest.fixture
def clean_db():
    settings = Settings()
    conn = db.connect(settings)
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLES)
        cur.execute("TRUNCATE ai_seed_sample, ai_labeled_sample, ai_model_version")
    conn.commit()
    seed.bootstrap_into_db(conn)
    conn.close()
    yield settings


def test_run_retrain_bootstraps_version_1_from_seed_only(clean_db):
    summary = run_retrain(embedder=DeterministicFakeEmbedder(), settings=clean_db)

    assert summary["promoted"] is True
    assert summary["version"] == 1

    conn = db.connect(clean_db)
    try:
        assert db.active_version(conn) == 1
    finally:
        conn.close()
