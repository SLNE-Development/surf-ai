import hashlib

import numpy as np
import pytest

from app import db, seed
from app.config import Settings
from app.pipeline import run_retrain
from app.train import train_head as pipeline_train_head

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


def test_run_retrain_warm_starts_from_base_checkpoint_when_present(clean_db, tmp_path, monkeypatch):
    import torch

    from app.head import HeadNet

    base_module = HeadNet(input_dim=384)
    checkpoint_path = tmp_path / "base-checkpoint.pt"
    torch.save(base_module.state_dict(), checkpoint_path)

    captured = {}
    original_train_head = pipeline_train_head

    def spy_train_head(samples, embedder, epochs=100, lr=1e-2, init_state_dict=None):
        captured["init_state_dict"] = init_state_dict
        return original_train_head(samples, embedder, epochs, lr, init_state_dict)

    monkeypatch.setattr("app.pipeline.train_head", spy_train_head)
    monkeypatch.setattr(
        "app.pipeline._download_base_checkpoint",
        lambda settings: checkpoint_path,
    )

    run_retrain(embedder=DeterministicFakeEmbedder(), settings=clean_db)

    assert captured["init_state_dict"] is not None
    assert set(captured["init_state_dict"].keys()) == set(base_module.state_dict().keys())


def test_run_retrain_falls_back_to_random_init_when_no_checkpoint(clean_db, monkeypatch):
    captured = {}
    original_train_head = pipeline_train_head

    def spy_train_head(samples, embedder, epochs=100, lr=1e-2, init_state_dict=None):
        captured["init_state_dict"] = init_state_dict
        return original_train_head(samples, embedder, epochs, lr, init_state_dict)

    monkeypatch.setattr("app.pipeline.train_head", spy_train_head)
    monkeypatch.setattr("app.pipeline._download_base_checkpoint", lambda settings: None)

    run_retrain(embedder=DeterministicFakeEmbedder(), settings=clean_db)

    assert captured["init_state_dict"] is None
