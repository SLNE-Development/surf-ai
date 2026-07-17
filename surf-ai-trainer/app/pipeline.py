import hashlib
import json
import tempfile
from pathlib import Path
from typing import Protocol

import boto3
import torch
from botocore.exceptions import ClientError

from app import db
from app.config import Settings
from app.evaluate import holdout_split, metrics as compute_metrics
from app.promote import maybe_promote
from app.train import export_head_onnx, train_head


class Embedder(Protocol):
    def embed(self, texts: list[str]): ...


def _s3_client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )


def _default_embedder(settings: Settings) -> Embedder:
    from app.embedding import Embedder as OnnxEmbedder

    cache_dir = Path("cache/embedding")
    onnx_path = cache_dir / "model.onnx"
    tokenizer_path = cache_dir / "tokenizer.json"
    if not onnx_path.exists() or not tokenizer_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        s3 = _s3_client(settings)
        s3.download_file(settings.s3_bucket, "models/embedding/model.onnx", str(onnx_path))
        s3.download_file(settings.s3_bucket, "models/embedding/tokenizer.json", str(tokenizer_path))

    return OnnxEmbedder(onnx_path, tokenizer_path, settings.embedding_prefix)


def _download_base_checkpoint(settings: Settings) -> Path | None:
    s3 = _s3_client(settings)
    dest = Path(tempfile.mkdtemp()) / "base-checkpoint.pt"
    try:
        s3.download_file(settings.s3_bucket, settings.base_head_checkpoint_s3_key, str(dest))
    except ClientError:
        return None
    return dest


def run_retrain(embedder: Embedder | None = None, settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    conn = db.connect(settings)
    try:
        samples = db.read_seed(conn) + db.read_labeled(conn, non_quarantined=True)
        embedder = embedder or _default_embedder(settings)

        train_samples, holdout = holdout_split(samples)
        checkpoint_path = _download_base_checkpoint(settings)
        init_state_dict = torch.load(checkpoint_path, weights_only=True) if checkpoint_path else None
        module, train_metrics = train_head(train_samples, embedder, init_state_dict=init_state_dict)
        eval_metrics = (
            compute_metrics(module, embedder, holdout) if holdout else {"macro_f1": 0.0}
        )

        active = db.active_version(conn)
        active_metrics = db.get_metrics(conn, active) if active is not None else None
        promoted = maybe_promote(eval_metrics, active_metrics)

        summary: dict = {
            "promoted": promoted,
            "metrics": eval_metrics,
            "sample_count": len(samples),
        }

        if promoted:
            version = db.next_version(conn)
            s3_key = f"models/head/head-v{version}.onnx"
            with tempfile.TemporaryDirectory() as tmp_dir:
                onnx_path = Path(tmp_dir) / f"head-v{version}.onnx"
                export_head_onnx(module, onnx_path)
                _s3_client(settings).upload_file(str(onnx_path), settings.s3_bucket, s3_key)

            training_data_hash = hashlib.sha256(
                json.dumps(sorted(s["text"] for s in samples)).encode()
            ).hexdigest()
            db.insert_version(conn, version, s3_key, settings.embedding_model_id, eval_metrics, training_data_hash)
            db.set_active(conn, version)
            summary["version"] = version

        return summary
    finally:
        conn.close()
