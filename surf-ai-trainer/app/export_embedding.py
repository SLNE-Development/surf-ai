from pathlib import Path

import boto3
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from app.config import Settings


def export(dest_dir: Path, settings: Settings | None = None) -> None:
    settings = settings or Settings()
    dest_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(settings.embedding_model_id, export=True)
    model.save_pretrained(dest_dir)

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_id)
    tokenizer.save_pretrained(dest_dir)

    onnx_path = next(dest_dir.glob("*.onnx"))
    tokenizer_path = dest_dir / "tokenizer.json"

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )
    s3.upload_file(str(onnx_path), settings.s3_bucket, "models/embedding/model.onnx")
    s3.upload_file(str(tokenizer_path), settings.s3_bucket, "models/embedding/tokenizer.json")
