from contextlib import asynccontextmanager
from pathlib import Path

import boto3
from apscheduler.schedulers.background import BackgroundScheduler
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app import db, seed
from app.config import Settings
from app.export_embedding import export as export_embedding
from app.pipeline import run_retrain

scheduler = BackgroundScheduler()


def _s3_client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )


def _embedding_model_exists(settings: Settings) -> bool:
    s3 = _s3_client(settings)
    try:
        s3.head_object(Bucket=settings.s3_bucket, Key="models/embedding/model.onnx")
        return True
    except ClientError:
        return False


def bootstrap(settings: Settings | None = None) -> None:
    settings = settings or Settings()

    if not _embedding_model_exists(settings):
        export_embedding(Path("cache/embedding"), settings)

    conn = db.connect(settings)
    try:
        seed.bootstrap_into_db(conn)
        if db.active_version(conn) is None:
            run_retrain(settings=settings)
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(bootstrap)
    scheduler.add_job(run_retrain, "cron", hour=3, id="nightly_retrain", replace_existing=True)
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/retrain")
async def retrain() -> dict:
    return await run_in_threadpool(run_retrain)
