# surf-ai-trainer

Python sidecar that trains the moderation MLP head from the seed corpus and
persisted feedback, and serves a small FastAPI control surface consumed by
the Kotlin microservice.

## Setup

```bash
cd surf-ai-trainer
pip install -e ".[dev]"
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Requires the dev docker-compose stack from `docker/dev/docker-compose.yml`
(Postgres + MinIO) to be up.

## Test

```bash
pytest
```
