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
pytest              # fast tests only (default; excludes network-dependent tests)
pytest -m slow       # network tests too (downloads multilingual-e5-small, ~470MB)
```

## Embedding model fixture (for Kotlin tests)

The embedding ONNX model (~470MB) and tokenizer (~17MB) are **not** committed
to the repo - they're fetched/exported locally instead. To populate the
fixture the Kotlin `EmbeddingModelFixtureTest` cross-consistency test looks
for at `surf-ai-microservice/src/test/resources/models/embedding/` (skipped
automatically when absent):

```bash
cd surf-ai-trainer
python -c "
from pathlib import Path
from app.export_embedding import export
export(Path('../surf-ai-microservice/src/test/resources/models/embedding'))
"
```

This also uploads the model to the dev MinIO bucket under
`models/embedding/` (requires `docker/dev/docker-compose.yml` to be up), so
the microservice's `ModelHolder.reloadTo` can download it too.
