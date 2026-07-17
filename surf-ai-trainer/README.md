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

## Bootstrap corpus pretraining (one-time, manual)

The classification head normally trains from just the seed corpus + moderator feedback
(a few hundred rows). To give it a broader starting point, you can warm-start it from a
larger public toxicity/hate-speech corpus. This is a manual, one-time process — none of
it runs in CI or the default `pytest` suite.

**1. Download source CSVs** (not committed to this repo — respect each dataset's
license/ToS):
- Jigsaw Toxic Comment Classification Challenge (`train.csv`, Kaggle)
- Jigsaw Unintended Bias in Toxicity Classification (`train.csv`, Kaggle)
- A self-harm/suicidal-ideation corpus, e.g. "Suicide and Depression Detection"
  (`text`, `class` columns, Kaggle)
- GermEval 2018/2019 offensive-language German dataset (`text`, `task1_label`,
  `task2_label` columns)

Verify each file's actual column names against the schemas expected in
`scripts/build_bootstrap_corpus.py` before running it — dataset mirrors sometimes rename
columns.

**2. Build the mapped corpus:**

```bash
python scripts/build_bootstrap_corpus.py \
  --jigsaw-toxic-csv path/to/jigsaw_toxic_train.csv \
  --jigsaw-unintended-bias-csv path/to/jigsaw_unintended_bias_train.csv \
  --suicidewatch-csv path/to/suicidewatch.csv \
  --germeval-csv path/to/germeval.csv
```

Writes chunked JSONL to `cache/bootstrap_corpus/`.

**3. Embed the corpus** (uses the same embedding model as the rest of the trainer;
requires the embedding fixture already exported per the section above; uses GPU via
`onnxruntime-gpu`'s `CUDAExecutionProvider` if installed and available, otherwise falls
back to CPU):

```bash
python scripts/embed_bootstrap_corpus.py
```

Writes cached `.npz` embedding chunks to `cache/bootstrap_embeddings/`. Safe to
interrupt and re-run — already-embedded chunks are skipped.

**4. Pretrain the base checkpoint:**

```bash
python -m app.pretrain
```

Writes `cache/base-checkpoint.pt`.

**5. Upload it to the S3/MinIO bucket** so `run_retrain` picks it up automatically on the
next retrain:

```bash
python -c "
from app.config import Settings
from app.pipeline import _s3_client
settings = Settings()
_s3_client(settings).upload_file('cache/base-checkpoint.pt', settings.s3_bucket, settings.base_head_checkpoint_s3_key)
"
```

From this point on, every `run_retrain` call (including the nightly scheduled one) warm-starts the head from this checkpoint instead of a random init. Note: this corpus has no CHILD_SAFETY coverage (no responsible public dataset exists for it) and is domain-mismatched (news comments / Reddit vs. Minecraft chat) — it's a broader starting point for the head, not a substitute for the in-domain seed + feedback data that still does the real fine-tuning.
