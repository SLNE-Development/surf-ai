# surf-ai

Self-trainable, multilingual (DE/EN) chat-moderation AI for Minecraft servers.

Every chat message can be scored across six toxicity categories in a few
milliseconds. A moderation plugin (out of scope for this repo) decides what
to *do* with those scores — surf-ai only supplies the model, the inference
API, and a feedback loop that keeps the model improving over time without
ever needing a manual retrain.

Full background reading:
- Design/spec: [docs/superpowers/specs/2026-07-17-surf-ai-moderation-design.md](docs/superpowers/specs/2026-07-17-surf-ai-moderation-design.md)
- Implementation plan: [docs/superpowers/plans/2026-07-17-surf-ai-moderation.md](docs/superpowers/plans/2026-07-17-surf-ai-moderation.md)
- Architecture rules & conventions: [CLAUDE.md](CLAUDE.md)

---

## Table of contents

- [How it fits together](#how-it-fits-together)
- [Categories](#categories)
- [Repo layout](#repo-layout)
- [Running it locally](#running-it-locally)
- [Using the API from a Minecraft plugin](#using-the-api-from-a-minecraft-plugin)
- [Ingame feedback dialog (Paper)](#ingame-feedback-dialog-paper)
- [The feedback → retrain → rollback loop](#the-feedback--retrain--rollback-loop)
- [Configuration reference](#configuration-reference)
- [Deploying (Docker / Coolify)](#deploying-docker--coolify)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## How it fits together

```
Minecraft plugin (Paper/Velocity)
   │  AIInstance.check(text) / .feedback(requestId, feedback)
   │  (RabbitMQ RPC, CBOR)
   ▼
surf-ai-microservice (Kotlin/JVM)
   │  micro-batches requests, embeds text (ONNX, frozen),
   │  runs classification head (ONNX, retrainable),
   │  caches request → scores in-memory (TTL, no DB write)
   │
   ├─ feedback() persists a labeled sample → Postgres (ai_labeled_sample)
   ├─ reads/writes model weights ────────────────────────► S3 / MinIO
   └─ HTTP calls ───────────────────────────────────────► surf-ai-trainer
                                                                │
                                                    surf-ai-trainer (Python/FastAPI)
                                                    trains the classification head
                                                    from seed corpus + feedback,
                                                    evaluates it, promotes/rejects it,
                                                    writes new snapshot to S3 + Postgres
```

- **Inference only ever runs in the microservice JVM.** Plugins are thin RPC
  clients — they never load a model.
- **Only the classification head is trainable.** Text embedding uses a frozen,
  multilingual sentence-embedding model (`intfloat/multilingual-e5-small`,
  384-dim) shared identically between Kotlin (inference) and Python
  (training), so the vector space always matches.
- **No raw chat ever hits the database.** `check()` results (text, embedding,
  scores) live only in an in-memory TTL cache (`RequestCache`). Only a
  `feedback()` call persists a labeled sample.
- **Retraining is safe by construction.** Every retrain evaluates the new
  head on a held-out split and only promotes it if macro-F1 doesn't regress
  by more than a small tolerance — a bad batch of feedback can't silently
  make the model worse. If it does get promoted and turns out bad anyway, an
  admin can roll back to any previous version.

## Categories

`check()` always returns a confidence score (0–100) for **all six**
categories — thresholding/acting on them is the moderation plugin's job, not
this one's. Category order is a wire contract (ONNX head output index) and
must never be reordered:

| Category | Meaning |
|---|---|
| `HARASSMENT` | Targeted insults/bullying against a person |
| `SELF_HARM` | Encouraging self-harm/suicide |
| `HATE_SPEECH` | Slurs/discrimination (origin, religion, gender, sexuality) |
| `SEXUAL` | Sexual harassment/content |
| `THREAT` | Real (non-ingame) threats of violence against others |
| `CHILD_SAFETY` | Grooming/predatory behavior toward minors |

## Repo layout

| Module | Role |
|---|---|
| `surf-ai-api` | Public DTOs (`AiCategory`, `AiCheckResult`, `AiFeedback`, `AiFeedbackResult`, `ModelVersionInfo`) and the `AIInstance` interface plugins call |
| `surf-ai-core/surf-ai-core-common` | `@RpcService` RPC contracts (`AiRpcService`, `AiAdminRpcService`) |
| `surf-ai-core/surf-ai-core-client` | Client-side core wiring shared by the Paper/Velocity plugins |
| `surf-ai-client/surf-ai-api-client-common` | Bridges `AIInstance` → the RPC proxy |
| `surf-ai-client/surf-ai-api-client-paper` | Paper plugin: ingame feedback dialog, `/aifeedback <requestId>` debug command |
| `surf-ai-client/surf-ai-api-client-velocity` | Velocity plugin wiring |
| `surf-ai-microservice` | The JVM inference service: embedding (HF tokenizer + ONNX), classification head, caches, DB repos, RPC impls, hot-reload watcher |
| `surf-ai-trainer` | Python FastAPI sidecar: trains the head from seed corpus + feedback, serves `/retrain` + `/health`, runs a nightly scheduled retrain |
| `docker/dev` | Local dev infra (Postgres, RabbitMQ, MinIO) |
| `docker-compose.yml` (repo root) | Coolify-consumable production compose |

## Running it locally

**Prerequisites:** Docker, JDK (via `./gradlew`), Python 3.11+.

```bash
# 1. Start local infra (Postgres, RabbitMQ, MinIO)
docker compose -f docker/dev/docker-compose.yml up -d

# 2. Start the trainer sidecar
cd surf-ai-trainer
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first startup the trainer:
1. exports/downloads the embedding ONNX model if it isn't already in MinIO,
2. bootstraps the seed corpus into Postgres,
3. if no active model version exists yet, runs an initial training pass and
   promotes it as v1.

It then schedules a nightly retrain (03:00) via APScheduler.

```bash
# 3. Start the microservice (separate terminal, from repo root)
./gradlew :surf-ai-microservice:run    # or however the standalone microservice is launched locally
```

The microservice connects to the same Postgres/RabbitMQ/MinIO and to the
trainer's HTTP API (`trainerBaseUrl`, see [Configuration](#configuration-reference)).
It polls for a new active model version every `hotReloadPollSeconds` and
hot-swaps its in-memory head without a restart.

Check the trainer is alive:

```bash
curl http://localhost:8000/health
```

## Using the API from a Minecraft plugin

Everything a plugin needs is `surf-ai-api`'s `AIInstance` singleton:

```kotlin
import dev.slne.surf.ai.api.AIInstance
import dev.slne.surf.ai.api.model.AiFeedback

// Score a chat message
val result = AIInstance.check(message)
// result.requestId: UUID          — keep this to submit feedback later
// result.scores: List<AiCategoryScore> — ALL 6 categories, confidence 0-100

for (score in result.scores) {
    if (score.category == AiCategory.THREAT && score.confidence > 80f) {
        // your moderation plugin decides what to do — surf-ai just scores
    }
}

// Later, once a moderator reviews the call:
AIInstance.feedback(result.requestId, AiFeedback.Correct)
AIInstance.feedback(result.requestId, AiFeedback.FalsePositive(setOf(AiCategory.SEXUAL)))
AIInstance.feedback(result.requestId, AiFeedback.FalsePositive(null)) // nothing should have flagged
AIInstance.feedback(result.requestId, AiFeedback.FalseNegative(setOf(AiCategory.HATE_SPEECH)))
```

`AiFeedback` is a sealed interface with three cases:

| Variant | Meaning |
|---|---|
| `Correct` | The scores were right, use as a reinforcing example |
| `FalsePositive(categories)` | These categories were wrongly flagged (or, if `categories == null`, nothing should have flagged at all) |
| `FalseNegative(categories)` | These categories *should* have been flagged (or scored higher) but weren't |

`feedback()` returns `AiFeedbackResult.ACCEPTED` (persisted, will be used in
the next retrain) or `EXPIRED` (the `requestId` fell out of the TTL cache —
the original text/embedding needed to build a labeled sample is gone, so
submit feedback promptly after a `check()` call).

Both `check()` and `feedback()` are `suspend` functions — call them from a
coroutine (e.g. via `plugin.launch { ... }` with MCCoroutine).

## Ingame feedback dialog (Paper)

The Paper client ships a ready-made feedback UI
(`dev.slne.surf.ai.client.paper.FeedbackDialog`): it lists the current scores
per category and lets a moderator flag which categories were false
positives/negatives, then submits the resulting `AiFeedback` and shows an
accepted/expired notice.

For manual testing there's a debug command:

```
/aifeedback <requestId>
```

This opens the dialog with an empty score list (the Paper client doesn't
hold the microservice's TTL cache) — the real trigger for opening it with
actual scores is expected to come from your moderation plugin, which is out
of scope for this repo.

## The feedback → retrain → rollback loop

1. **Immediate mitigation:** accepted feedback can also populate an
   in-JVM `OverrideCache` so a corrected score can take effect before the
   next retrain even runs.
2. **Persistence:** `feedback()` writes a labeled sample to
   `ai_labeled_sample` in Postgres. No feedback is ever silently dropped or
   auto-applied to the live model without going through evaluation.
3. **Retrain:** triggered nightly (03:00, cron) or on demand
   (`AiAdminRpcService.triggerRetrain()` → `POST /retrain` on the trainer).
   It trains the head on `seed corpus + non-quarantined labeled samples`,
   holds out a split for evaluation, and computes macro-F1 + other metrics.
4. **Promotion gate:** the new head is only promoted (made active, uploaded
   to S3, given a new version row) if its macro-F1 isn't worse than the
   currently active version's by more than a small tolerance
   (`maybe_promote` in `surf-ai-trainer/app/promote.py`). A regression is
   simply discarded — the active model keeps serving.
5. **Hot reload:** the microservice's `HotReloadWatcher` polls the active
   version every `hotReloadPollSeconds` and swaps in the new head without a
   restart.
6. **Rollback:** an admin can call `AiAdminRpcService.listVersions()` /
   `.rollbackTo(version)` at any time. Rolling back also quarantines any
   labeled samples collected after that version (`quarantineAbove`) so a bad
   batch of feedback that caused the regression doesn't get pulled into a
   future retrain again.

`AiAdminRpcService` (RabbitMQ RPC, not exposed to `AIInstance`/plugins —
intended for an admin tool/console) exposes:

```kotlin
suspend fun listVersions(): List<ModelVersionInfo>   // version, active, createdAt, metrics
suspend fun rollbackTo(version: Int)
suspend fun triggerRetrain()
```

## Configuration reference

### Microservice (`config.yml`, Sponge-style config in `AIInstance.dataPath`)

| Key | Default | Purpose |
|---|---|---|
| `s3Endpoint` | `http://localhost:9000` | MinIO/S3 endpoint for model weights |
| `s3Bucket` | `surf-ai` | Bucket holding `models/embedding/*` and `models/head/*` |
| `s3AccessKey` / `s3SecretKey` | `surfai` / `surfaikey` | S3 credentials |
| `trainerBaseUrl` | `http://localhost:8000` | Trainer sidecar's HTTP base URL |
| `ttlCacheMinutes` | `30` | How long a `check()` result stays available for `feedback()` before `EXPIRED` |
| `overrideCacheMaxSize` | `10000` | Max entries in the immediate-correction override cache |
| `batchWindowMillis` | `20` | Micro-batching window for inference requests |
| `batchMaxSize` | `32` | Max requests per inference batch |
| `hotReloadPollSeconds` | `30` | How often to check Postgres for a new active model version |
| `embeddingPrefix` | `"query: "` | e5 embedding prefix — must match the trainer's `embedding_prefix` |

### Trainer (env vars, prefix `SURF_AI_TRAINER_`)

| Env var | Default | Purpose |
|---|---|---|
| `SURF_AI_TRAINER_POSTGRES_DSN` | `postgresql://surf_ai:surf_ai@localhost:5432/surf_ai` | Postgres connection |
| `SURF_AI_TRAINER_S3_ENDPOINT` | `http://localhost:9000` | S3/MinIO endpoint |
| `SURF_AI_TRAINER_S3_BUCKET` | `surf-ai` | Bucket for model weights |
| `SURF_AI_TRAINER_S3_ACCESS_KEY` / `_S3_SECRET_KEY` | `surfai` / `surfaikey` | S3 credentials |
| `SURF_AI_TRAINER_EMBEDDING_MODEL_ID` | `intfloat/multilingual-e5-small` | HF model id to export/use for embeddings |
| `SURF_AI_TRAINER_EMBEDDING_PREFIX` | `"query: "` | Must match the microservice's `embeddingPrefix` |

### Root `.env` (docker-compose)

Copy `.env.example` to `.env` and fill in real secrets — it wires
`POSTGRES_*`, `RABBITMQ_*`, `S3_*`, and `TRAINER_BASE_URL` for the compose
stack (see below).

> Switching the embedding model requires a full head retrain (the vector
> space changes) — this is recorded in the model version's metadata.

## Deploying (Docker / Coolify)

The root `docker-compose.yml` is Coolify-consumable directly, or usable with
plain Docker:

```bash
cp .env.example .env   # then fill in real POSTGRES_PASSWORD / RABBITMQ_PASSWORD / S3_SECRET_KEY
docker compose --env-file .env up -d
```

It brings up `postgres`, `rabbitmq`, `minio` (+ a one-shot `minio-init` job
that creates the bucket), the `microservice`, and the `trainer`. If you
already have managed Postgres/RabbitMQ/MinIO elsewhere (e.g. Coolify's own
shared services), remove the corresponding services from the compose file
and point the env vars directly at them instead.

## Testing

```bash
./gradlew test                       # all Kotlin modules; needs docker/dev services up for DB/S3-backed tests
cd surf-ai-trainer && pytest         # fast tests only
pytest -m slow                       # also runs network-dependent tests (downloads the embedding model, ~470MB)
```

The Kotlin `EmbeddingModelFixtureTest` (verifies Kotlin's and Python's
embedding preprocessing produce identical vectors) needs a local model
fixture that isn't committed to the repo (~490MB). See
[`surf-ai-trainer/README.md`](surf-ai-trainer/README.md#embedding-model-fixture-for-kotlin-tests)
for the one-line export command; the test is skipped automatically when the
fixture is absent.

## Troubleshooting

- **`feedback()` always returns `EXPIRED`** — the `requestId` fell out of
  `RequestCache` before you submitted feedback. Either raise
  `ttlCacheMinutes` or submit feedback sooner after the original `check()`.
- **Microservice never picks up a new model version** — check
  `hotReloadPollSeconds` and that the trainer actually promoted a new
  version (`listVersions()` — a retrain that regresses macro-F1 is silently
  *not* promoted, this is expected, not a bug).
- **`ClassCastException` around RabbitMQ RPC** — `ServerRabbitMQApi` and
  `ClientRabbitMQApi` cannot coexist in the same JVM process; make sure
  you're not accidentally constructing both in one process.
- **Kotlin/Python embeddings disagree** — check `embeddingPrefix` /
  `EMBEDDING_PREFIX` match on both sides, and run
  `EmbeddingModelFixtureTest` locally after exporting the fixture.
