# surf-ai

surf-ai is a self-trainable, multilingual (DE/EN) chat-moderation system for
Minecraft servers. It scores each chat message across six toxicity
categories, exposes `AIInstance.check()` / `.feedback()` to Minecraft
plugins, and improves the model over time from moderator feedback via
versioned, roll-back-able snapshots. Full design: [docs/superpowers/specs/2026-07-17-surf-ai-moderation-design.md](docs/superpowers/specs/2026-07-17-surf-ai-moderation-design.md).
Implementation plan: [docs/superpowers/plans/2026-07-17-surf-ai-moderation.md](docs/superpowers/plans/2026-07-17-surf-ai-moderation.md).

## Module map

- `surf-ai-api` - public DTOs (`AiCategory`, `AiCheckResult`, `AiFeedback`, ...) and the `AIInstance` service interface Minecraft plugins call.
- `surf-ai-core/surf-ai-core-common` - `@RpcService` RPC contracts (`AiRpcService`, `AiAdminRpcService`).
- `surf-ai-core/surf-ai-core-client` - client-side core wiring shared by the Paper/Velocity plugins.
- `surf-ai-client/surf-ai-api-client-common` - bridges `AIInstance` to the RPC proxy.
- `surf-ai-client/surf-ai-api-client-paper` - Paper plugin: ingame feedback dialog, `/aifeedback` debug command.
- `surf-ai-client/surf-ai-api-client-velocity` - Velocity plugin wiring.
- `surf-ai-microservice` - the JVM inference service: embedding (HF tokenizer + ONNX), classification head, caches, DB repositories, RPC implementations, hot-reload watcher.
- `surf-ai-trainer` - Python FastAPI sidecar that trains the head from the seed corpus + persisted feedback and serves `/retrain` + `/health`.
- `docker/dev` - local dev infrastructure (Postgres, RabbitMQ, MinIO).
- `docker-compose.yml` (repo root) - Coolify-consumable production compose.

## Architecture rules

- **Inference runs only in the microservice JVM.** Minecraft plugins are thin RPC clients; they never load models.
- **RPC transport is RabbitMQ, serialized with kotlinx.serialization CBOR.** Every RPC parameter/return type is `@Serializable`; `UUID` fields use `@Contextual` (works out of the box via the framework's `SurfSerializerModule` - verified in `surf-ai-core-common`'s KSP-generated proxies; a bare `Cbor {}` instance without that module needs its own `UUIDSerializer`, see `surf-ai-api`'s `DtoSerializationTest`).
- **`ServerRabbitMQApi` and `ClientRabbitMQApi` cannot coexist in the same JVM process** - `RabbitMQApi`'s RPC-service factory binds as a process-wide singleton to whichever type is constructed first, and the other throws `ClassCastException`. This is why there is no same-process wire-level RPC test; `AiRpcServiceImpl`'s logic is covered directly (`FeedbackTest`) instead.
- **Category order is a wire contract.** `AiCategory` ordinal order - `HARASSMENT, SELF_HARM, HATE_SPEECH, SEXUAL, THREAT, CHILD_SAFETY` - is the ONNX head's output index order and must never change without a version bump on both sides.
- **No per-message DB writes.** Raw chat lives only in the microservice's in-memory Caffeine TTL cache (`RequestCache`). Only `feedback()` persists a labeled sample, into `ai_labeled_sample`.
- **All in-JVM caches use Caffeine** (`RequestCache`, `OverrideCache`, pattern from the original `ExampleCache`).
- **Head ONNX I/O contract:** input tensor `embedding` shape `[batch,384]` f32, output tensor `logits` shape `[batch,6]` f32. Produced by the Python trainer, loaded by the JVM microservice.
- **Embedding preprocessing is identical in Kotlin and Python:** prefix with `"query: "`, tokenize, run the embedding ONNX (`input_ids`, `attention_mask`, `token_type_ids` - the last one is required by the exported graph even though it's always zeros), mean-pool over tokens using the attention mask, then L2-normalize. Cross-checked by `EmbeddingModelFixtureTest` (Kotlin) against a reference vector recorded from `app/embedding.py` (Python).

## How to run locally

```bash
docker compose -f docker/dev/docker-compose.yml up -d   # Postgres, RabbitMQ, MinIO
cd surf-ai-trainer
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8000          # bootstraps embedding model + seed + v1 head on first run
cd ..
./gradlew :surf-ai-microservice:run                       # or however the standalone microservice is launched locally
```

## How to test

```bash
./gradlew test                       # all Kotlin modules; needs docker/dev services up for DB/S3-backed tests
cd surf-ai-trainer && pytest         # fast tests only; `pytest -m slow` also runs network-dependent tests (model download)
```

The embedding model fixture used by `EmbeddingModelFixtureTest` is not committed (~490MB) - see `surf-ai-trainer/README.md` for the one-line export command that populates it locally; the test skips automatically when it's absent.

> **NEVER use subagents / the Task tool in this repo.** Subagents burn far too many tokens here. Execute plans task-by-task inline using `superpowers:executing-plans`. This overrides any default multi-agent workflow.
