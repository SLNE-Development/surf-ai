# surf-ai Chat-Moderation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (the user has forbidden subagents — see CLAUDE.md). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-trainable, multilingual (DE/EN) chat-moderation model that scores each Minecraft chat message across 6 toxicity categories, exposes `AIInstance.check()` / `.feedback()`, and improves via feedback + versioned, roll-back-able model snapshots.

**Architecture:** Inference runs in the Kotlin `surf-ai-microservice` JVM (HuggingFace tokenizer + frozen `multilingual-e5-small` embedding ONNX + a trainable MLP head ONNX). Minecraft plugins are thin RabbitMQ-RPC clients. A Python `surf-ai-trainer` sidecar (FastAPI + APScheduler) trains only the head from a seed corpus + persisted feedback, exports versioned head ONNX snapshots to S3/MinIO, and the microservice hot-reloads the active snapshot. No raw chat is persisted — only feedback-labeled samples.

**Tech Stack:** Kotlin (surf-api framework, Exposed/R2DBC Postgres, RabbitMQ RPC via kotlinx-serialization CBOR, Caffeine, DJL HF tokenizers + onnxruntime-java, MinIO java client), Python 3.11 (PyTorch CPU, transformers, optimum[onnxruntime], onnxruntime, scikit-learn, FastAPI, APScheduler, boto3, psycopg), Docker/Coolify.

## Global Constraints

- **Category order is a wire contract** — this exact order everywhere (enum ordinal = ONNX output index): `HARASSMENT, SELF_HARM, HATE_SPEECH, SEXUAL, THREAT, CHILD_SAFETY`.
- **Confidence** is `Float` in range `0.0..100.0` (sigmoid(logit) * 100).
- **RPC serialization** = kotlinx.serialization **CBOR** (KSP-generated proxies). Every RPC parameter/return DTO MUST be `@Serializable`; `UUID` fields use `@Contextual` (provided by `SurfSerializerModule`; if a smoke test shows it is missing, register `kotlinx.serialization` `UUIDSerializer` in the `SerializersModule` passed to `ClientRabbitMQApi.create` / `ServerRabbitMQApi.create`).
- **DTOs** live in `surf-ai-api`. **RPC contracts** (`@RpcService`) live in `surf-ai-core:surf-ai-core-common`.
- **Embedding model** = `intfloat/multilingual-e5-small`. Preprocessing (both JVM & Python, identical): prefix input with `"query: "`, tokenize, run ONNX, **mean-pool over tokens using the attention mask, then L2-normalize** → 384-dim float32 vector.
- **Head ONNX I/O contract:** input tensor name `embedding` shape `[batch,384]` f32 → output tensor name `logits` shape `[batch,6]` f32. Same file loaded by JVM (inference) and produced by Python (training).
- **All in-JVM caches** use **Caffeine** (pattern: existing `ExampleCache`).
- **No per-message DB writes.** Raw chat lives only in an in-memory Caffeine TTL cache. Only `feedback()` persists a labeled sample.
- **DB** = Postgres via `dev.slne.surf.database`; tables extend `AuditableLongIdTable`; repos use `suspendTransaction`.
- **S3/MinIO layout** (bucket from config, default `surf-ai`): `models/embedding/model.onnx`, `models/embedding/tokenizer.json`, `models/head/head-v{n}.onnx`.
- **Package root:** `dev.slne.surf.ai`.
- **Commit after every task.** Commit messages end with the standard `Co-Authored-By` trailer used by this repo.
- **No subagents** during execution (see Task 26 CLAUDE.md).

---

## Task 0: Dev infrastructure (docker-compose: Postgres + RabbitMQ + MinIO)

**Files:**
- Create: `docker/dev/docker-compose.yml`
- Create: `docker/dev/README.md`

**Interfaces:**
- Produces: local services on `localhost:5432` (Postgres `surf_ai`/`surf_ai`/`surf_ai`), `localhost:5672`+`15672` (RabbitMQ `surf`/`surf`), `localhost:9000`+`9001` (MinIO `surfai`/`surfaikey`). Bucket `surf-ai` auto-created by a `mc` init container.

- [ ] **Step 1: Write `docker/dev/docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: surf_ai
      POSTGRES_USER: surf_ai
      POSTGRES_PASSWORD: surf_ai
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U surf_ai"]
      interval: 3s
      timeout: 3s
      retries: 20
  rabbitmq:
    image: rabbitmq:3.13-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: surf
      RABBITMQ_DEFAULT_PASS: surf
    ports: ["5672:5672", "15672:15672"]
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
      interval: 5s
      timeout: 5s
      retries: 20
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: surfai
      MINIO_ROOT_PASSWORD: surfaikey
    ports: ["9000:9000", "9001:9001"]
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 20
  minio-init:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 surfai surfaikey &&
      mc mb --ignore-existing local/surf-ai &&
      echo 'bucket ready'"
```

- [ ] **Step 2: Write `docker/dev/README.md`** with the one-liner: `docker compose -f docker/dev/docker-compose.yml up -d` and the credentials/ports table above.

- [ ] **Step 3: Bring it up and verify**

Run: `docker compose -f docker/dev/docker-compose.yml up -d`
Then: `docker compose -f docker/dev/docker-compose.yml ps`
Expected: `postgres`, `rabbitmq`, `minio` show `healthy`; `minio-init` exited 0.

- [ ] **Step 4: Commit**

```bash
git add docker/dev
git commit -m "chore: add dev docker-compose (postgres, rabbitmq, minio)"
```

---

## Task 1: RPC serialization smoke test (verify CBOR + UUID)

Confirms the wire contract before we build DTOs on top of it.

**Files:**
- Modify: `surf-ai-core/surf-ai-core-common/.../ExampleRpcService.kt` (temporarily add a UUID method)
- Modify: `surf-ai-microservice/.../rpc/ExampleRpcServiceImpl.kt`
- Create: `surf-ai-microservice/src/test/kotlin/dev/slne/surf/ai/microservice/RpcSmokeTest.kt`

**Interfaces:**
- Produces: confirmed answer to "is `@Contextual UUID` serializable out of the box?" recorded in the commit message.

- [ ] **Step 1: Add a UUID round-trip method to `ExampleRpcService`**

```kotlin
@RpcService
interface ExampleRpcService {
    suspend fun exampleMethod(param: String): String
    suspend fun echoUuid(id: @Contextual java.util.UUID): java.util.UUID
}
```
Implement in `ExampleRpcServiceImpl`: `override suspend fun echoUuid(id: UUID) = id`.

- [ ] **Step 2: Build to trigger KSP**

Run: `./gradlew :surf-ai-microservice:compileKotlin :surf-ai-core:surf-ai-core-common:compileKotlin`
Expected: BUILD SUCCESSFUL. If KSP errors that no serializer is found for `UUID`, note it — the Global Constraints fallback (register `UUIDSerializer`) then becomes mandatory for all DTO modules.

- [ ] **Step 3: Record the finding & revert the example changes**

Revert both example files to their original 1-method form (keep the codebase clean). Commit the finding.

```bash
git add -A
git commit -m "test: verify RPC CBOR/UUID serialization (contextual UUID works out of the box)"
```
(If UUID needed manual registration, say so in the message instead and keep a note in `surf-ai-api` package docs.)

---

## Task 2: Category enum + score DTOs (`surf-ai-api`)

**Files:**
- Create: `surf-ai-api/src/main/kotlin/dev/slne/surf/ai/api/model/AiCategory.kt`
- Create: `surf-ai-api/src/main/kotlin/dev/slne/surf/ai/api/model/AiCategoryScore.kt`
- Create: `surf-ai-api/src/main/kotlin/dev/slne/surf/ai/api/model/AiCheckResult.kt`
- Test: `surf-ai-api/src/test/kotlin/dev/slne/surf/ai/api/model/DtoSerializationTest.kt`
- Modify: `surf-ai-api/build.gradle.kts` (ensure kotlinx-serialization plugin/runtime available — usually provided by `dev.slne.surf.api.gradle.core`; add `withRabbitModule` only where needed, not here)

**Interfaces:**
- Produces: `enum class AiCategory { HARASSMENT, SELF_HARM, HATE_SPEECH, SEXUAL, THREAT, CHILD_SAFETY }`; `@Serializable data class AiCategoryScore(val category: AiCategory, val confidence: Float)`; `@Serializable data class AiCheckResult(val requestId: @Contextual UUID, val scores: List<AiCategoryScore>)`.

- [ ] **Step 1: Write the failing test** `DtoSerializationTest.kt`

```kotlin
package dev.slne.surf.ai.api.model

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.cbor.Cbor
import kotlinx.serialization.decodeFromByteArray
import kotlinx.serialization.encodeToByteArray
import java.util.UUID
import kotlin.test.Test
import kotlin.test.assertEquals

@OptIn(ExperimentalSerializationApi::class)
class DtoSerializationTest {
    private val cbor = Cbor { ignoreUnknownKeys = true }

    @Test fun `category order is the wire contract`() {
        assertEquals(0, AiCategory.HARASSMENT.ordinal)
        assertEquals(5, AiCategory.CHILD_SAFETY.ordinal)
        assertEquals(6, AiCategory.entries.size)
    }

    @Test fun `AiCheckResult round-trips through cbor`() {
        val id = UUID.randomUUID()
        val r = AiCheckResult(id, listOf(AiCategoryScore(AiCategory.HARASSMENT, 57.7f)))
        val bytes = cbor.encodeToByteArray(r)
        assertEquals(r, cbor.decodeFromByteArray<AiCheckResult>(bytes))
    }
}
```

- [ ] **Step 2: Run it, verify it fails** — Run: `./gradlew :surf-ai-api:test --tests "*DtoSerializationTest*"` — Expected: FAIL (types unresolved).

- [ ] **Step 3: Implement the three files**

`AiCategory.kt`:
```kotlin
package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
enum class AiCategory { HARASSMENT, SELF_HARM, HATE_SPEECH, SEXUAL, THREAT, CHILD_SAFETY }
```
`AiCategoryScore.kt`:
```kotlin
package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
data class AiCategoryScore(val category: AiCategory, val confidence: Float)
```
`AiCheckResult.kt`:
```kotlin
package dev.slne.surf.ai.api.model

import kotlinx.serialization.Contextual
import kotlinx.serialization.Serializable
import java.util.UUID

@Serializable
data class AiCheckResult(
    @Contextual val requestId: UUID,
    val scores: List<AiCategoryScore>,
)
```
If Task 1 found CBOR needs the serialization gradle plugin here, add `kotlin("plugin.serialization")` and `kotlinx-serialization-cbor` test dependency to `surf-ai-api/build.gradle.kts`.

- [ ] **Step 4: Run tests, verify pass** — Run: `./gradlew :surf-ai-api:test --tests "*DtoSerializationTest*"` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add surf-ai-api
git commit -m "feat(api): add AiCategory, AiCategoryScore, AiCheckResult DTOs"
```

---

## Task 3: Feedback DTOs (`surf-ai-api`)

**Files:**
- Create: `surf-ai-api/src/main/kotlin/dev/slne/surf/ai/api/model/AiFeedback.kt`
- Create: `surf-ai-api/src/main/kotlin/dev/slne/surf/ai/api/model/AiFeedbackResult.kt`
- Test: extend `DtoSerializationTest.kt`

**Interfaces:**
- Produces: `@Serializable sealed interface AiFeedback` with `FalsePositive(categories: Set<AiCategory>? = null)`, `FalseNegative(categories: Set<AiCategory>)`, `data object Correct`; `@Serializable enum class AiFeedbackResult { ACCEPTED, EXPIRED }`.

- [ ] **Step 1: Add failing tests** for polymorphic round-trip:

```kotlin
@Test fun `feedback polymorphism round-trips`() {
    val fp: AiFeedback = AiFeedback.FalseNegative(setOf(AiCategory.SEXUAL))
    val bytes = cbor.encodeToByteArray(AiFeedback.serializer(), fp)
    assertEquals(fp, cbor.decodeFromByteArray(AiFeedback.serializer(), bytes))
}
```

- [ ] **Step 2: Run, verify fail** — Run: `./gradlew :surf-ai-api:test --tests "*DtoSerializationTest*"` — Expected: FAIL.

- [ ] **Step 3: Implement**

`AiFeedback.kt`:
```kotlin
package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
sealed interface AiFeedback {
    @Serializable data class FalsePositive(val categories: Set<AiCategory>? = null) : AiFeedback
    @Serializable data class FalseNegative(val categories: Set<AiCategory>) : AiFeedback
    @Serializable data object Correct : AiFeedback
}
```
`AiFeedbackResult.kt`:
```kotlin
package dev.slne.surf.ai.api.model

import kotlinx.serialization.Serializable

@Serializable
enum class AiFeedbackResult { ACCEPTED, EXPIRED }
```

- [ ] **Step 4: Run, verify pass.** Run: `./gradlew :surf-ai-api:test` — Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(api): add AiFeedback sealed type + AiFeedbackResult"`

---

## Task 4: Extend `AIInstance` + admin DTO

**Files:**
- Modify: `surf-ai-api/src/main/kotlin/dev/slne/surf/ai/api/AIInstance.kt`
- Create: `surf-ai-api/src/main/kotlin/dev/slne/surf/ai/api/model/ModelVersionInfo.kt`

**Interfaces:**
- Produces: `AIInstance.check(input: String): AiCheckResult`, `AIInstance.feedback(requestId: UUID, feedback: AiFeedback): AiFeedbackResult`; `@Serializable data class ModelVersionInfo(val version: Int, val active: Boolean, val createdAtEpochMs: Long, val metrics: Map<String, Float>)`.

- [ ] **Step 1: Add methods to `AIInstance`** (keep the existing companion delegation):

```kotlin
interface AIInstance {
    val dataPath: Path
    suspend fun check(input: String): AiCheckResult
    suspend fun feedback(requestId: UUID, feedback: AiFeedback): AiFeedbackResult

    companion object : AIInstance by instance { val INSTANCE get() = instance }
}
```

- [ ] **Step 2: Create `ModelVersionInfo.kt`** (code as in Interfaces).

- [ ] **Step 3: Compile** — Run: `./gradlew :surf-ai-api:compileKotlin` — Expected: SUCCESS.

- [ ] **Step 4: Commit** — `git commit -am "feat(api): add check/feedback to AIInstance + ModelVersionInfo"`

---

## Task 5: RPC contracts (`surf-ai-core-common`)

**Files:**
- Create: `surf-ai-core/surf-ai-core-common/.../rpc/AiRpcService.kt`
- Create: `surf-ai-core/surf-ai-core-common/.../rpc/AiAdminRpcService.kt`

**Interfaces:**
- Consumes: DTOs from `surf-ai-api`.
- Produces: `@RpcService interface AiRpcService { suspend fun check(input: String): AiCheckResult; suspend fun feedback(requestId: @Contextual UUID, feedback: AiFeedback): AiFeedbackResult }` and `@RpcService interface AiAdminRpcService { suspend fun listVersions(): List<ModelVersionInfo>; suspend fun rollbackTo(version: Int); suspend fun triggerRetrain() }`.

- [ ] **Step 1: Write both interfaces** (package `dev.slne.surf.ai.core.common.rpc`, annotate UUID param `@Contextual`).

- [ ] **Step 2: Build (KSP generates proxies)** — Run: `./gradlew :surf-ai-core:surf-ai-core-common:compileKotlin` — Expected: SUCCESS (descriptors generated).

- [ ] **Step 3: Commit** — `git commit -am "feat(core-common): add AiRpcService + AiAdminRpcService contracts"`

---

## Task 6: Client bridge (`surf-ai-client-common`)

Wire `AIInstance.check/feedback` on the Minecraft side to the RPC proxy.

**Files:**
- Modify: `surf-ai-client/surf-ai-api-client-common/.../AiClientInstance.kt`
- Modify: `surf-ai-client/surf-ai-api-client-common/.../proxies.kt`

**Interfaces:**
- Consumes: `AiRpcService` proxy via `rabbitApi.createRpcService<AiRpcService>()`.
- Produces: `AiClientInstance` implements `check`/`feedback` by delegating to `aiProxy`.

- [ ] **Step 1: Add the proxy** in `proxies.kt`:

```kotlin
val aiProxy by lazy { rabbitApi.createRpcService<AiRpcService>() }
val aiAdminProxy by lazy { rabbitApi.createRpcService<AiAdminRpcService>() }
```

- [ ] **Step 2: Implement in `AiClientInstance`** (it already `abstract class ... : AIInstance`):

```kotlin
override suspend fun check(input: String) = aiProxy.check(input)
override suspend fun feedback(requestId: UUID, feedback: AiFeedback) = aiProxy.feedback(requestId, feedback)
```

- [ ] **Step 3: Compile paper + velocity clients** — Run: `./gradlew :surf-ai-client:surf-ai-api-client-paper:compileKotlin :surf-ai-client:surf-ai-api-client-velocity:compileKotlin` — Expected: SUCCESS.

- [ ] **Step 4: Commit** — `git commit -am "feat(client): bridge AIInstance to AiRpcService proxy"`

---

## Task 7: DB tables (`surf-ai-microservice`)

**Files:**
- Create: `surf-ai-microservice/.../db/tables/AiSeedSampleTable.kt`
- Create: `surf-ai-microservice/.../db/tables/AiLabeledSampleTable.kt`
- Create: `surf-ai-microservice/.../db/tables/AiModelVersionTable.kt`
- Delete: `surf-ai-microservice/.../db/tables/ExampleTable.kt` (and its repo) once unused.

**Interfaces:**
- Produces: three `AuditableLongIdTable` objects. Labels/metrics stored as JSON text (kotlinx). Categories serialized as their enum `name`.

- [ ] **Step 1: Write the tables**

```kotlin
// AiSeedSampleTable.kt
object AiSeedSampleTable : AuditableLongIdTable("ai_seed_sample") {
    val text = text("text")
    val labels = text("labels")        // JSON array of AiCategory names
    val language = varchar("language", 8)
    val source = varchar("source", 64)
}
// AiLabeledSampleTable.kt  — only path for live chat into the DB
object AiLabeledSampleTable : AuditableLongIdTable("ai_labeled_sample") {
    val text = text("text")
    val labels = text("labels")            // JSON array of AiCategory names (corrected truth)
    val feedbackType = varchar("feedback_type", 32) // FALSE_POSITIVE | FALSE_NEGATIVE | CORRECT
    val source = varchar("source", 64)
    val quarantined = bool("quarantined").default(false)
    val weight = float("weight").default(1f)
    val originModelVersion = integer("origin_model_version").nullable()
}
// AiModelVersionTable.kt
object AiModelVersionTable : AuditableLongIdTable("ai_model_version") {
    val version = integer("version").uniqueIndex()
    val s3Key = varchar("s3_key", 256)
    val embeddingModelId = varchar("embedding_model_id", 128)
    val metrics = text("metrics")          // JSON map<String,Float>
    val trainingDataHash = varchar("training_data_hash", 128)
    val active = bool("active").default(false)
}
```
Import `AuditableLongIdTable` from `dev.slne.surf.database.table` (as in the deleted example).

- [ ] **Step 2: Register schema creation** in `AiMicroservice.onBootstrap` — replace `SchemaUtils.create(ExampleTable)` with `SchemaUtils.create(AiSeedSampleTable, AiLabeledSampleTable, AiModelVersionTable)`.

- [ ] **Step 3: Compile** — Run: `./gradlew :surf-ai-microservice:compileKotlin` — Expected: SUCCESS.

- [ ] **Step 4: Commit** — `git commit -am "feat(microservice): add ai_seed_sample, ai_labeled_sample, ai_model_version tables"`

---

## Task 8: Repositories

**Files:**
- Create: `surf-ai-microservice/.../db/repositories/LabeledSampleRepository.kt`
- Create: `surf-ai-microservice/.../db/repositories/ModelVersionRepository.kt`
- Delete: `ExampleRepository.kt`

**Interfaces:**
- Produces:
  - `LabeledSampleRepository.insert(text: String, labels: Set<AiCategory>, feedbackType: String, source: String, originModelVersion: Int?)`
  - `ModelVersionRepository.activeVersion(): ModelVersionRow?`; `insertVersion(...)`; `setActive(version: Int)`; `all(): List<ModelVersionRow>`
  - `data class ModelVersionRow(val version: Int, val s3Key: String, val metrics: Map<String,Float>, val active: Boolean, val createdAtEpochMs: Long)`

- [ ] **Step 1: Write the failing test** `surf-ai-microservice/src/test/kotlin/.../db/ModelVersionRepositoryTest.kt` — a Testcontainers-less integration test that assumes the dev Postgres from Task 0 is up (connect via `DatabaseApi` test config). Test: insert two versions, `setActive(2)`, assert `activeVersion().version == 2`.

```kotlin
@Test fun `setActive flips the active pointer`() = runBlocking {
    ModelVersionRepository.insertVersion(1, "models/head/head-v1.onnx", "intfloat/multilingual-e5-small", mapOf(), "h1")
    ModelVersionRepository.insertVersion(2, "models/head/head-v2.onnx", "intfloat/multilingual-e5-small", mapOf(), "h2")
    ModelVersionRepository.setActive(2)
    assertEquals(2, ModelVersionRepository.activeVersion()!!.version)
}
```

- [ ] **Step 2: Run, verify fail** — Run: `./gradlew :surf-ai-microservice:test --tests "*ModelVersionRepositoryTest*"` — Expected: FAIL.

- [ ] **Step 3: Implement both repos** using `suspendTransaction { }` (pattern from `ExampleRepository`). `setActive` must, in one transaction, set `active=false` for all then `active=true` where `version == v`. Serialize/deserialize `labels`/`metrics` with `kotlinx.serialization.json.Json`.

- [ ] **Step 4: Run, verify pass** (dev Postgres running) — Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): add labeled-sample + model-version repositories"`

---

## Task 9: Microservice config (`AiConfig`)

**Files:**
- Modify: `surf-ai-microservice/.../config/AiConfig.kt`

**Interfaces:**
- Produces: config fields: `s3Endpoint`, `s3Bucket`, `s3AccessKey`, `s3SecretKey`, `trainerBaseUrl`, `ttlCacheMinutes`, `overrideCacheMaxSize`, `batchWindowMillis`, `batchMaxSize`, `hotReloadPollSeconds`, `embeddingPrefix` (default `"query: "`).

- [ ] **Step 1: Replace `something: String`** with the fields above (sensible defaults matching Task 0 dev services; `trainerBaseUrl = "http://localhost:8000"`). Keep the `SpongeYmlConfigClass` companion.

- [ ] **Step 2: Compile** — Run: `./gradlew :surf-ai-microservice:compileKotlin` — Expected: SUCCESS.

- [ ] **Step 3: Commit** — `git commit -am "feat(microservice): flesh out AiConfig (s3, trainer, cache, batching)"`

---

## Task 10: S3 client wrapper

**Files:**
- Create: `surf-ai-microservice/.../storage/ModelStorage.kt`
- Modify: `surf-ai-microservice/build.gradle.kts` (add `io.minio:minio:8.5.12`)
- Test: `surf-ai-microservice/src/test/kotlin/.../storage/ModelStorageTest.kt`

**Interfaces:**
- Produces: `class ModelStorage(config)` with `suspend fun download(key: String, dest: Path)`, `suspend fun exists(key: String): Boolean`. Uses MinIO java client against `s3Endpoint`.

- [ ] **Step 1: Write failing test** — put a small object via the client, `download` it, assert bytes match (dev MinIO from Task 0).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `ModelStorage`** — build `MinioClient.builder().endpoint(s3Endpoint).credentials(access, secret).build()`; `download` streams `getObject` to `dest` (create parent dirs); wrap blocking calls in `withContext(Dispatchers.IO)`.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): add MinIO-backed ModelStorage"`

---

## Task 11: Embedding engine (tokenizer + ONNX + pooling)

**Files:**
- Create: `surf-ai-microservice/.../inference/EmbeddingModel.kt`
- Modify: `surf-ai-microservice/build.gradle.kts` (add `ai.djl.huggingface:tokenizers:0.30.0`, `com.microsoft.onnxruntime:onnxruntime:1.19.2`)
- Test: `surf-ai-microservice/src/test/kotlin/.../inference/EmbeddingModelTest.kt`

**Interfaces:**
- Consumes: `EmbeddingModel(onnxPath: Path, tokenizerPath: Path, prefix: String)`.
- Produces: `fun embed(texts: List<String>): Array<FloatArray>` (each length 384, L2-normalized).

- [ ] **Step 1: Write failing test** — load the model from a fixture (downloaded in Task 19 or a tiny committed test asset), embed `"hello"` twice, assert vector length 384, assert deterministic (same vector), assert L2 norm ≈ 1.0. If no model asset exists yet at this point, mark the test `@Disabled("enabled after Task 19 exports embedding.onnx to test resources")` and instead unit-test the pooling+normalize math on a hand-made `[tokens,dims]` matrix + mask.

```kotlin
@Test fun `mean pool + l2 normalize`() {
    val hidden = arrayOf(floatArrayOf(1f,0f), floatArrayOf(3f,0f)) // 2 tokens, 2 dims
    val mask = longArrayOf(1,1)
    val v = EmbeddingModel.meanPoolNormalize(hidden, mask)
    assertEquals(1.0f, v[0], 1e-5f); assertEquals(0.0f, v[1], 1e-5f) // mean=(2,0)->normalize->(1,0)
}
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**
  - `HuggingFaceTokenizer.newInstance(tokenizerPath)`; encode `prefix + text` → `ids` (Long[]), `attentionMask` (Long[]).
  - `OrtEnvironment` + `OrtSession(onnxPath)`; feed `input_ids` and `attention_mask` as `OnnxTensor` shape `[batch, seqLen]` (pad batch to max len with mask 0).
  - Read output `last_hidden_state` `[batch, seqLen, 384]`.
  - `meanPoolNormalize(hidden: Array<FloatArray>, mask: LongArray): FloatArray` — sum token vectors where mask==1, divide by count, then L2-normalize. Expose it as a top-level/companion function for unit testing.
  - Guard the ORT session with a mutex or make it thread-confined; ORT sessions are thread-safe for `run` so concurrent calls are fine.

- [ ] **Step 4: Run, verify pass** (pooling unit test at least).

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): add EmbeddingModel (HF tokenizer + e5 ONNX + mean-pool)"`

---

## Task 12: Classification head + snapshot holder

**Files:**
- Create: `surf-ai-microservice/.../inference/ClassificationHead.kt`
- Create: `surf-ai-microservice/.../inference/ModelHolder.kt`
- Test: `surf-ai-microservice/src/test/kotlin/.../inference/ClassificationHeadTest.kt`

**Interfaces:**
- Produces:
  - `class ClassificationHead(onnxPath: Path)` with `fun score(embeddings: Array<FloatArray>): Array<FloatArray>` → sigmoid(logits)*100, length 6 per row, order = `AiCategory.entries`.
  - `class ModelHolder` holding the current `EmbeddingModel` + `ClassificationHead` + `activeVersion: Int?`; `suspend fun reloadTo(version: ModelVersionRow, storage: ModelStorage, config)`; thread-safe swap via `@Volatile` reference; `fun ready(): Boolean`.

- [ ] **Step 1: Write failing test** — build a tiny linear head ONNX in the test (or load a fixture) mapping 384→6; assert `score` returns 6 values each in `0..100`. If building ONNX in a Kotlin test is impractical, `@Disabled` until Task 20 provides a fixture and instead unit-test `sigmoidTimes100(logit)` math (`0f -> 50f`).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** — ORT session for head; input tensor `embedding` `[batch,384]`; output `logits` `[batch,6]`; map with `100f / (1f + exp(-logit))`. `ModelHolder.reloadTo` downloads embedding assets (once) + `head-v{n}.onnx` via `ModelStorage`, constructs new models, atomically swaps the volatile ref, updates `activeVersion`.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): add ClassificationHead + hot-swappable ModelHolder"`

---

## Task 13: Micro-batching executor

**Files:**
- Create: `surf-ai-microservice/.../inference/BatchingEmbedder.kt`
- Test: `surf-ai-microservice/src/test/kotlin/.../inference/BatchingEmbedderTest.kt`

**Interfaces:**
- Consumes: `EmbeddingModel`, `config.batchWindowMillis`, `config.batchMaxSize`.
- Produces: `suspend fun embed(text: String): FloatArray` — internally coalesces concurrent calls within the time window into one `EmbeddingModel.embed(batch)` call.

- [ ] **Step 1: Write failing test** — launch 50 concurrent `embed` calls; assert all return length-384 vectors and (via a spy counting `EmbeddingModel.embed` invocations) that they were grouped into far fewer than 50 batch calls.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** — a `Channel<Pair<String, CompletableDeferred<FloatArray>>>`; a single consumer coroutine drains up to `batchMaxSize` items or waits `batchWindowMillis`, calls `embed(batch)`, completes each deferred with its row. Run on `Dispatchers.Default`.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): add micro-batching embedder"`

---

## Task 14: Caches (TTL request cache + override cache)

**Files:**
- Create: `surf-ai-microservice/.../inference/RequestCache.kt`
- Create: `surf-ai-microservice/.../inference/OverrideCache.kt`
- Modify: `surf-ai-microservice/build.gradle.kts` (add `com.github.benmanes.caffeine:caffeine:3.1.8`)
- Test: `surf-ai-microservice/src/test/kotlin/.../inference/CachesTest.kt`

**Interfaces:**
- Produces:
  - `RequestCache(ttlMinutes)` : `data class Entry(val text: String, val embedding: FloatArray, val scores: List<AiCategoryScore>)`; `put(id: UUID, entry: Entry)`, `get(id: UUID): Entry?`.
  - `OverrideCache(maxSize, ttlMinutes)` : `putCorrection(text: String, scores: List<AiCategoryScore>)`, `get(text: String): List<AiCategoryScore>?` (key = normalized text: trim + lowercase + collapse whitespace via `normalize(text)`).

- [ ] **Step 1: Write failing tests** — `RequestCache.get` returns what was put; `OverrideCache.get("  HELLO  ") == get("hello")` (normalization); expired entries (use a tiny ttl override for the test) return null.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement with Caffeine** — `Caffeine.newBuilder().expireAfterWrite(ttl).maximumSize(...).build<K,V>()`. Provide a `normalize(text)` top-level fun (shared).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): add Caffeine TTL request cache + override cache"`

---

## Task 15: InferenceService (orchestration)

**Files:**
- Create: `surf-ai-microservice/.../inference/InferenceService.kt`
- Test: `surf-ai-microservice/src/test/kotlin/.../inference/InferenceServiceTest.kt`

**Interfaces:**
- Consumes: `ModelHolder`, `BatchingEmbedder`, `RequestCache`, `OverrideCache`.
- Produces:
  - `suspend fun check(text: String): AiCheckResult` — if `!modelHolder.ready()` return `AiCheckResult(randomId, emptyList())`; else embed → override-cache lookup (if hit, use corrected scores) else head.score → build sorted `List<AiCategoryScore>` (desc) → `RequestCache.put` → return with new `requestId`.
  - `fun applyOverride(text: String, corrected: List<AiCategoryScore>)` → writes OverrideCache.

- [ ] **Step 1: Write failing test** — with a stub `ModelHolder` returning fixed scores: `check("x")` returns a non-null `requestId`, 6 scores sorted descending, and the entry is retrievable from the injected `RequestCache`.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** the orchestration exactly as specified in Interfaces.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): add InferenceService orchestration"`

---

## Task 16: RPC impl + feedback persistence + service registration

**Files:**
- Create: `surf-ai-microservice/.../rpc/AiRpcServiceImpl.kt`
- Modify: `surf-ai-microservice/.../AiMicroservice.kt` (wire singletons, register service)
- Test: `surf-ai-microservice/src/test/kotlin/.../rpc/FeedbackTest.kt`

**Interfaces:**
- Consumes: `InferenceService`, `RequestCache`, `LabeledSampleRepository`, `ModelHolder`.
- Produces: `AiRpcServiceImpl` implementing `AiRpcService`. `feedback` logic:
  1. `val entry = requestCache.get(requestId) ?: return AiFeedbackResult.EXPIRED`
  2. compute corrected label set from `entry.scores` + feedback (FalsePositive removes/zeros categories; FalseNegative adds; Correct = current above-threshold set) → persist via `LabeledSampleRepository.insert(entry.text, labels, feedbackType, source="ingame", modelHolder.activeVersion)`.
  3. `inferenceService.applyOverride(entry.text, correctedScores)` where corrected FP→that category's score set low (e.g. 0f), FN→high (e.g. 95f).
  4. return `AiFeedbackResult.ACCEPTED`.

- [ ] **Step 1: Write failing test** — put a `RequestCache.Entry` manually, call `feedback(id, FalseNegative(setOf(SEXUAL)))`, assert result ACCEPTED, assert a row landed in `ai_labeled_sample` with SEXUAL in labels, and `OverrideCache.get(text)` now shows SEXUAL high. Also: unknown id → EXPIRED.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `AiRpcServiceImpl`; in `AiMicroservice.onBootstrap` construct the object graph (config, `ModelStorage`, `ModelHolder`, `BatchingEmbedder`, caches, `InferenceService`, repos) and `rabbitApi.registerRpcService<AiRpcService>(AiRpcServiceImpl(...))`.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): implement AiRpcService (check + feedback persistence)"`

---

## Task 17: Admin RPC + hot-reload watcher + trainer trigger

**Files:**
- Create: `surf-ai-microservice/.../rpc/AiAdminRpcServiceImpl.kt`
- Create: `surf-ai-microservice/.../inference/HotReloadWatcher.kt`
- Create: `surf-ai-microservice/.../trainer/TrainerClient.kt`
- Test: `surf-ai-microservice/src/test/kotlin/.../HotReloadWatcherTest.kt`

**Interfaces:**
- Produces:
  - `AiAdminRpcServiceImpl`: `listVersions()` → maps `ModelVersionRepository.all()` to `ModelVersionInfo`; `rollbackTo(v)` → `setActive(v)` + `modelHolder.reloadTo(...)` + mark labeled samples with `originModelVersion > v` as `quarantined=true`; `triggerRetrain()` → `TrainerClient.retrain()`.
  - `HotReloadWatcher(pollSeconds)`: coroutine loop comparing `ModelVersionRepository.activeVersion()` to `modelHolder.activeVersion`, reloading on change.
  - `TrainerClient(baseUrl)`: `suspend fun retrain()` → HTTP `POST {baseUrl}/retrain` (use Java 11 `HttpClient` on `Dispatchers.IO`).

- [ ] **Step 1: Write failing test** — set active version to 2 in the repo while holder is at 1; run one watcher tick; assert `modelHolder.reloadTo` was invoked with version 2 (use a stub holder capturing the call).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** the three files; start `HotReloadWatcher` from `AiMicroservice.onBootstrap` on the microservice scope; register `AiAdminRpcService`.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(microservice): admin RPC, hot-reload watcher, trainer trigger client"`

---

## Task 18: Paper ingame feedback dialog

**Files:**
- Create: `surf-ai-client/surf-ai-api-client-paper/.../FeedbackDialog.kt`
- Create: `surf-ai-client/surf-ai-api-client-paper/.../AiFeedbackCommand.kt` (debug/test entry point)
- Modify: `PaperMain.kt` (register command)

**Interfaces:**
- Consumes: `AIInstance.INSTANCE.feedback(...)`, `AiCheckResult`.
- Produces: `FeedbackDialog.open(player, requestId, result)` — a dialog (pattern: existing `ExampleDialog`) showing the message scores + inputs to pick FalsePositive/FalseNegative categories; on confirm calls `feedback(...)` (use `afterAction(WAIT_FOR_RESPONSE)`), then opens a **notice** dialog (`type { notice { } }`) showing the `AiFeedbackResult`.

- [ ] **Step 1: Implement `FeedbackDialog`** using the `dev.slne.surf.api.paper.dialog` DSL:
  - `base { title("KI-Feedback"); body { plainMessage { scores per category }; input { boolean per category "war falsch?" / "fehlte?" } } }`
  - `type { confirmation { yes(submitButton) no(cancelButton) } }`
  - submit reads booleans → builds `AiFeedback` → `feedback(requestId, fb)` → `player.showDialog(noticeDialog(result))`.
  - `noticeDialog(result)`: `dialog { base { title("Feedback"); body { plainMessage { if ACCEPTED "Danke – fließt ins nächste Training ein." else "Anfrage zu alt (abgelaufen)." } } }; type { notice { } } }`.

- [ ] **Step 2: Implement `/aifeedback <uuid>`** command (debug/test hook only — the real trigger comes from the moderation plugin, which is out of scope). Keep it minimal: parse the UUID arg, then `FeedbackDialog.open(player, uuid, AiCheckResult(uuid, emptyList()))`. The score list is empty here because the client does not hold the cache; the dialog still lets a mod submit feedback for that `requestId`, and the microservice resolves it against its TTL cache.

- [ ] **Step 3: Register command** in `PaperMain.onEnableAsync`.

- [ ] **Step 4: Compile** — Run: `./gradlew :surf-ai-client:surf-ai-api-client-paper:compileKotlin` — Expected: SUCCESS.

- [ ] **Step 5: Commit** — `git commit -am "feat(paper): ingame feedback dialog + notice result dialog"`

---

## Task 19: Python trainer scaffold

**Files:**
- Create: `surf-ai-trainer/pyproject.toml`, `surf-ai-trainer/README.md`
- Create: `surf-ai-trainer/app/__init__.py`, `app/config.py`
- Create: `surf-ai-trainer/tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` (pydantic-settings) with `postgres_dsn`, `s3_endpoint`, `s3_bucket`, `s3_access_key`, `s3_secret_key`, `embedding_model_id="intfloat/multilingual-e5-small"`, `embedding_prefix="query: "`. Defaults match Task 0 dev services.

- [ ] **Step 1: Write `pyproject.toml`** with deps: `torch` (CPU), `transformers`, `optimum[onnxruntime]`, `onnxruntime`, `numpy`, `scikit-learn`, `fastapi`, `uvicorn[standard]`, `apscheduler`, `boto3`, `psycopg[binary]`, `pydantic-settings`, and dev dep `pytest`.

- [ ] **Step 2: Write failing test** `test_config.py` — `Settings()` loads defaults; `embedding_prefix == "query: "`.

- [ ] **Step 3: Run, verify fail** — Run: `cd surf-ai-trainer && pip install -e . && pytest tests/test_config.py` — Expected: FAIL (no module).

- [ ] **Step 4: Implement `config.py`.**

- [ ] **Step 5: Run, verify pass.**

- [ ] **Step 6: Commit** — `git commit -am "feat(trainer): scaffold python project + config"`

---

## Task 20: Embedding export + Python embedding util

**Files:**
- Create: `surf-ai-trainer/app/embedding.py`
- Create: `surf-ai-trainer/app/export_embedding.py`
- Test: `surf-ai-trainer/tests/test_embedding.py`

**Interfaces:**
- Produces:
  - `export_embedding.export(dest_dir)` — uses `optimum` to export `multilingual-e5-small` to `model.onnx` + saves `tokenizer.json`; uploads both to S3 under `models/embedding/`.
  - `embedding.Embedder(onnx_path, tokenizer_path, prefix)` with `embed(texts: list[str]) -> np.ndarray` (N×384, mean-pooled + L2-normalized) — **must byte-match the Kotlin `EmbeddingModel`** (same onnx, same pooling).

- [ ] **Step 1: Write failing test** — export to a temp dir (network needed once; mark `@pytest.mark.slow`), then `Embedder.embed(["hello","hello"])` → shape (2,384), rows identical, L2 norm ≈ 1. Add a fast unit test for `mean_pool_normalize(hidden, mask)` mirroring the Kotlin test values.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** both modules; `Embedder` runs onnxruntime `InferenceSession`, feeds `input_ids`/`attention_mask`, mean-pools `last_hidden_state`, normalizes. Also copy the exported `model.onnx` + `tokenizer.json` into `surf-ai-microservice/src/test/resources/models/embedding/` so Tasks 11/12 fixtures can enable their disabled tests.

- [ ] **Step 4: Run, verify pass** (fast unit test always; slow test when network available).

- [ ] **Step 5: Enable Task 11/12 disabled tests** now that the fixture exists; run `./gradlew :surf-ai-microservice:test` — Expected: PASS. **Cross-consistency check:** add a Kotlin test that loads the committed fixture and asserts `embed("hallo welt")` matches a vector recorded from the Python `Embedder` (store the reference vector as a test resource JSON); tolerance `1e-4`.

- [ ] **Step 6: Commit** — `git commit -am "feat(trainer): embedding export + Embedder; JVM/Python vector parity test"`

---

## Task 21: Seed corpus

**Files:**
- Create: `surf-ai-trainer/seed/seed_de.jsonl`, `seed/seed_en.jsonl`, `seed/seed_minecraft.jsonl`
- Create: `surf-ai-trainer/app/seed.py`
- Create: `surf-ai-trainer/scripts/generate_seed.py` (LLM-assisted generator, documented, offline-runnable stub + instructions)
- Test: `surf-ai-trainer/tests/test_seed.py`

**Interfaces:**
- Produces: JSONL rows `{"text": str, "labels": [AiCategory...], "language": "de|en", "source": str}`. `seed.load_seed() -> list[dict]`; `seed.bootstrap_into_db(conn)` inserts rows into `ai_seed_sample` if empty.

- [ ] **Step 1: Author seed data** — for EACH category ≥20 clear positives per language; PLUS a dedicated block of Minecraft **hard negatives** (all-labels-empty) e.g. `"geh doch einfach hinten auf den berg und spring runter"`, `"ich bomb gleich deine base weg"`, `"kill the ender dragon"`, `"geh sterben im pvp"`; PLUS the required positives `"kys"`→SELF_HARM, `"geh dich doch einfach umbringen"`→SELF_HARM, `"du hurensohn"`→HARASSMENT, `"penis"`→SEXUAL. Document public-dataset import (Jigsaw/GermEval/HASOC) as TODO-with-instructions in `scripts/generate_seed.py` docstring (license note), not committed raw.

- [ ] **Step 2: Write failing test** — `load_seed()` returns >200 rows; every `labels` entry is a valid `AiCategory`; the four required examples are present with the stated labels; at least 10 hard-negative (empty-label) Minecraft rows exist.

- [ ] **Step 3: Run, verify fail.**

- [ ] **Step 4: Implement `seed.py`** (loader + `bootstrap_into_db`).

- [ ] **Step 5: Run, verify pass.**

- [ ] **Step 6: Commit** — `git commit -am "feat(trainer): seed corpus (DE/EN + minecraft hard negatives)"`

---

## Task 22: Head training + ONNX export

**Files:**
- Create: `surf-ai-trainer/app/head.py` (PyTorch MLP), `app/train.py`
- Test: `surf-ai-trainer/tests/test_train.py`

**Interfaces:**
- Produces:
  - `head.HeadNet(input=384, hidden=256, out=6)` (Linear→ReLU→Dropout→Linear).
  - `train.train_head(samples, embedder) -> (torch_module, metrics)` — multi-label `BCEWithLogitsLoss`, class-weighted; labels vector length 6 in category order.
  - `train.export_head_onnx(module, dest)` — exports with input name `embedding` `[batch,384]`, output name `logits` `[batch,6]`, dynamic batch axis.

- [ ] **Step 1: Write failing test** — train on a tiny synthetic set (10 rows) for a few epochs; assert the exported ONNX, run via onnxruntime on a known embedding, produces 6 logits; assert overfit sanity: a trained-on `"kys"` row scores SELF_HARM logit highest.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `HeadNet`, training loop (Adam, ~N epochs, early stop on holdout), `export_head_onnx` (`torch.onnx.export`, `dynamic_axes={'embedding':{0:'b'},'logits':{0:'b'}}`).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(trainer): MLP head training + ONNX export"`

---

## Task 23: Eval, regression guard, version promotion

**Files:**
- Create: `surf-ai-trainer/app/evaluate.py`, `app/promote.py`, `app/db.py`
- Test: `surf-ai-trainer/tests/test_promote.py`

**Interfaces:**
- Produces:
  - `db.py`: psycopg helpers `read_seed()`, `read_labeled(non_quarantined=True)`, `insert_version(version, s3_key, embedding_model_id, metrics, hash)`, `set_active(version)`, `active_version()`, `next_version()`. **Must use the exact column names from Task 7's `ai_model_version` / `ai_seed_sample` / `ai_labeled_sample` tables** (`version, s3_key, embedding_model_id, metrics, training_data_hash, active`; `text, labels, language, source`; `text, labels, feedback_type, source, quarantined, weight, origin_model_version`). `metrics`/`labels` are JSON text. Note: Exposed's `AuditableLongIdTable` also creates `id`, `created_at`, `updated_at` columns — `set_active` must not touch those.
  - `evaluate.py`: `holdout_split(samples)`, `metrics(module, embedder, holdout) -> dict[str,float]` (per-category precision/recall + macro-F1).
  - `promote.py`: `maybe_promote(new_metrics, active_metrics, min_macro_f1_delta=-0.02) -> bool` — promote only if macro-F1 not worse than active by more than the tolerance.

- [ ] **Step 1: Write failing test** — `maybe_promote` returns False when new macro-F1 is 0.10 below active; True when equal/better; True when there is no active version (bootstrap).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** the three modules.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(trainer): eval metrics, regression guard, version promotion"`

---

## Task 24: Trainer orchestration + FastAPI + scheduler

**Files:**
- Create: `surf-ai-trainer/app/pipeline.py`, `app/main.py`
- Test: `surf-ai-trainer/tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `pipeline.run_retrain()` — load seed+labeled → embed → train head → eval → `maybe_promote` → on promote: upload `head-v{n}.onnx` to S3, `insert_version`, `set_active`. Returns a summary dict.
  - `main.py`: FastAPI app; `POST /retrain` → runs `run_retrain()` in a threadpool, returns summary; `GET /health` → `{"status":"ok"}`; APScheduler `BackgroundScheduler` nightly cron calling `run_retrain`; on startup, if S3 has no embedding model, run `export_embedding.export`; if no active version, run `run_retrain` (bootstrap).

- [ ] **Step 1: Write failing test** — with a fake S3 (moto or a stub) + the dev Postgres, `run_retrain()` from empty DB (seed only) produces version 1, sets it active, uploads a `head-v1.onnx` object. Assert `active_version() == 1`.

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `pipeline.run_retrain` and `main.py` (FastAPI + APScheduler + startup bootstrap).

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit** — `git commit -am "feat(trainer): retrain pipeline + FastAPI (/retrain,/health) + nightly scheduler"`

---

## Task 25: Dockerfiles + prod/Coolify docker-compose

**Files:**
- Create: `surf-ai-microservice/Dockerfile`
- Create: `surf-ai-trainer/Dockerfile`
- Create: `docker-compose.yml` (repo root — Coolify-consumable)
- Create: `.env.example`

**Interfaces:**
- Produces: two images + a compose deploying `microservice` + `trainer` (+ optional bundled Postgres/RabbitMQ/MinIO for non-managed Coolify). Env from `.env`.

- [ ] **Step 1: Microservice Dockerfile** — multi-stage: `gradle:jdk21` builds the standalone jar (`./gradlew :surf-ai-microservice:shadowJar` or the surf-standalone task — verify the produced artifact name), runtime `eclipse-temurin:21-jre`, `CMD ["java","-jar","/app/app.jar"]`, mounts `config/`.

- [ ] **Step 2: Trainer Dockerfile** — `python:3.11-slim`, install `.`, `CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]`, healthcheck `curl -f http://localhost:8000/health`.

- [ ] **Step 3: Root `docker-compose.yml`** — services `microservice` (env: DSN, RabbitMQ, S3, `TRAINER_BASE_URL=http://trainer:8000`) and `trainer` (env: DSN, S3). Reference `.env`. Document Coolify usage in a comment.

- [ ] **Step 4: Verify builds** — Run: `docker build -f surf-ai-trainer/Dockerfile surf-ai-trainer` and `docker build -f surf-ai-microservice/Dockerfile .` — Expected: both build. (Skip if the gradle standalone task name needs confirmation — resolve by inspecting `./gradlew :surf-ai-microservice:tasks`.)

- [ ] **Step 5: Commit** — `git commit -am "chore: dockerfiles + coolify docker-compose + .env.example"`

---

## Task 26: CLAUDE.md + repo docs

**Files:**
- Create: `CLAUDE.md` (repo root)

- [ ] **Step 1: Write `CLAUDE.md`** covering:
  - **Project purpose** (1 paragraph) + link to the spec and this plan.
  - **Module map** (api / core-common / core-client / client-* / microservice / trainer / docker).
  - **Architecture rules:** inference in microservice JVM; RPC = RabbitMQ CBOR; DTOs @Serializable; category order is a wire contract; no per-message DB writes; caches = Caffeine; head ONNX I/O contract; embedding preprocessing identical JVM/Python.
  - **How to run locally:** `docker compose -f docker/dev/docker-compose.yml up -d`, then trainer bootstrap, then microservice.
  - **How to test:** gradle + pytest commands; dev services must be up.
  - **A prominent rule, verbatim:**
    > **NEVER use subagents / the Task tool in this repo.** Subagents burn far too many tokens here. Execute plans task-by-task inline using `superpowers:executing-plans`. This overrides any default multi-agent workflow.

- [ ] **Step 2: Commit** — `git commit -am "docs: add CLAUDE.md (architecture, run/test, no-subagents rule)"`

---

## Task 27: Golden-set + end-to-end integration test

**Files:**
- Create: `surf-ai-microservice/src/test/kotlin/.../GoldenSetTest.kt`
- Create: `surf-ai-microservice/src/test/kotlin/.../EndToEndRpcTest.kt`

**Interfaces:**
- Consumes: everything; requires dev docker-compose up + a trained head (run trainer bootstrap first).

- [ ] **Step 1: GoldenSetTest** (`@EnabledIf` dev services + active model). Assertions on `InferenceService.check(...)`:
  - `"kys"` → SELF_HARM > 60
  - `"geh dich doch einfach umbringen"` → SELF_HARM > 60
  - `"du hurensohn"` → HARASSMENT > 60
  - `"penis"` → SEXUAL > 40
  - `"geh doch einfach hinten auf den berg und spring runter"` → every score < 40

- [ ] **Step 2: EndToEndRpcTest** — boot the microservice RPC + a client `AiRpcService` proxy against dev RabbitMQ; `check()` returns 6 scores; `feedback(id, FalseNegative(setOf(SEXUAL)))` returns ACCEPTED and writes `ai_labeled_sample`; expired id → EXPIRED.

- [ ] **Step 3: Run the full flow**
```bash
docker compose -f docker/dev/docker-compose.yml up -d
cd surf-ai-trainer && uvicorn app.main:app & curl -XPOST localhost:8000/retrain   # bootstrap head-v1
cd .. && ./gradlew :surf-ai-microservice:test
```
Expected: GoldenSet + E2E PASS. (Golden thresholds may need a slightly larger/longer-trained seed — if a case fails, expand that category's seed rows and re-run trainer, do not weaken the assertion below the values above.)

- [ ] **Step 4: Commit** — `git commit -am "test: golden-set + end-to-end RPC/feedback integration tests"`

---

## Task 28: Finalize

- [ ] **Step 1:** Run the whole suite: `./gradlew test` and `cd surf-ai-trainer && pytest`. Expected: green.
- [ ] **Step 2:** Verify `docker compose config` (root) is valid.
- [ ] **Step 3:** Use `superpowers:finishing-a-development-branch` to decide merge/PR.
