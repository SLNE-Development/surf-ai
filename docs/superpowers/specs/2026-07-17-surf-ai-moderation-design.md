# surf-ai — Chat-Moderations-KI — Design / Spec

**Datum:** 2026-07-17
**Status:** Freigegeben (Brainstorming abgeschlossen), bereit für Implementation-Plan

## 1. Ziel & Kontext

Ein großer Minecraft-Server (Velocity-Proxy + Paper-Server) erzeugt ca. **50–100 Chat-Nachrichten pro Sekunde**. Jede Nachricht soll durch eine KI-Prüfung laufen, die pro Nachricht eine Liste von **Kategorie-Confidence-Scores** zurückgibt. Ein **separates** Moderations-Plugin entscheidet anhand dieser Scores über Maßnahmen — das ist **nicht** Teil dieses Projekts.

Dieses Projekt liefert ausschließlich das **Modell-System**: Inferenz-API, trainierbares Modell, Feedback-Loop, Versionierung/Snapshots, Persistenz und Deployment.

### Kern-Anforderungen

- Öffentliche API: `AIInstance.check(input)` → Liste von `(Kategorie, Confidence)` + eine `requestId` (UUID) pro Anfrage.
- Feedback-Mechanismus: `AIInstance.feedback(requestId, feedback)`, damit man falsche Einschätzungen korrigieren kann und das Modell sich anpasst.
- Primär **Deutsch + Englisch**, einfach um weitere Sprachen erweiterbar.
- Kontextverständnis statt reinem Keyword-Matching:
  - Flaggen: `"kys"`, `"geh dich doch einfach umbringen"`, `"du hurensohn"`.
  - **Nicht** flaggen: `"geh doch einfach hinten auf den berg und spring runter"` (Minecraft-Kontext), `"Penis"` als harmloser Begriff.
- Persistentes Training + Model-Snapshots mit Rollback (Schutz gegen Vergiftung durch fehlerhaftes Feedback).

### Kategorien (Multi-Label)

Eine Nachricht kann mehrere Kategorien treffen:

- `HARASSMENT` — gezielte Beleidigung/Mobbing gegen eine Person
- `SELF_HARM` — Aufforderung zu Selbstverletzung/Suizid
- `HATE_SPEECH` — Slurs/Diskriminierung (Herkunft, Religion, Geschlecht, Sexualität)
- `SEXUAL` — sexuelle Belästigung/Inhalte
- `THREAT` — reale Gewaltandrohung gegen andere (nicht Ingame)
- `CHILD_SAFETY` — Grooming / predatorisches Verhalten ggü. Minderjährigen

## 2. Architektur-Entscheidungen (getroffen)

| Entscheidung | Wahl | Begründung |
|---|---|---|
| Modell-Ansatz | Eingefrorenes multilinguales Sentence-Embedding-Model + kleiner trainierbarer Klassifikations-Kopf (MLP) | Embedding liefert multilinguales Kontextverständnis fertig; nur der winzige Kopf wird trainiert → sekundenschnelles Training, einfaches Feedback, neue Sprachen = neue Beispiele |
| Wo läuft Inferenz | **In der Microservice-JVM** (Kotlin, DJL + ONNX-Runtime) | Konsistent mit bestehender Microservice-Struktur; Modell einmal zentral geladen statt auf jedem Proxy/Server; Minecraft-Plugins bleiben dünne RPC-Clients |
| Wo läuft Training | **Python-Sidecar** (Hybrid) | Python-ML-Ökosystem (PyTorch/sentence-transformers); trainiert nur den Kopf, nie im Request-Pfad |
| Transport | **RabbitMQ-RPC** (`@RpcService`) | Bestehendes Muster des Projekts (`surf-ai-core-common`) |
| Persistenz | `dev.slne.surf.database` (R2DBC + Exposed, Postgres) | Bestehendes Muster |
| Weight-Storage | **S3 / MinIO** (Object-Storage) | Entkoppelt Trainer (Writer) und Microservice (Reader) |
| Feedback-Wirkung | **Batch-Retrain + In-JVM Override-Cache** | Sofortige gefühlte Wirkung + robuster echter Fix beim nächsten Retrain |
| Deployment | **Docker / Coolify**, inkl. `docker-compose.yml` | Ziel-Plattform des Nutzers |

### Embedding-Model

- **`intfloat/multilingual-e5-small`** (384-dim, stark für DE/EN, klein genug für CPU).
- **Identisch** in Kotlin (DJL, Inferenz) und Python (sentence-transformers, Training), damit der Vektorraum exakt übereinstimmt.
- Preprocessing (e5 verlangt Prefix, z.B. `"query: "`) wird **an einer Stelle** definiert und in beiden Welten gleich angewandt.
- Frei austauschbar per Config; Wechsel des Embedding-Models = zwingend voller Retrain des Kopfes (im Snapshot-Metadatensatz festgehalten).

## 3. Modul-Struktur

Baut auf der bestehenden Struktur auf:

| Modul | Rolle |
|---|---|
| `surf-ai-api` | Plugin-facing Interface `AIInstance` + DTOs (`@Serializable`) |
| `surf-ai-core:surf-ai-core-common` | `@RpcService`-Contracts (RabbitMQ) |
| `surf-ai-core:surf-ai-core-client` | Client-seitige Core-Config |
| `surf-ai-client:*` (common/paper/velocity) | Dünne RPC-Clients; bridgen `AIInstance` → RPC-Proxy |
| `surf-ai-microservice` | Inferenz-Engine (DJL+ONNX), Batching, Override-Cache, Snapshot-Loader/Hot-Reload, DB-Tabellen & Repos, RPC-Impl |
| `surf-ai-trainer` *(neu, Python)* | FastAPI + APScheduler; trainiert Kopf; S3-Snapshots |
| `docker/` + `docker-compose.yml` | Deployment für Coolify + lokales Dev |

## 4. API-Oberfläche (`surf-ai-api`)

```kotlin
interface AIInstance {
    val dataPath: Path
    suspend fun check(input: String): AiCheckResult
    suspend fun feedback(requestId: UUID, feedback: AiFeedback)
}

@Serializable
data class AiCheckResult(
    val requestId: @Contextual UUID,       // für spätere feedback()-Calls
    val scores: List<AiCategoryScore>      // ALLE Kategorien, sortiert desc
)

@Serializable
data class AiCategoryScore(
    val category: AiCategory,
    val confidence: Float                  // 0.0 – 100.0
)

enum class AiCategory { HARASSMENT, SELF_HARM, HATE_SPEECH, SEXUAL, THREAT, CHILD_SAFETY }

@Serializable
sealed interface AiFeedback {
    // fälschlich geflaggt; categories == null ⇒ nichts hätte flaggen sollen
    @Serializable data class FalsePositive(val categories: Set<AiCategory>? = null) : AiFeedback
    // hätte flaggen sollen
    @Serializable data class FalseNegative(val categories: Set<AiCategory>) : AiFeedback
    @Serializable data object Correct : AiFeedback
}
```

Design-Entscheidungen:

- `check()` liefert **immer alle 6 Scores** — das Moderations-Plugin thresholded selbst.
- `check()`/`feedback()` sind `suspend` (passt zu MCCoroutine, non-blocking).
- Reicher `AiFeedback`-Typ statt nur `GOOD/BAD`: bei Multi-Label ist „schlecht" mehrdeutig; der Trainer braucht *welche* Kategorie falsch war. Optionaler Convenience-Overload `feedback(uuid, FeedbackType.BAD)` = `FalsePositive(null)` kann als Shortcut ergänzt werden.
- DTOs müssen über RabbitMQ serialisierbar sein → `@Serializable` (kotlinx). **Verifikation nötig**, welches Serialisierungsformat der RabbitMQ-RPC-Layer erwartet (in der Implementierungsphase am `ExampleRpcService`-Muster prüfen).

## 5. RPC-Contract (`surf-ai-core-common`)

```kotlin
@RpcService
interface AiRpcService {
    suspend fun check(input: String): AiCheckResult
    suspend fun feedback(requestId: UUID, feedback: AiFeedback)
}

@RpcService
interface AiAdminRpcService {
    suspend fun listVersions(): List<ModelVersionInfo>
    suspend fun rollbackTo(version: Int)
    suspend fun triggerRetrain()
}
```

`AiClientInstance` (in `surf-ai-client-common`) implementiert `AIInstance`, indem es die Calls auf den `AiRpcService`-Proxy (`rabbitApi.createRpcService<AiRpcService>()`) bridged — analog zum bestehenden `exampleProxy`.

## 6. Inferenz-Pipeline (Microservice, Hot Path)

```
check(text)
  → Micro-Batching (sammelt Nachrichten in ~10–20ms-Fenster zu Batches)
  → Text-Normalisierung + e5-Preprocessing
  → Tokenizer + Embedding-Model (ONNX, frozen)  → 384-dim Vektor
  → Override-Cache-Lookup (exakter/naher Treffer? → korrigierter Score)
  → Klassifikations-Kopf (MLP, aktueller Snapshot) → 6 Logits → Sigmoid → 0–100
  → persist(requestId, text, embedding, scores) in ai_request
  → return AiCheckResult(requestId, scores)
```

- **Micro-Batching** bündelt einzelne `check()`-Calls für effiziente Embedding-Inferenz; deckt 50–100/s mit Reserve ab. Batch-Fenster + max Batchgröße konfigurierbar.
- **Thread-/Coroutine-Pool** für Embedding-Inferenz, damit RabbitMQ-Worker nicht blockieren.
- Embedding wird gespeichert (für spätere Retrains muss der Text nicht neu embeddet werden — spart Rechenzeit; Vektor gilt nur solange das Embedding-Model gleich bleibt).

## 7. Feedback-Mechanismus

- **Override-Cache (in-JVM, flüchtig):** Bei `feedback()` wird ein Override in einen LRU-Cache geschrieben (Key: normalisierter Text / Embedding-Nähe). Identische/sehr ähnliche Nachricht → sofort korrigierter Score. Wird beim nächsten Retrain überflüssig.
- **Persistenter Feedback-Store (DB):** `ai_feedback` (request_id → typ + categories + source + timestamp). Echte Trainingsquelle.
- Feedback bekommt **Quarantäne-Status** und ein Gewicht; fließt beim nächsten Retrain als Trainingssignal ein.

## 8. Model-Versionierung & Snapshots

- **Jeder Retrain = unveränderlicher Snapshot** `head-v{n}` (Gewichte in S3) + `ai_model_version`-Zeile mit: Zeitstempel, Trainingsdaten-Hash, eingeflossene Feedback-IDs, Embedding-Model-ID, **Eval-Metriken** (Precision/Recall pro Kategorie auf festem Holdout-Set).
- **Regressions-Guard:** Neuer Snapshot wird nur **aktiv promotet**, wenn Holdout-Metriken nicht signifikant schlechter als aktive Version. Sonst bleibt er als inaktiver Kandidat gespeichert.
- **Aktive Version = Pointer** in DB. Microservice lädt beim Reload den aktiven Pointer.
- **Rollback:** `rollbackTo(version)` setzt den Pointer zurück → Hot-Reload → alter Stand sofort aktiv. Markiert seit der Zielversion eingeflossenes Feedback als „verdächtig" (Quarantäne).
- **Hot-Reload:** Microservice erkennt neue aktive Version (Poll oder Trigger) und lädt den Kopf ohne Neustart.

## 9. Training-Pipeline (`surf-ai-trainer`, Python, Cold Path)

- **FastAPI + APScheduler** in einem long-running Container:
  - Nightly-Cron (interner Scheduler) **und** `POST /retrain` (on-demand, ausgelöst durch `triggerRetrain()` im Microservice).
  - `GET /health` als Coolify-Healthcheck.
- **Ablauf Retrain:**
  1. Lädt Seed-Korpus (`ai_seed_sample`) + bestätigtes Feedback (`ai_feedback` + verknüpfte `ai_request`) aus Postgres (direkter Treiber).
  2. Erzeugt Embeddings mit **identischem** frozen Model wie die JVM.
  3. Trainiert MLP-Kopf (Multi-Label, `BCEWithLogitsLoss`, ggf. Klassen-Gewichte gegen Imbalance).
  4. Evaluiert auf festem Holdout-Set → Metriken.
  5. Exportiert Kopf (ONNX oder simples Gewichts-Format) + Label-Config → **S3/MinIO** als `head-v{n}`.
  6. Schreibt `ai_model_version`-Zeile; promotet gemäß Regressions-Guard.
- Der Trainer liest Postgres direkt; **DB muss Postgres sein** (Bestätigung: `dev.slne.surf.database` läuft auf Postgres).

## 10. Persistenz (DB-Schema)

Alle Tabellen via `dev.slne.surf.database`, Exposed, `AuditableLongIdTable` (createdAt/updatedAt gratis), Repos mit `suspendTransaction`.

- **`ai_request`** — `requestId` (nativeUuid), `input_text`, `embedding` (optional, als Array/BLOB), `scores_json`, `model_version`.
- **`ai_feedback`** — `request_id` (FK), `feedback_type`, `categories`, `source`, `quarantine_status`, `weight`.
- **`ai_model_version`** — `version`, `s3_key`, `embedding_model_id`, `metrics_json`, `training_data_hash`, `is_active`.
- **`ai_seed_sample`** — `text`, `labels` (Set<AiCategory>), `language`, `source`.

## 11. Seed-Datensatz

- Kuratierter, gelabelter Multi-Label-Korpus **DE + EN**:
  - Gängige Schimpf-/Toxizitäts-Muster pro Kategorie.
  - Gezielte **Negativ-Beispiele mit Minecraft-Kontext** (z.B. „vom Berg springen", „Creeper töten", „Penis" als harmloser Begriff), damit das Modell Kontext statt Keywords lernt.
- Klein starten (einige hundert–tausend Beispiele reicht für den Kopf), wächst via Feedback.
- Als versionierte Datei(en) im Repo (`surf-ai-trainer/seed/`), beim ersten Bootstrap in `ai_seed_sample` geladen.

## 12. Deployment (Docker / Coolify)

- **`surf-ai-microservice`** → Docker-Image, Coolify-Service. Env/Config: Postgres-DSN, RabbitMQ, S3/MinIO-Creds.
- **`surf-ai-trainer`** → Docker-Image, Coolify-Service (long-running). Env: Postgres-DSN, S3/MinIO-Creds. `/health`-Healthcheck.
- **`docker-compose.yml`** im Repo-Root:
  - Services: `microservice`, `trainer`, plus Dev-Abhängigkeiten (Postgres, RabbitMQ, MinIO) für lokales Testen.
  - Von Coolify direkt konsumierbar; lokal `docker compose up` als Dev-Umgebung.
- **Dockerfiles**: JVM-Image (Gradle-Build → Runtime-JRE) + Python-Image (slim + ML-Deps).

## 13. Testing-Strategie

- **Unit (Kotlin):** DTO-Serialisierung (RabbitMQ-kompatibel), Override-Cache-Logik, Snapshot-Pointer/Rollback, Batching-Executor, Repos.
- **Golden-Set-Test:** Fester Satz erwarteter Klassifikationen als Regressions-Gate, inkl. der Beispiele:
  - `"kys"` → `SELF_HARM` hoch
  - `"geh dich doch einfach umbringen"` → `SELF_HARM` hoch
  - `"du hurensohn"` → `HARASSMENT` hoch
  - `"geh doch einfach hinten auf den berg und spring runter"` → alle niedrig
  - `"Penis"` → `SEXUAL` niedrig
- **Trainer (Python):** Tests für Export-Format, Metrik-Guard, Holdout-Eval, S3-Roundtrip (gegen MinIO).
- **Integrationstest:** Microservice ↔ Trainer Retrain-Zyklus (Trigger → neue Version → Hot-Reload → geänderte Prediction).

## 14. Nicht-Ziele (out of scope)

- Die eigentliche Moderationsaktion (Mute/Ban/Warn) — macht ein separates Plugin.
- SPAM/Werbung-Erkennung — besser per Regex/Rate-Limit im Moderations-Plugin.
- UI/Dashboard fürs Feedback (nur API/RPC).

## 15. Offene Punkte / benötigt vom Nutzer

- **Bestätigung**, dass `dev.slne.surf.database` gegen **Postgres** läuft (Trainer braucht direkten Treiber).
- **S3/MinIO-Endpoint + Credentials** (Bucket-Name, Region/Endpoint) fürs Deployment.
- **RabbitMQ-Serialisierungsformat** des RPC-Layers — in der Implementierungsphase am bestehenden `ExampleRpcService`-Muster verifizieren, damit die DTOs korrekt über die Leitung gehen.

## 16. Betriebsregeln für Folge-Agents

- Es wird eine `CLAUDE.md` im Repo-Root angelegt mit Projektkontext, Architektur, Konventionen — **und der expliziten Regel: keine Subagents verwenden** (zu hoher Token-Verbrauch). Pläne werden via `executing-plans` sequenziell abgearbeitet.
