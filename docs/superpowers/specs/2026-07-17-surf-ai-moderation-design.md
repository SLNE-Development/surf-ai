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
  - Flaggen: `"kys"`, `"geh dich doch einfach umbringen"`, `"du hurensohn"`, `"Penis"` (→ `SEXUAL`).
  - **Nicht** flaggen: `"geh doch einfach hinten auf den berg und spring runter"` (Minecraft-Kontext).
  - Feedback korrigiert nicht nur Falsch-Flags, sondern auch **zu niedrige** Scores: wird `"Penis"` z.B. nur mit 35% `SEXUAL` bewertet, hebt `FalseNegative(SEXUAL)` den Score bei künftigem Training an.
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
| `surf-ai-client:*` (common/paper/velocity) | Dünne RPC-Clients; bridgen `AIInstance` → RPC-Proxy. Paper-Client zusätzlich: Ingame-Feedback-Dialog (§5a) |
| `surf-ai-microservice` | Inferenz-Engine (DJL+ONNX), Batching, Override-Cache, Snapshot-Loader/Hot-Reload, DB-Tabellen & Repos, RPC-Impl |
| `surf-ai-trainer` *(neu, Python)* | FastAPI + APScheduler; trainiert Kopf; S3-Snapshots |
| `docker/` + `docker-compose.yml` | Deployment für Coolify + lokales Dev |

## 4. API-Oberfläche (`surf-ai-api`)

```kotlin
interface AIInstance {
    val dataPath: Path
    suspend fun check(input: String): AiCheckResult
    suspend fun feedback(requestId: UUID, feedback: AiFeedback): AiFeedbackResult
}

@Serializable
enum class AiFeedbackResult { ACCEPTED, EXPIRED }  // EXPIRED = requestId nicht mehr im TTL-Cache

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
    suspend fun feedback(requestId: UUID, feedback: AiFeedback): AiFeedbackResult
}

@RpcService
interface AiAdminRpcService {
    suspend fun listVersions(): List<ModelVersionInfo>
    suspend fun rollbackTo(version: Int)
    suspend fun triggerRetrain()
}
```

`AiClientInstance` (in `surf-ai-client-common`) implementiert `AIInstance`, indem es die Calls auf den `AiRpcService`-Proxy (`rabbitApi.createRpcService<AiRpcService>()`) bridged — analog zum bestehenden `exampleProxy`.

## 5a. Ingame-Feedback-Dialog (Paper-Client)

Damit Mods Feedback direkt ingame geben können, stellt der Paper-Client einen Dialog über euer `dev.slne.surf.api.paper.dialog`-DSL bereit (Muster: `ExampleDialog`).

- **Auslöser:** eine wiederverwendbare, anklickbare Komponente (analog `renderDialogShowComponent`), die eine `requestId` + das `AiCheckResult` trägt, sowie ein Debug-/Test-Command (z.B. `/surfai feedback <requestId>`), um den Dialog zu öffnen. Wie der Klick real ausgelöst wird (z.B. durch das Moderations-Plugin), ist nicht unser Scope — wir liefern den Dialog-Baustein.
- **Dialog-Inhalt (`base { body { ... } }`):** zeigt den Nachrichtentext + aktuelle Scores pro Kategorie; Inputs zur Erfassung des Feedbacks (welche Kategorien fälschlich/zu niedrig/zu hoch waren) → gemappt auf `AiFeedback` (`FalsePositive` / `FalseNegative` / `Correct`).
- **Submit:** Action-Button ruft `AIInstance.feedback(requestId, feedback)` (RPC → Microservice). `afterAction(WAIT_FOR_RESPONSE)`, da suspending.
- **Ergebnis:** danach wird direkt ein **Notice-Dialog** (`type { notice { } }`) angezeigt — mit dem `AiFeedbackResult` (ACCEPTED → „Feedback gespeichert, fließt beim nächsten Training ein" / EXPIRED → „Anfrage zu alt").

## 6. Inferenz-Pipeline (Microservice, Hot Path)

```
check(text)
  → Micro-Batching (sammelt Nachrichten in ~10–20ms-Fenster zu Batches)
  → Text-Normalisierung + e5-Preprocessing
  → Tokenizer + Embedding-Model (ONNX, frozen)  → 384-dim Vektor
  → Override-Cache-Lookup (exakter/naher Treffer? → korrigierter Score)
  → Klassifikations-Kopf (MLP, aktueller Snapshot) → 6 Logits → Sigmoid → 0–100
  → in flüchtigen TTL-Cache legen (requestId → text, embedding, scores)  ← KEIN DB-Write
  → return AiCheckResult(requestId, scores)
```

- **Kein DB-Write pro Nachricht.** Bei 50–100/s wären das ~4–8 Mio. Zeilen/Tag — unerwünscht. Jedes `check()`-Resultat landet nur in einem **flüchtigen In-Memory-TTL-Cache** (z.B. 15–30 Min, konfigurierbar), damit ein späteres `feedback(uuid)` die Anfrage noch auflösen kann. Nach der TTL wird die Nachricht verworfen. Persistiert wird **ausschließlich**, was per Feedback gelabelt wird (siehe §7) — plus der Seed-Korpus.
  - Implementiert mit **Caffeine** (`com.github.benmanes.caffeine.cache.Caffeine`), `.expireAfterWrite(ttl)` + `.maximumSize(...)`. Muster wie `ExampleCache` (keyed auf `requestId`).
- **Micro-Batching** bündelt einzelne `check()`-Calls für effiziente Embedding-Inferenz; deckt 50–100/s mit Reserve ab. Batch-Fenster + max Batchgröße konfigurierbar.
- **Thread-/Coroutine-Pool** für Embedding-Inferenz, damit RabbitMQ-Worker nicht blockieren.
- Das Embedding wird im TTL-Cache mitgehalten: kommt Feedback, ist der Vektor schon da und muss nicht neu berechnet werden (Vektor gilt nur solange das Embedding-Model gleich bleibt).

## 7. Feedback-Mechanismus

Ablauf `feedback(requestId, feedback)`:

1. Auflösung der `requestId` im **TTL-Cache** (§6). Ist sie abgelaufen/unbekannt (Feedback kam zu spät oder nach Microservice-Neustart) → No-Op mit klarem Ergebnis (`FeedbackResult.Expired`), kein Fehler.
2. **Persistieren als gelabeltes Trainingsbeispiel:** Der aufgelöste Text (+ ggf. Embedding) wird zusammen mit den korrigierten Labels dauerhaft in `ai_labeled_sample` geschrieben. **Das ist der einzige Weg, wie reguläre Chatnachrichten in die DB gelangen.**
3. **Override-Cache (in-JVM, flüchtig):** Zusätzlich wird ein Override in einen **Caffeine**-Cache geschrieben (`maximumSize` + `expireAfterWrite`, Key: normalisierter Text / Embedding-Nähe). Identische/sehr ähnliche Nachricht → sofort korrigierter Score. Wird beim nächsten Retrain überflüssig.

Alle In-JVM-Caches im Projekt werden über **Caffeine** realisiert (Muster: `ExampleCache` im Microservice).

Semantik: Feedback beschreibt das **korrekte Label**, unabhängig vom angezeigten Confidence-Wert. `FalseNegative(SEXUAL)` = „SEXUAL ist korrekt/soll hoch sein" (hebt einen zu niedrigen Score an); `FalsePositive(SEXUAL)` = „SEXUAL ist falsch" (senkt ihn). Jedes gelabelte Beispiel bekommt einen **Quarantäne-Status** und ein Gewicht; es fließt beim nächsten Retrain als Trainingssignal ein.

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
  1. Lädt Seed-Korpus (`ai_seed_sample`) + gelabelte Feedback-Samples (`ai_labeled_sample`, nicht in Quarantäne) aus Postgres (direkter Treiber).
  2. Erzeugt Embeddings mit **identischem** frozen Model wie die JVM.
  3. Trainiert MLP-Kopf (Multi-Label, `BCEWithLogitsLoss`, ggf. Klassen-Gewichte gegen Imbalance).
  4. Evaluiert auf festem Holdout-Set → Metriken.
  5. Exportiert Kopf (ONNX oder simples Gewichts-Format) + Label-Config → **S3/MinIO** als `head-v{n}`.
  6. Schreibt `ai_model_version`-Zeile; promotet gemäß Regressions-Guard.
- Der Trainer liest Postgres direkt; **DB muss Postgres sein** (Bestätigung: `dev.slne.surf.database` läuft auf Postgres).

## 10. Persistenz (DB-Schema)

Alle Tabellen via `dev.slne.surf.database`, Exposed, `AuditableLongIdTable` (createdAt/updatedAt gratis), Repos mit `suspendTransaction`.

**Wichtig: es gibt KEINE `ai_request`-Tabelle.** Rohe Chatnachrichten werden nie persistiert — sie leben nur im flüchtigen TTL-Cache im RAM des Microservice (§6). In die DB gelangt eine Nachricht ausschließlich, wenn Feedback sie labelt.

- **`ai_labeled_sample`** — die einzige Trainingsquelle aus Live-Traffic: `text`, `embedding` (optional, Array/BLOB), `labels` (Set<AiCategory>), `feedback_type`, `source`, `quarantine_status`, `weight`, `origin_model_version`. Entsteht ausschließlich durch `feedback()`.
- **`ai_model_version`** — `version`, `s3_key`, `embedding_model_id`, `metrics_json`, `training_data_hash`, `is_active`.
- **`ai_seed_sample`** — kuratierter Grund-Korpus: `text`, `labels` (Set<AiCategory>), `language`, `source`.

Grober Storage-Umfang: Seed-Korpus (hunderte–tausende Zeilen) + gelabelte Feedback-Samples (wächst nur so schnell wie Mods tatsächlich Feedback geben — Größenordnung Zeilen/Tag, nicht Millionen). Vollkommen unkritisch.

## 11. Seed-Datensatz

Kuratierter, gelabelter Multi-Label-Korpus **DE + EN** aus drei Quellen:

1. **Öffentliche Toxizitäts-Datensätze** für das allgemeine DE/EN-Grundsignal (Beleidigung/Hass/Bedrohung): u.a. Jigsaw Toxic Comment (EN), Jigsaw Multilingual, **GermEval** & **HASOC** (deutsche Offensive/Hate-Datensätze), HateCheck. Lizenzen prüfen und dokumentieren; auf unsere 6 Kategorien mappen.
2. **Kuratierte Wort-/Phrasenlisten** (DE+EN Schimpfwörter/Slurs) für offensichtliche Fälle pro Kategorie.
3. **Minecraft-Kontext (nicht öffentlich → selbst erzeugt):** hand-geschriebene + **LLM-gestützt synthetisch generierte** Beispiele, human-reviewed:
   - **Hard Negatives** — Gaming-Phrasen, die out-of-context toxisch klingen, aber benigne sind: „geh hinten auf den berg und spring runter", „ich bomb deine base weg", „geh sterben im pvp", „kill the ender dragon".
   - **Positives im Gaming-Slang** — echte Toxizität in Gaming-Sprache.

- Klein starten (einige hundert–tausend Beispiele reicht für den Kopf), wächst dann via echtes Feedback (`ai_labeled_sample`), das die synthetischen Daten mit der realen Server-Verteilung anreichert.
- Als versionierte Datei(en) im Repo (`surf-ai-trainer/seed/`), beim ersten Bootstrap in `ai_seed_sample` geladen. Generierungs-/Kurations-Skripte liegen dabei.

## 12. Deployment (Docker / Coolify)

- **`surf-ai-microservice`** → Docker-Image, Coolify-Service. Env/Config: Postgres-DSN, RabbitMQ, S3/MinIO-Creds.
- **`surf-ai-trainer`** → Docker-Image, Coolify-Service (long-running). Env: Postgres-DSN, S3/MinIO-Creds. `/health`-Healthcheck.
- **`docker-compose.yml`** im Repo-Root:
  - Services: `microservice`, `trainer`, plus Dev-Abhängigkeiten (**Postgres**, RabbitMQ, **MinIO**) für lokales Testen.
  - Von Coolify direkt konsumierbar; lokal `docker compose up` als Dev-Umgebung.
- **Dockerfiles**: JVM-Image (Gradle-Build → Runtime-JRE) + Python-Image (slim + ML-Deps).
- **Lokales Test-Setup:** Bereits **zu Beginn der Implementierung** wird eine (dev-)`docker-compose` mit **Postgres + MinIO** hochgezogen (Docker Desktop ist beim Nutzer aktiv) und für Integrationstests genutzt — Prod nutzt später MinIO/Postgres aus Coolify. Verbindungsdaten kommen aus Env/Config.

## 13. Testing-Strategie

- **Unit (Kotlin):** DTO-Serialisierung (RabbitMQ-kompatibel), Override-Cache-Logik, Snapshot-Pointer/Rollback, Batching-Executor, Repos.
- **Golden-Set-Test:** Fester Satz erwarteter Klassifikationen als Regressions-Gate, inkl. der Beispiele:
  - `"kys"` → `SELF_HARM` hoch
  - `"geh dich doch einfach umbringen"` → `SELF_HARM` hoch
  - `"du hurensohn"` → `HARASSMENT` hoch
  - `"geh doch einfach hinten auf den berg und spring runter"` → alle niedrig
  - `"Penis"` → `SEXUAL` erhöht (soll geflaggt werden)
- **Trainer (Python):** Tests für Export-Format, Metrik-Guard, Holdout-Eval, S3-Roundtrip (gegen MinIO).
- **Integrationstest:** Microservice ↔ Trainer Retrain-Zyklus (Trigger → neue Version → Hot-Reload → geänderte Prediction).

## 14. Nicht-Ziele (out of scope)

- Die eigentliche Moderationsaktion (Mute/Ban/Warn) — macht ein separates Plugin.
- SPAM/Werbung-Erkennung — besser per Regex/Rate-Limit im Moderations-Plugin.
- UI/Dashboard fürs Feedback (nur API/RPC).

## 15. Offene Punkte / benötigt vom Nutzer

- ✅ **DB = Postgres** (bestätigt). Trainer nutzt direkten Postgres-Treiber.
- ✅ **MinIO** kommt in Coolify (Prod). Fürs lokale Testen zieht die Implementierung selbst Postgres + MinIO per docker-compose hoch (Docker Desktop aktiv).
- **Prod-Credentials** (Coolify): Postgres-DSN, MinIO-Endpoint/Bucket/Keys, RabbitMQ — werden erst beim Deployment gebraucht, via Env/Config.
- **RabbitMQ-Serialisierungsformat** des RPC-Layers — in der Implementierungsphase am bestehenden `ExampleRpcService`-Muster verifizieren, damit die DTOs korrekt über die Leitung gehen.

## 16. Betriebsregeln für Folge-Agents

- Es wird eine `CLAUDE.md` im Repo-Root angelegt mit Projektkontext, Architektur, Konventionen — **und der expliziten Regel: keine Subagents verwenden** (zu hoher Token-Verbrauch). Pläne werden via `executing-plans` sequenziell abgearbeitet.
