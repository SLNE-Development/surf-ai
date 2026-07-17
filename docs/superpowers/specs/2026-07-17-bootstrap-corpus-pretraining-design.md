# surf-ai-trainer — Bootstrap Corpus Pretraining — Design / Spec

**Date:** 2026-07-17
**Status:** Approved (brainstorming complete), ready for implementation plan
**Related:** [2026-07-17-surf-ai-moderation-design.md](2026-07-17-surf-ai-moderation-design.md)

## 1. Goal & context

Today `surf-ai-trainer` trains `HeadNet` (a small MLP classification head on top of the
frozen `multilingual-e5-small` embedder) from **259 hand-written seed samples**
(`surf-ai-trainer/seed/*.jsonl`) plus whatever moderator feedback has accumulated in
`ai_labeled_sample`. This is enough to bootstrap the system but leaves the head with very
little signal to generalize from.

This project adds a **one-time offline bootstrap step** that trains an initial head
checkpoint from a much larger (~5-10M row) public toxicity/hate-speech corpus, then
warm-starts the existing nightly/on-demand retrain (`run_retrain`) from that checkpoint
instead of from random initialization. The frozen embedding model itself is **not**
retrained — `multilingual-e5-small` is already pretrained on far more data than we could
gather here, and retraining it would break the Kotlin/Python embedding-space contract
(`EmbeddingModelFixtureTest`).

## 2. Data sourcing & label mapping

New script: `surf-ai-trainer/scripts/build_bootstrap_corpus.py`.

Public corpora, each with a documented, auditable mapping onto our 6 wire-contract
categories (`app/categories.py`), using a binary presence/absence threshold (e.g. score >
0.5) on each source's continuous label columns:

| Source | Rows (approx) | Language | Mapped category rule |
|---|---|---|---|
| Jigsaw Toxic Comment Classification | ~160k | EN | `threat`→THREAT, `obscene`→SEXUAL, `identity_hate`→HATE_SPEECH, `insult`/`toxic`/`severe_toxic`→HARASSMENT |
| Jigsaw Unintended Bias / Civil Comments | ~2M | EN | `threat`→THREAT, `sexual_explicit`/`obscene`→SEXUAL, `identity_attack`→HATE_SPEECH, `insult`/`toxicity`→HARASSMENT |
| Reddit SuicideWatch (or comparable self-harm/suicidal-ideation corpus) | ~200k | EN | positive class →SELF_HARM |
| GermEval 2018/2019 offensive-language | ~10k | DE | offensive/abusive subtype →HARASSMENT or HATE_SPEECH depending on subtype label |

Each row is normalized to the same shape as `seed/*.jsonl`: `{text, labels, language,
source}`.

**Known gap — CHILD_SAFETY:** there is no responsible public dataset for this category.
It is intentionally left uncovered by the bootstrap corpus; the model's CHILD_SAFETY
signal continues to come only from the hand-written seed rows and real moderator
feedback. This is a deliberate scope boundary, not an oversight.

Output is chunked JSONL written to a **gitignored** local directory (too large to
commit — same treatment as the ~490MB embedding fixture already excluded from the repo).
The trainer README documents how to regenerate it.

## 3. Offline embedding + caching

New script: `surf-ai-trainer/scripts/embed_bootstrap_corpus.py`.

- Runs the exact same preprocessing as `app/embedding.py` (`"query: " + text` → tokenize
  → embed → mean-pool by attention mask → L2-normalize), batched (e.g. 512–2048 rows per
  batch).
- Uses `onnxruntime-gpu`'s `CUDAExecutionProvider` when available, falling back to CPU.
- Writes embeddings + label vectors to disk as chunked `.npz` files (not one
  in-memory array) so the ~5-10M × 384 float32 embeddings (roughly 7-15GB) are never
  fully materialized in RAM at once.
- Resumable: skips chunks whose output file already exists, since a multi-hour job may
  be interrupted.
- One-time, manual, network/GPU-dependent — documented in the README next to the
  existing embedding-fixture-export instructions. Not run by `pytest` or CI, matching how
  the `slow`-marked embedding download is already excluded from the default test run.

## 4. Base-checkpoint pretraining

New module: `surf-ai-trainer/app/pretrain.py`.

- `pretrain_head(embedding_chunks_dir, epochs, batch_size, lr) -> HeadNet`: mini-batch
  training loop (`DataLoader` iterating the cached `.npz` chunks) using the same
  `HeadNet` architecture and `BCEWithLogitsLoss(pos_weight=...)` scheme as today's
  `train_head`, adapted from a single full-batch pass to batched iteration since the
  corpus no longer fits comfortably as one training step.
- Saves the resulting weights as a plain **torch `state_dict`** (not ONNX — ONNX is a
  serving format the JVM microservice loads; training needs the raw torch weights to
  warm-start from) to S3 at a fixed key: `models/head/base-checkpoint.pt`.
- One-time manual run (e.g. `python -m app.pretrain`), not part of the nightly cycle.

## 5. Wiring the base checkpoint into the existing retrain loop

Minimal, additive changes so the existing fast path (seed + feedback only, run nightly)
is unaffected when no base checkpoint exists:

- `train_head(samples, embedder, epochs=100, lr=1e-2, init_state_dict=None)` gains an
  optional parameter. When given, `HeadNet` loads those weights before training instead
  of random init. When `None` (today's behavior, and every existing test), nothing
  changes.
- `run_retrain` (`app/pipeline.py`) attempts to download `models/head/base-checkpoint.pt`
  from S3; if present, it's passed through as `init_state_dict`. If absent (e.g. local
  dev where the bootstrap pretrain has never been run), `run_retrain` behaves exactly as
  it does today.
- No database schema change — the base checkpoint is an internal warm-start artifact at
  a fixed S3 key, not a served `ai_model_version` row.

## 6. Testing

Fast, no network/GPU required — run under the normal `pytest` (non-`slow`) suite:

- Label-mapping heuristic: pure function, tested against small fixture rows per source.
- `pretrain_head`: tiny synthetic embeddings/labels, asserts loss decreases and the
  returned state_dict loads cleanly into a fresh `HeadNet`.
- `train_head(..., init_state_dict=...)`: asserts the model actually warm-starts (e.g.
  loss immediately after loading the checkpoint, before any epochs, matches the
  checkpoint's own loss rather than a random-init loss).
- Existing `test_train.py` / `test_pipeline.py`: unchanged, since they never pass
  `init_state_dict`.

The corpus-build, GPU-embedding, and pretrain scripts themselves are one-time, manual,
network/GPU-dependent operations and are **not** exercised by `pytest` or CI, same as the
embedding-model fixture export.

## 7. Known limitations

- **CHILD_SAFETY** gets no benefit from this corpus (see §2) — remains dependent on seed
  + feedback only.
- **Domain mismatch:** the bootstrap corpus is news-comment / Reddit text, not Minecraft
  chat. Its value is giving the head a broader starting point to warm-start from, not
  in-domain accuracy — the small in-domain seed+feedback set, trained after the warm
  start, is still what determines real prediction quality.
