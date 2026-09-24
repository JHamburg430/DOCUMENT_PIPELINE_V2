# Manuals RAG metadata rollout gate — 2026-09-24

## Decision

**FAIL CLOSED.** Do not apply the corpus-wide metadata report and do not enqueue a corpus-wide embedding refresh.

The visual metadata pilot passed, but repeated extraction of the same LR-T source did not produce a stable evidence ledger. Temperature-zero sampling, stable per-purpose retry seeds, and serialized metadata map calls reduced but did not eliminate variation. The latest comparable pair contained 91 and 97 claims with 80 canonical claims in common. This is not acceptable for a durable production routing contract.

No v6 metadata was persisted to PostgreSQL. No corpus-wide embedding jobs were queued. Agentic retrieval remains disabled.

## Accepted evidence

- Visual metadata cohort: `visual_document_understanding_cohort7_final_v1_20260924.json`
  - 7 documents
  - 154/154 metadata claims visually supported
  - QA 40/57; one document lacked a verified QA answer on the selected diagram page
- CA-S20D visual pilot: `visual_document_understanding_ca_s20d_pilot_v1_20260924.json`
  - 15/15 metadata claims supported
  - QA 4/7
- Aggregate metadata visual gate: 8/8 documents, 169/169 metadata claims
- Focused deterministic-client/parser/backfill tests: 122 passed
- Broader relevant metadata/Ollama/Qdrant suite: 133 passed

The visual result supports metadata precision on the adjudicated sample. It does not override the repeatability failure or establish answer-quality readiness.

## Rejected extraction artifacts

The following are diagnostic only and must never be supplied to `--apply-planned-report`:

- `document_metadata_backfill_20260924_071233.invalid_mixed_scope.json`
- `document_metadata_backfill_20260924_082012.json`
  - interrupted production-only v5 inventory
  - 16/53 documents checkpointed: 14 planned, 1 already current, 1 OCR-blocked
  - SHA-256 `582e7a3310adb2e80ff5f9819b185303c627960460f21dee46679c2f97c03fe6`
- Same-document v6 repeatability trials:
  - `document_metadata_backfill_20260924_084604.json` — `0863d14b047d8443b8814f958eb02746cabdc8d5ac45e57bb9f6d5c470ac203d`
  - `document_metadata_backfill_20260924_085029.json` — `e6676a07504f9b7657bfda3196a5b767b637820c4d83d8a535941dc557e5810f`
  - `document_metadata_backfill_20260924_085541.json` — `bffc4568034d2f01984e62f3e7a6b92b81a581da7f21c12c6b82e74b18732c45`
  - `document_metadata_backfill_20260924_090129.json` — `fd348ec6f16b1645773c4710022141ff5155ed0e3264f2c3f9111506475da816`
  - `document_metadata_backfill_20260924_090814.json` — `69550c22b15f5426e84eace0db9351488be020c9ce0cf534fcfce87eb1fca0f7`
  - `document_metadata_backfill_20260924_091406.json` — `562a17046449d70ddc6fea6a34d040da66a50baac21efe46c724f0f92706372b`
  - `document_metadata_backfill_20260924_092740.json` — `6af6f301092dcf3e9aa08280a82000a4710d541bcd90139996095968eceb0f5a`
  - `document_metadata_backfill_20260924_093317.json` — `2e06df126b40064604ab8c52953aa13c7c1d6f62158abfa61c3d1f184f2f8c22`

The tests also exposed and rejected invalid model shapes: `+`-joined option combinations as product families, slash-compressed model lists as standalone models, and accessory/specification tokens (`OP-*`, `COM*`, bare `M*`, `IP*`, `SUS*`) as devices.

## Rollback bundle

Verified pre-mutation backup:

`/home/john/Backups/manuals-rag/metadata-rollout-20260924T0835/manifest.json`

It contains a validated PostgreSQL custom dump and Qdrant snapshots for dense, BM25, and document metadata collections. The corpus metadata rollout did not use the backup because the acceptance gate failed before persistence.

## PostgreSQL/Qdrant reconciliation

Production corpus `manuals_vendor_keyence`:

- source documents: 53
- current document versions: 53
- PostgreSQL active chunks: 384,288
- Qdrant dense points: 384,288, green
- Qdrant BM25 points: 384,288, green
- Qdrant document selectors: 53, green
- exact chunk-ID sets: reconciled

The first full read-only audit found one pre-existing stale-index document, VJ-H500CX (`8d83cb71-55d2-42db-9630-bc16f0d6877c`): PostgreSQL and BM25 carried v1 routing metadata, while dense payloads and the selector were stale. Only that current document version was refreshed in ingestion run `f2dd95c3-5079-4a27-b13d-4f6953a1f624`; it reached `completed`.

Targeted post-refresh audit:

- PostgreSQL chunks: 49
- dense points: 49
- BM25 points: 49
- selectors: 1
- chunk IDs: match
- routing scope fields: match
- pipeline payloads: match
- selector document/version/pipeline: match

Because the initial full audit passed the other 52 documents and the only failing document passed the targeted re-audit, the existing persisted corpus is reconciled after the isolated refresh.

## Remaining blockers

1. Define a deterministic consensus or source-native candidate ledger for model-extracted long-tail claims; seeds and serialized generation are insufficient.
2. Re-run and visually adjudicate the v6 representative cohort after that contract is implemented.
3. Re-run a clean 53-document production-only dry run; the existing reports are rejected.
4. Repair/reingest the XG-X Ver. 3.5 source with full OCR (the exact five-page RapidOCR pilot passed, but the 1,322-page source remains unreadable in production).
5. Only then take a fresh immutable report hash, apply that exact report, refresh affected embeddings, audit PostgreSQL/Qdrant, and prove idempotent re-entry.

Answer-quality and agentic-production gates remain separate and closed.
