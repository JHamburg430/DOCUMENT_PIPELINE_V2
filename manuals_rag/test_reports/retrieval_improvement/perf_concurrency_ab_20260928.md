# Qdrant and agent concurrency acceptance — 2026-09-28

## Scope

- Source: `7345a3ddde00d337e06af2c4c47c93208c891096`
  (`perf/qdrant-agent-parallelism-20260927`), clean in every accepted run.
- Dataset SHA-256:
  `96d8f64d051391d1308286c44369e90fd8174c045c80f4696cae07d86f10fb22`.
- Slice: offset 29, limit 10, with the same ordered case keys in all runs.
- Qdrant collection: `manuals_manuals_vendor_keyence`, 422,606 points.
- Verified keyword payload indexes: `source_document_id`, `chunk_type`,
  `document_version_id`, `version_signal`, and `keywords`.
- Qdrant status and optimizer status were both healthy after the comparison.

## Accepted results

| Mode | Case concurrency | Backend concurrency | Wall time | Cases/hour | LangGraph full-pass | LlamaIndex full-pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial reference | 1 | 1 | 383.585 s | 93.85 | 10/10 | 10/10 |
| Case parallel | 2 | 1 | 247.885 s | 145.23 | 10/10 | 10/10 |
| Backend parallel | 1 | 2 | 253.763 s | 141.86 | 10/10 | 10/10 |

Case concurrency 2 reduced wall time by 35.38% (1.547x throughput) and
backend concurrency 2 reduced it by 33.84% (1.512x throughput). Case
parallelism was 2.32% faster than backend parallelism on this slice.

Every accepted artifact has exit status 0, 10 unique ordered rows, and zero
non-pass agent-matrix cells.

## Artifacts

- `perf_ab_serial10_r3_20260928T1830-7345a3d.json`
  - SHA-256: `8dbc4eabf4a455b7c9980afc6ff6b6ad3341762c7193a9905737978ac1807f8f`
- `perf_ab_case2_20260928T1837-7345a3d.json`
  - SHA-256: `b7b70e9075fea995c13704ee28eac15fd114ce28574264f3215b75fd5d948af5`
- `perf_ab_backend2_20260928T1842-7345a3d.json`
  - SHA-256: `f54493f895cef65f3500f814b01259be8bd6c7cae20344bfaa2c70d809601226`

Two earlier serial attempts exited 143 because their evaluator containers were
bound to the five-minute tool launcher lifetime. They are partial, invalid, and
excluded. The accepted runs used transient user-systemd units and completed
with final artifacts.

## Integration and verification

- Performance implementation patch IDs match between
  `c46f13d` and release commit `0b5cd3a`.
- Concurrency guard patch IDs match between
  `7345a3d` and release commit `6ab55b2`.
- Optimization-focused regression suite: 510 passed.
- A broader focused suite reached 819 passes; its only two failures were
  unrelated missing-PDF-fixture errors.

## Decision

Use case concurrency 2 and backend concurrency 1 as the standard evaluation
mode. Keep backend concurrency 2 available for targeted experiments, but do
not stack it with case concurrency 2. Keep the Qdrant request limit at 2 and
retain all five payload indexes. Qdrant GPU indexing is not enabled because
the active workload is steady-state querying rather than index construction.
