# Metadata extraction model comparison: Qwen3.5 9B vs 27B

Date: 2026-09-10

## Method

- Dry-run only; PostgreSQL, Qdrant, and live corpus metadata were not modified.
- Both models used the same metadata code, prompts, source segments, 12,000-character batching, and five difficult documents.
- Execution used the RTX 3090 through its dedicated Ollama worker.
- The set covered a short datasheet, a problematic cover title, grouped identifiers, a version-heavy PLC catalog, and a long vision-system catalog.

## Aggregate results

| Metric | Qwen3.5 9B | Qwen3.5 27B |
|---|---:|---:|
| Documents completed | 5/5 | 4/5 |
| Automated audit passes | 3/5 | 3/5 |
| Total elapsed time | 443.6 s | 1,900.8 s |
| Relative elapsed time | 1.00x | 4.29x |
| Peak RTX 3090 allocation | about 9.3 GiB | about 23.2 GiB |

## Per-document review

### VJ-H500CX datasheet

- Both models selected `VJ-H500CX`, generated the expected alias, and passed all gates.
- 27B took 91.8 seconds versus 39.1 seconds for 9B.
- No meaningful accuracy gain from 27B.

### LJ-S8000 Easy Configuration Manual

- Both models recovered the grounded opening-page title and `LJ-S8000` identity.
- 9B produced the conservative routing set `LJ-S8000`.
- 27B also routed individual LJ-S heads and the `LJ-S` family. These are grounded but broaden hard-routing behavior and caused the benchmark's strict evidence-type gate to fail.
- 27B found a plausible manufacturer footer; 9B left manufacturer unknown.

### KV-X catalog

- 9B completed in 143.4 seconds but missed the expected `SV2` identity and expected software-version evidence.
- 27B failed the document after 886.1 seconds. Its version-applicability responses repeatedly omitted required `value` and `kind` fields, exhausting retries and recursive splitting.
- This is the clearest evidence against using 27B as either the default or a general fallback for schema-v2 metadata.

### SR-1000/SR-2000/SR-PN1 PROFINET guide

- Both models retained the three expected product identifiers.
- 27B chose the more specific printed title, `Connection Guide: PROFINET Communication`, and explicitly routed PROFINET.
- 9B chose the broader cover heading, `Auto-ID Technical Guide`.
- 27B incorrectly promoted footer/document code `D65GB` to a part number. Its apparent automated-gate win therefore was not a clean human-audit win.

### CV-X catalog

- Both models completed and passed the generic automated gates.
- 9B retained `CV-X482`, a known retrieval target, while 27B omitted it and returned a different controller subset.
- 27B extracted more accessories and EtherCAT, but required 780.9 seconds versus 174.2 seconds and generated malformed JSON that triggered recursive splitting.
- Neither result demonstrates a decisive overall completeness advantage.

## Decision

Keep Qwen3.5 9B as the default metadata model. Do not use Qwen3.5 27B as a global fallback: it was 4.29 times slower, consumed nearly the full RTX 3090, failed one of five documents, and did not produce a consistent accuracy improvement.

Prefer deterministic validation, document-code rejection, targeted completeness checks, and selective field-specific retrying with 9B. A larger model should only be reconsidered if a future adjudicated evaluation shows a material gain on a narrowly defined field.

## Artifacts

- `metadata_model_comparison_qwen3.5-9b_20260910.json`
- `metadata_model_comparison_qwen3.5-27b_20260910.json`
- `scripts/benchmark/compare_metadata_models.py`
