# Agent evidence gate (staged)

The conversational Agent can make several `manuals_rag` calls. Earlier, its
finalizer saw their answer prose and pooled citation IDs but not the source
passages. This allowed a rephrased claim to inherit a citation without proving
that the cited passage supported that claim.

`AGENT_EVIDENCE_GATE_ENABLED` (default `false`) adds a fail-closed check to
`/agent/stream` for both LangGraph and LlamaIndex. The API carries bounded
retrieved passage text internally with each successful RAG tool answer. Before
completion, the gate requires:

1. Every final citation to match a cited retrieved chunk/document pair from a
   successful tool call. A missing or invented quote is rejected.
2. Explicit requested model identifiers to occur in the cited passage, title,
   or product-model metadata; numeric role/value and per-unit count relations
   to agree with the cited passages.
3. A separate local verifier to find verbatim supporting quotes and check both
   factual support and coverage of every requested answer part. A malformed,
   unavailable, or negative verdict causes an abstention, never a reconstructed
   citation.

The gate does not fix retrieval misses, OCR, or stream errors. Its semantic
verifier is not infallible: a live case-018 replay initially accepted a count
from a table row for the wrong XG-X model. The deterministic model-scope check
was added afterward. Keep the flag off for production until a clean frozen
evaluation confirms no unsupported accepted answers and acceptable answer
coverage for both backends. Preserve failed run artifacts; do not relabel an
abstention as a correct answer.

Focused regression: `tests/unit/test_evidence_gate.py`. Live canary should
exercise a supported answer, wrong number, omitted requested part, wrong table
model, and both Agent backends through `/agent/stream`. Then validate rendered
Agent UI behavior, the full frozen bank, stream errors, latency, and rollback
before enabling the flag in the API environment.
