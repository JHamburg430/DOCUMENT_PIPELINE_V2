# Independent matrix audit: `agent_matrix_full48_b20f9a3_20260920_1200`

## Decision

The artifact is structurally accepted for diagnostic adjudication.
It is **not** accepted for production enablement.

## Reconciliation

- Cases: `48`
- Dataset SHA-256: `fbff861ca1394d281cba9ce63dff090e4024568f72fca1cf0fd46e7fac04d124`
- Source: `{"branch": "release/manuals-rag-agentic-disabled-20260919", "dirty": false, "dirty_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "dirty_status": [], "revision": "b20f9a3", "untracked_paths": []}`

## langgraph

- Full matrix passes: `30/48`
- Required evidence groups: `73`
- Stage retention: `{"dense": {"observed_groups": 14, "retained_groups": 9}, "final_context": {"observed_groups": 14, "retained_groups": 9}, "fusion": {"observed_groups": 14, "retained_groups": 9}, "rerank": {"observed_groups": 14, "retained_groups": 6}}`
- Judge states: `{"deterministic:checked": 62, "llm:checked": 12, "llm:unchecked": 1}`
- Failure classes: `{"answer_or_answer_scoring": 2, "candidate_anchor_or_scoring": 8, "dense_retrieval_miss": 3, "evidence_verification": 2, "pass": 30, "reranker_loss": 2, "verifier_contract_failure": 1}`

### Failed cases

- `37c2ce23-e5ac-5c60-836f-84a4707b925d::curated4` — `answer_or_answer_scoring`; cells=grounded_answer; stop=`sufficient`
- `0d116180-22a3-511a-b00b-9fb8c4ea796a::contextual_multi_step::1` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `02b921f5-082d-5b31-8565-8d22f5023169::contextual_multi_step::4` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`plan_exhausted`
- `02b921f5-082d-5b31-8565-8d22f5023169::contextual_multi_step::5` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `ece874eb-6b99-52c5-9a83-d2bcce1d9a75::contextual_multi_step::6` — `answer_or_answer_scoring`; cells=grounded_answer; stop=`sufficient`
- `ece874eb-6b99-52c5-9a83-d2bcce1d9a75::contextual_multi_step::7` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `3a9570f2-e15b-5214-82d4-c70ef88c0499::contextual_multi_step::8` — `dense_retrieval_miss`; cells=evidence_sufficiency,grounded_answer,latency_token_cost; stop=`plan_exhausted`
- `b2e9c647-ba17-55e6-afc6-ee778d11a9c9::contextual_multi_step::9` — `evidence_verification`; cells=evidence_sufficiency,grounded_answer,latency_token_cost; stop=`plan_exhausted`
- `curated_cross_document_v2::1` — `dense_retrieval_miss`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `curated_cross_document_v2::2` — `verifier_contract_failure`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`runtime_budget_exhausted`
- `curated_cross_document_v2::3` — `evidence_verification`; cells=evidence_sufficiency,grounded_answer; stop=`plan_exhausted`
- `curated_cross_document_v2::4` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`plan_exhausted`
- `curated_cross_document_v2::5` — `dense_retrieval_miss`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `curated_cross_document_v2::6` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`plan_exhausted`
- `curated_cross_document_v2::7` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`runtime_budget_exhausted`
- `curated_cross_document_v2::8` — `reranker_loss`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`plan_exhausted`
- `curated_cross_document_v2::9` — `reranker_loss`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`plan_exhausted`
- `curated_cross_document_v2::10` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`plan_exhausted`

## llamaindex

- Full matrix passes: `30/48`
- Required evidence groups: `73`
- Stage retention: `{"dense": {"observed_groups": 15, "retained_groups": 10}, "final_context": {"observed_groups": 15, "retained_groups": 10}, "fusion": {"observed_groups": 15, "retained_groups": 10}, "rerank": {"observed_groups": 15, "retained_groups": 7}}`
- Judge states: `{"deterministic:checked": 62, "llm:checked": 14, "llm:unchecked": 2}`
- Failure classes: `{"answer_or_answer_scoring": 2, "candidate_anchor_or_scoring": 6, "dense_retrieval_miss": 3, "evidence_verification": 3, "pass": 30, "reranker_loss": 2, "verifier_contract_failure": 2}`

### Failed cases

- `37c2ce23-e5ac-5c60-836f-84a4707b925d::curated4` — `answer_or_answer_scoring`; cells=grounded_answer; stop=`sufficient`
- `0d116180-22a3-511a-b00b-9fb8c4ea796a::contextual_multi_step::2` — `evidence_verification`; cells=evidence_sufficiency,grounded_answer,latency_token_cost; stop=`runtime_budget_exhausted`
- `02b921f5-082d-5b31-8565-8d22f5023169::contextual_multi_step::4` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`
- `02b921f5-082d-5b31-8565-8d22f5023169::contextual_multi_step::5` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `ece874eb-6b99-52c5-9a83-d2bcce1d9a75::contextual_multi_step::6` — `answer_or_answer_scoring`; cells=grounded_answer; stop=`sufficient`
- `ece874eb-6b99-52c5-9a83-d2bcce1d9a75::contextual_multi_step::7` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `3a9570f2-e15b-5214-82d4-c70ef88c0499::contextual_multi_step::8` — `evidence_verification`; cells=evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`
- `b2e9c647-ba17-55e6-afc6-ee778d11a9c9::contextual_multi_step::9` — `dense_retrieval_miss`; cells=evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`
- `curated_cross_document_v2::1` — `dense_retrieval_miss`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`sufficient`
- `curated_cross_document_v2::2` — `verifier_contract_failure`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`
- `curated_cross_document_v2::3` — `evidence_verification`; cells=evidence_sufficiency,grounded_answer; stop=`subquestions_exhausted`
- `curated_cross_document_v2::4` — `verifier_contract_failure`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`
- `curated_cross_document_v2::5` — `dense_retrieval_miss`; cells=candidate_recall,document_retention,evidence_sufficiency,grounded_answer; stop=`subquestions_exhausted`
- `curated_cross_document_v2::6` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`subquestions_exhausted`
- `curated_cross_document_v2::7` — `candidate_anchor_or_scoring`; cells=candidate_recall,document_retention,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`
- `curated_cross_document_v2::8` — `reranker_loss`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`
- `curated_cross_document_v2::9` — `reranker_loss`; cells=candidate_recall,evidence_sufficiency,grounded_answer; stop=`subquestions_exhausted`
- `curated_cross_document_v2::10` — `candidate_anchor_or_scoring`; cells=candidate_recall,evidence_sufficiency,grounded_answer,latency_token_cost; stop=`subquestions_exhausted`

## Open production blockers

- source-backed adjudication of failed answers and citations is incomplete
- a separate frozen held-out bank has not passed
- human visual-PDF adjudication is incomplete
