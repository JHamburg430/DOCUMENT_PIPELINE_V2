from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api import main


client = TestClient(app)
ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


class _FakeWorkflow:
    def invoke(self, _payload):
        return {
            "query": "What is the voltage?",
            "filters": {"source_document_id": "doc-1"},
            "analysis": {"query_types": ["spec_lookup"], "preferred_chunk_types": ["spec_record"]},
            "step_timings_ms": {
                "run_dense_search": 12.5,
                "run_sparse_search": 4.2,
                "run_special_search": 0.8,
                "fuse_results": 1.1,
                "rerank_results": 2.3,
                "assemble_context": 0.2,
            },
            "dense_results": [
                {
                    "chunk_id": "chunk-1",
                    "score": 0.91,
                    "title": "Doc",
                    "document_version_id": "ver-1",
                    "source_document_id": "doc-1",
                    "pages": [1],
                    "section_path": ["Specs"],
                    "content": "Voltage: 24 V",
                    "metadata": {"chunk_type": "spec_record", "retrieval_stage": "dense"},
                }
            ],
            "sparse_results": [],
            "special_results": [],
            "fused_results": [],
            "retrieval_results": [
                {
                    "chunk_id": "chunk-1",
                    "score": 0.93,
                    "title": "Doc",
                    "document_version_id": "ver-1",
                    "source_document_id": "doc-1",
                    "pages": [1],
                    "section_path": ["Specs"],
                    "content": "Voltage: 24 V",
                    "metadata": {
                        "chunk_type": "spec_record",
                        "retrieval_stage": "reranked",
                        "context_window": "Specs\n\nVoltage: 24 V",
                    },
                }
            ],
        }


def _fake_debug_snapshot():
    return {
        "analysis": {"query_types": ["spec_lookup"]},
        "step_timings_ms": {
            "run_dense_search": 12.5,
            "run_sparse_search": 4.2,
            "run_special_search": 0.8,
            "fuse_results": 1.1,
            "rerank_results": 2.3,
            "assemble_context": 0.2,
            "judge_answer_inputs": 5.0,
            "summarize_answer_inputs": 6.0,
            "generate_answer": 7.0,
        },
        "stages": [
            {"name": "dense_results", "count": 1, "samples": [{"chunk_type": "spec_record"}], "duration_ms": 12.5},
            {"name": "sparse_results", "count": 0, "samples": [], "duration_ms": 4.2},
            {"name": "special_results", "count": 0, "samples": [], "duration_ms": 0.8},
            {"name": "fused_results", "count": 1, "samples": [], "duration_ms": 1.1},
            {"name": "retrieval_results", "count": 1, "samples": [], "duration_ms": 2.5},
        ],
        "answer_generation_inputs": {
            "count": 1,
            "samples": [{"chunk_type": "spec_record", "relevance_verdict": "relevant"}],
            "duration_ms": 5.0,
        },
        "answer_summaries": {
            "count": 1,
            "samples": [{"summary": "Voltage is 24 V."}],
            "duration_ms": 6.0,
        },
        "answer": {"answer": "Voltage is 24 V."},
        "answer_generation_trace": {
            "relevance_review": {"provider": "ollama", "model": "qwen3.5:4b"},
            "summarization": {"provider": "ollama", "model": "qwen3.5:4b"},
            "final_answer": {
                "provider": "ollama",
                "model": "qwen3.5:9b",
                "used_fallback": False,
                "answer_source": "model",
                "summarized_evidence": [{"summary": "Voltage is 24 V."}],
            },
        },
        "progress": {
            "current_step": "generate_answer",
            "current_label": "Generating final answer",
            "completed_steps": ["classify_query", "build_filters", "run_dense_search", "run_sparse_search", "run_special_search", "fuse_results", "rerank_results", "assemble_context", "judge_answer_inputs", "summarize_answer_inputs", "generate_answer"],
            "total_steps": 11,
            "step_sequence": [],
        },
    }


def test_debug_query_endpoint_returns_pipeline_snapshot(monkeypatch):
    monkeypatch.setattr(main, "execute_query_debug_run", lambda request, sample_limit=10: _fake_debug_snapshot())

    response = client.post(
        "/debug/query?sample_limit=5",
        headers=ADMIN_HEADERS,
        json={
            "query": "What is the voltage?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "filters": {"source_document_id": "doc-1"},
            "response_mode": "answer_with_citations",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["query_types"] == ["spec_lookup"]
    assert [stage["name"] for stage in payload["stages"]] == [
        "dense_results",
        "sparse_results",
        "special_results",
        "fused_results",
        "retrieval_results",
    ]
    assert payload["answer_generation_inputs"]["samples"][0]["chunk_type"] == "spec_record"
    assert payload["answer_generation_inputs"]["samples"][0]["relevance_verdict"] == "relevant"
    assert payload["answer_summaries"]["samples"][0]["summary"] == "Voltage is 24 V."
    assert payload["answer"]["answer"] == "Voltage is 24 V."
    assert payload["answer_generation_trace"]["relevance_review"]["model"] == "qwen3.5:4b"
    assert payload["answer_generation_trace"]["final_answer"]["model"] == "qwen3.5:9b"
    assert payload["answer_generation_trace"]["final_answer"]["summarized_evidence"][0]["summary"] == "Voltage is 24 V."


def test_debug_query_run_endpoints_return_polled_progress(monkeypatch):
    class ImmediateThread:
        def __init__(self, target=None, args=(), daemon=None):
            self.target = target
            self.args = args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(main, "Thread", ImmediateThread)
    monkeypatch.setattr(main, "execute_query_debug_run", lambda request, sample_limit=10, progress_callback=None: _fake_debug_snapshot())
    main.debug_query_runs.clear()

    created = client.post(
        "/debug/query-runs?sample_limit=5",
        headers=ADMIN_HEADERS,
        json={
            "query": "What is the voltage?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "filters": {"source_document_id": "doc-1"},
            "response_mode": "answer_with_citations",
        },
    )

    assert created.status_code == 200
    run_id = created.json()["run_id"]

    polled = client.get(f"/debug/query-runs/{run_id}", headers=ADMIN_HEADERS)
    assert polled.status_code == 200
    payload = polled.json()
    assert payload["status"] == "completed"
    assert payload["result"]["answer"]["answer"] == "Voltage is 24 V."
    assert payload["result"]["progress"]["current_step"] == "generate_answer"


def test_debug_documents_endpoint_returns_recent_listing(monkeypatch):
    captured = {}

    def fake_list_recent_documents(*, limit):
        captured["limit"] = limit
        return [{"document_id": "doc-1", "source_filename": "CA-EN100U_Datasheet.pdf"}]

    monkeypatch.setattr(main, "list_recent_documents", fake_list_recent_documents)

    response = client.get("/debug/documents?limit=5", headers=ADMIN_HEADERS)

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "doc-1"
    assert captured == {"limit": 5}


def test_debug_document_snapshot_endpoint_returns_extraction_samples(monkeypatch):
    def fake_fetch_one(_query, _params):
        return {
            "document_id": "doc-1",
            "corpus_id": "manuals_vendor_keyence",
            "title": "CA-EN100U Datasheet",
            "source_filename": "CA-EN100U_Datasheet.pdf",
            "manufacturer": "Keyence",
            "product_family": "CA",
            "product_model": "CA-EN100U",
            "document_kind": "datasheet",
            "ingest_status": "parsed",
            "version_id": "ver-1",
            "page_count": 2,
            "parse_profile": "deep_manual",
            "quality_score": 0.92,
            "parse_warnings": [],
            "docling_artifact_uri": "s3://manuals-artifacts/test.json",
            "ingested_at": "2026-04-08T00:00:00",
        }

    def fake_fetch_all(query, params):
        if "from logical_nodes" in query and "group by node_type" in query:
            return [{"node_type": "spec", "count": 3}, {"node_type": "paragraph", "count": 2}]
        if "from logical_nodes" in query:
            return [
                {
                    "id": "node-1",
                    "ordinal": 1,
                    "node_type": "spec",
                    "heading_text": None,
                    "section_path_json": ["Specs"],
                    "page_from": 1,
                    "page_to": 1,
                    "warning_level": None,
                    "procedure_step_number": None,
                    "spec_name": "Voltage",
                    "spec_value": "24",
                    "spec_unit": "V",
                    "keywords_json": ["24 V"],
                    "citability_score": 0.99,
                    "token_count": 3,
                    "table_json": None,
                    "text_raw": "Voltage: 24 V",
                    "text_normalized": "Voltage: 24 V",
                }
            ]
        if "from retrieval_chunks" in query and "group by chunk_type" in query:
            return [{"chunk_type": "spec_record", "count": 3}]
        if "from retrieval_chunks" in query:
            return [
                {
                    "id": "chunk-1",
                    "chunk_type": "spec_record",
                    "chunk_level": 1,
                    "title": "CA-EN100U Datasheet",
                    "section_path_text": "Specs",
                    "page_from": 1,
                    "page_to": 1,
                    "priority_score": 20.0,
                    "logical_node_ids_json": ["node-1"],
                    "content": "Voltage: 24 V",
                    "content_for_rerank": "Voltage: 24 V",
                    "metadata_json": {"chunk_type": "spec_record"},
                }
            ]
        raise AssertionError(f"Unexpected query: {query} {params}")

    monkeypatch.setattr(main, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)
    monkeypatch.setattr("apps.api.debug.fetch_one", fake_fetch_one)
    monkeypatch.setattr("apps.api.debug.fetch_all", fake_fetch_all)

    response = client.get("/debug/documents/doc-1?sample_limit=5", headers=ADMIN_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["document"]["document_id"] == "doc-1"
    assert payload["logical_nodes"]["samples"][0]["spec_name"] == "Voltage"
    assert payload["retrieval_chunks"]["samples"][0]["chunk_type"] == "spec_record"
