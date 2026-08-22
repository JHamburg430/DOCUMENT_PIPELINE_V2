from fastapi.testclient import TestClient
from time import sleep

from apps.api.main import app
from apps.api import main
from manuals_rag_evals.retrieval_eval import RetrievalEvalCase


client = TestClient(app)
ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


def _fake_eval_case() -> RetrievalEvalCase:
    return RetrievalEvalCase(
        case_id="case-1",
        query="voltage",
        source_document_id="doc-1",
        document_version_id="ver-1",
        source_chunk_id="chunk-1",
        source_title="Doc",
        source_filename="doc.pdf",
        chunk_type="spec_record",
        section_path="Specs",
        page_from=1,
        page_to=1,
        expected_terms=["24", "v"],
        expected_snippet="Voltage: 24 V",
        generation_method="unit",
        source_metadata={},
    )


def test_answer_term_check_matches_slash_terms_across_answer_text():
    evaluation = main._answer_contains_expected_terms(
        {"answer": "The LJ-X8000 Series supports a profile data count of 3200 points."},
        ["3200", "points/profile", "linearity", "significantly"],
    )

    assert evaluation["passed"] is True
    assert evaluation["matched_terms"] == ["3200", "points/profile"]


def _install_fake_run_store(monkeypatch):
    runs = {}
    events = []

    def fake_execute(query, params=()):
        normalized = " ".join(query.split()).lower()
        if normalized.startswith("insert into app_runs"):
            run_id, run_type, request_json = params
            runs[run_id] = {
                "id": run_id,
                "run_type": run_type,
                "status": "queued",
                "request_json": request_json,
                "progress_json": {},
                "result_json": None,
                "error": None,
                "updated_at": "now",
            }
            return
        if normalized.startswith("update app_runs set status = coalesce"):
            status, progress_json, result_json, error, run_id = params
            run = runs[run_id]
            if status is not None:
                run["status"] = status
            if progress_json is not None:
                run["progress_json"] = main.json.loads(progress_json)
            if result_json is not None:
                run["result_json"] = main.json.loads(result_json)
            if error is not None:
                run["error"] = error
            return
        if normalized.startswith("insert into app_run_events"):
            run_id, event_index, event_json = params
            events.append({"run_id": run_id, "event_index": event_index, "event_json": main.json.loads(event_json)})
            return
        if normalized.startswith("update app_runs set status = 'failed'"):
            for run in runs.values():
                if run["status"] == "running":
                    run["status"] = "failed"
                    run["error"] = run["error"] or "Run was left running without progress and was marked failed."
            return
        raise AssertionError(f"Unexpected execute query: {query}")

    def fake_fetch_one(query, params=()):
        normalized = " ".join(query.split()).lower()
        if normalized.startswith("select progress_json from app_runs"):
            return {"progress_json": runs[params[0]]["progress_json"]}
        if normalized.startswith("select status from app_runs"):
            run = runs.get(params[0])
            return {"status": run["status"]} if run else None
        if normalized.startswith("select * from app_runs"):
            return runs.get(params[0])
        raise AssertionError(f"Unexpected fetch_one query: {query}")

    def fake_fetch_all(query, params=()):
        normalized = " ".join(query.split()).lower()
        if "from app_runs" in normalized:
            rows = [dict(run) for run in runs.values()]
            if "null::jsonb as result_json" in normalized:
                for row in rows:
                    summary = (row.get("progress_json") or {}).get("summary")
                    row["progress_json"] = {"summary": summary} if summary is not None else {}
                    row["result_json"] = None
            return rows
        if "from app_run_events" in normalized:
            run_id = params[0]
            return [event for event in events if event["run_id"] == run_id]
        raise AssertionError(f"Unexpected fetch_all query: {query}")

    monkeypatch.setattr(main, "execute", fake_execute)
    monkeypatch.setattr(main, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)
    return runs, events


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


def test_streaming_eval_persists_completion_and_progress_events(monkeypatch):
    runs, events = _install_fake_run_store(monkeypatch)
    monkeypatch.setattr(main, "_load_eval_cases", lambda payload: (["corpus"], None, [_fake_eval_case()], []))
    monkeypatch.setattr(main, "_eval_search_scope", lambda corpus_ids, document_id: (corpus_ids, {}))
    monkeypatch.setattr(main, "score_search_results", lambda _case, _results: {"passed": True, "rank": 1})
    monkeypatch.setattr(main, "_score_answer", lambda _case, _answer, _retrieval: {"passed": True, "failure_reasons": []})
    monkeypatch.setattr(main, "_top_results_from_query_debug", lambda _result: [{"chunk_id": "chunk-1"}])

    def fake_stream_query_debug_events(_request, sample_limit=10):
        yield {
            "event": "run_started",
            "step_sequence": [{"name": "classify_query", "label": "Analyzing query", "model": None}],
        }
        yield {"event": "step_started", "step": "classify_query"}
        yield {"event": "step_completed", "step": "classify_query", "duration_ms": 1.2}
        yield {"event": "run_completed", "result": {"answer": {"answer": "Voltage is 24 V."}, "top_results": [{"chunk_id": "chunk-1"}]}}

    monkeypatch.setattr(main, "stream_query_debug_events", fake_stream_query_debug_events)

    with client.stream(
        "POST",
        "/eval/end-to-end-stream?sample_limit=5",
        headers=ADMIN_HEADERS,
        json={"corpus_ids": ["corpus"], "max_questions": 1, "use_llm_generation": False},
    ) as response:
        assert response.status_code == 200
        streamed = [main.json.loads(line) for line in response.iter_lines() if line]

    run_id = streamed[0]["run_id"]
    assert [event["event"] for event in streamed] == [
        "eval_queued",
        "eval_started",
        "eval_question_started",
        "eval_query_event",
        "eval_query_event",
        "eval_query_event",
        "eval_query_event",
        "eval_question_completed",
        "eval_completed",
    ]
    assert runs[run_id]["status"] == "completed"
    assert runs[run_id]["result_json"]["summary"]["total_questions"] == 1
    assert events[-1]["event_json"]["event"] == "eval_completed"


def test_async_eval_start_returns_run_id_and_persists_queued_event(monkeypatch):
    runs, events = _install_fake_run_store(monkeypatch)
    monkeypatch.setattr(main, "_load_eval_cases", lambda payload: (["corpus"], None, [], ["no cases"]))

    response = client.post(
        "/eval/end-to-end-run?sample_limit=5",
        headers=ADMIN_HEADERS,
        json={"corpus_ids": ["corpus"], "max_questions": 1, "use_llm_generation": False},
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["run_id"]
    assert payload["status"] == "queued"
    assert events[0]["event_json"]["event"] == "eval_queued"
    for _ in range(20):
        if runs[run_id]["status"] == "completed":
            break
        sleep(0.01)
    assert runs[run_id]["status"] == "completed"


def test_streaming_eval_fails_when_nested_query_stream_does_not_complete(monkeypatch):
    runs, events = _install_fake_run_store(monkeypatch)
    monkeypatch.setattr(main, "_load_eval_cases", lambda payload: (["corpus"], None, [_fake_eval_case()], []))
    monkeypatch.setattr(main, "_eval_search_scope", lambda corpus_ids, document_id: (corpus_ids, {}))
    monkeypatch.setattr(main, "stream_query_debug_events", lambda _request, sample_limit=10: iter([{"event": "run_started", "step_sequence": []}]))

    response = client.post(
        "/eval/end-to-end-stream?sample_limit=5",
        headers=ADMIN_HEADERS,
        json={"corpus_ids": ["corpus"], "max_questions": 1, "use_llm_generation": False},
    )

    assert response.status_code == 200
    streamed = [main.json.loads(line) for line in response.text.splitlines() if line]
    run_id = streamed[0]["run_id"]
    assert streamed[0]["event"] == "eval_queued"
    assert streamed[-1]["event"] == "eval_failed"
    assert "ended without a completed result" in streamed[-1]["error"]
    assert runs[run_id]["status"] == "failed"
    assert "ended without a completed result" in runs[run_id]["error"]
    assert events[-1]["event_json"]["event"] == "eval_failed"


def test_run_history_marks_stale_running_runs_failed(monkeypatch):
    runs, _events = _install_fake_run_store(monkeypatch)
    runs["run-1"] = {
        "id": "run-1",
        "run_type": "end_to_end_eval",
        "status": "running",
        "request_json": {},
        "progress_json": {},
        "result_json": None,
        "error": None,
        "updated_at": "old",
    }

    response = client.get("/runs?run_type=end_to_end_eval", headers=ADMIN_HEADERS)

    assert response.status_code == 200
    assert response.json()[0]["status"] == "failed"
    assert "left running" in response.json()[0]["error"]


def test_run_history_can_omit_large_result_payloads(monkeypatch):
    runs, _events = _install_fake_run_store(monkeypatch)
    runs["run-1"] = {
        "id": "run-1",
        "run_type": "end_to_end_eval",
        "status": "completed",
        "request_json": {},
        "progress_json": {"summary": {"total_questions": 1}, "items": [{"large": "progress"}]},
        "result_json": {"items": [{"large": "payload"}]},
        "error": None,
        "updated_at": "now",
    }

    response = client.get("/runs?include_result=false", headers=ADMIN_HEADERS)

    assert response.status_code == 200
    assert response.json()[0]["status"] == "completed"
    assert response.json()[0]["progress_json"]["summary"]["total_questions"] == 1
    assert "items" not in response.json()[0]["progress_json"]
    assert response.json()[0]["result_json"] is None


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
