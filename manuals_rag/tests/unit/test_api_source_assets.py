import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from apps.api import main
from apps.api.main import app


client = TestClient(app)
USER_HEADERS = {"Authorization": "Bearer user-token"}


@pytest.mark.parametrize(
    ("orchestrator", "attribute"),
    [
        ("langgraph_agent", "langgraph_agentic_retriever"),
        ("llamaindex_agent", "llamaindex_agentic_retriever"),
    ],
)
def test_query_routes_to_selected_agentic_retriever(monkeypatch, orchestrator, attribute):
    calls = []

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(**{**vars(main.settings), "agentic_retrieval_enabled": True}),
    )

    class FakeAgenticRetriever:
        def invoke(self, payload):
            calls.append(payload)
            return {
                "retrieval_results": [
                    {
                        "chunk_id": "chunk-1",
                        "score": 0.9,
                        "title": "Manual",
                        "document_version_id": "ver-1",
                        "source_document_id": "doc-1",
                        "pages": [2],
                        "section_path": ["Serial cables"],
                        "content": "OP-26487 is a straight serial cable.",
                        "metadata": {},
                    }
                ],
                "retrieval_trace": {"completed_hops": ["identify", "orientation"]},
            }

    class FakeAnswer:
        def model_dump(self):
            return {
                "answer": "OP-26487 is straight.",
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            }

    monkeypatch.setattr(main, attribute, FakeAgenticRetriever())
    monkeypatch.setattr(main, "generate_answer", lambda _query, _results: FakeAnswer())

    response = client.post(
        "/query",
        headers=USER_HEADERS,
        json={
            "query": "Which cable connects the port, then what is its orientation?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "retrieval_orchestrator": orchestrator,
            "max_retrieval_hops": 3,
            "max_retrieval_seconds": 12,
        },
    )

    assert response.status_code == 200
    assert calls[0]["max_hops"] == 3
    assert calls[0]["max_seconds"] == 12.0
    assert response.json()["retrieval_orchestrator"] == orchestrator
    assert response.json()["retrieval_trace"]["completed_hops"] == ["identify", "orientation"]


def test_agentic_retrieval_kill_switch_does_not_block_baseline(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(agentic_retrieval_enabled=False))

    main._require_agentic_retrieval_enabled(
        main.QueryRequest(query="ordinary lookup", corpus_ids=["manuals"], retrieval_orchestrator="baseline")
    )
    with pytest.raises(HTTPException) as exc_info:
        main._require_agentic_retrieval_enabled(
            main.QueryRequest(
                query="multi-step lookup",
                corpus_ids=["manuals"],
                retrieval_orchestrator="langgraph_agent",
            )
        )

    assert exc_info.value.status_code == 503


def test_agentic_runtime_budget_is_api_bounded():
    with pytest.raises(ValueError):
        main.QueryRequest(query="lookup", corpus_ids=["manuals"], max_retrieval_seconds=4)
    with pytest.raises(ValueError):
        main.QueryRequest(query="lookup", corpus_ids=["manuals"], max_retrieval_seconds=301)


def test_agentic_runtime_budget_clamps_invalid_environment_value(monkeypatch):
    monkeypatch.setattr(main, "settings", SimpleNamespace(agentic_retrieval_max_seconds=-10))
    request = main.QueryRequest(query="lookup", corpus_ids=["manuals"])

    assert main._agentic_max_seconds(request) == 5.0

    monkeypatch.setattr(main, "settings", SimpleNamespace(agentic_retrieval_max_seconds=999))
    assert main._agentic_max_seconds(request) == 300.0


def test_agentic_query_stream_emits_live_trace_and_final_answer(monkeypatch):
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(**{**vars(main.settings), "agentic_retrieval_enabled": True}),
    )

    class FakeAgenticRetriever:
        def __init__(self, event_callback):
            self.event_callback = event_callback

        def invoke(self, _payload):
            self.event_callback(
                {
                    "event": "plan_completed",
                    "plan": {"mode": "dependent", "rationale": "Two hops", "hops": []},
                    "max_hops": 3,
                }
            )
            self.event_callback(
                {
                    "event": "hop_completed",
                    "hop_id": "identify",
                    "sufficient": True,
                    "assessment": {"query_term_coverage": 1.0},
                    "results": [],
                    "ledger_entry": {},
                }
            )
            return {
                "retrieval_results": [
                    {
                        "chunk_id": "chunk-1",
                        "score": 0.9,
                        "title": "Manual",
                        "document_version_id": "ver-1",
                        "source_document_id": "doc-1",
                        "pages": [2],
                        "section_path": ["Serial cables"],
                        "content": "OP-26487 is a straight serial cable.",
                        "metadata": {},
                    }
                ],
                "retrieval_trace": {"completed_hops": ["identify"], "sufficient": True},
            }

    class FakeAnswer:
        def model_dump(self):
            return {
                "answer": "OP-26487 is straight.",
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            }

    monkeypatch.setattr(
        main,
        "build_langgraph_agentic_retriever",
        lambda *, event_callback: FakeAgenticRetriever(event_callback),
    )
    monkeypatch.setattr(main, "generate_answer", lambda _query, _results: FakeAnswer())

    response = client.post(
        "/query/stream",
        headers=USER_HEADERS,
        json={
            "query": "Which cable, then what orientation?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "retrieval_orchestrator": "langgraph_agent",
            "max_retrieval_hops": 3,
        },
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["event"] for event in events] == [
        "run_started",
        "plan_completed",
        "hop_completed",
        "answer_started",
        "answer_completed",
        "run_completed",
    ]
    assert events[-1]["result"]["answer"] == "OP-26487 is straight."


def test_agentic_query_stream_rejects_baseline():
    response = client.post(
        "/query/stream",
        headers=USER_HEADERS,
        json={
            "query": "What product is this?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "retrieval_orchestrator": "baseline",
        },
    )

    assert response.status_code == 422


def test_agentic_answer_reduces_each_confirmed_required_claim(monkeypatch):
    results = [
        main.SearchResult(
            chunk_id="weight",
            score=1.0,
            title="Camera",
            document_version_id="camera-v1",
            source_document_id="camera-doc",
            pages=[1],
            section_path=["Specifications"],
            content="Weight: 280 g",
            metadata={},
        ),
        main.SearchResult(
            chunk_id="direction",
            score=1.0,
            title="Display",
            document_version_id="display-v1",
            source_document_id="display-doc",
            pages=[12],
            section_path=["Display"],
            content="Slide to the right to increase grayscale.",
            metadata={},
        ),
    ]
    calls = []

    class FakeAnswer:
        def __init__(self, answer, result):
            self.answer = answer
            self.confidence = "high"
            self.used_documents = [
                {
                    "document_id": result.source_document_id,
                    "title": result.title,
                    "version": result.document_version_id,
                    "pages": result.pages,
                    "section_path": result.section_path,
                }
            ]
            self.citations = [
                {
                    "chunk_id": result.chunk_id,
                    "document_id": result.source_document_id,
                    "pages": result.pages,
                    "quote_span": None,
                }
            ]
            self.warnings = []
            self.followup_questions = []
            self.insufficient_evidence = False

    def fake_reduce(objective, branch_results, _rationale):
        calls.append((objective, [result.chunk_id for result in branch_results]))
        return FakeAnswer(
            "280 g" if branch_results[0].chunk_id == "weight" else "Slide right",
            branch_results[0],
        )

    monkeypatch.setattr(main, "_answer_confirmed_claim", fake_reduce)
    answer = main._generate_agentic_answer(
        "Compare the camera weight and display direction.",
        results,
        {
            "required_claim_support": {"weight_claim": ["weight"], "direction_claim": ["direction"]},
            "evidence_ledger": {
                "weight_claim": {"objective": "Find the camera weight"},
                "direction_claim": {"objective": "Find the display direction"},
            },
        },
    )

    assert calls == [
        ("Find the camera weight", ["weight"]),
        ("Find the display direction", ["direction"]),
    ]
    assert answer.answer == "280 g\n\nSlide right"
    assert {citation["chunk_id"] for citation in answer.citations} == {"weight", "direction"}


def _fake_query_result():
    return {
        "answer": {
            "answer": "Use page 2.",
            "confidence": "high",
            "used_documents": [],
            "citations": [{"chunk_id": "chunk-1", "document_id": "doc-1", "pages": [2]}],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        },
        "retrieval_results": [
            {
                "chunk_id": "chunk-1",
                "source_document_id": "doc-1",
                "document_version_id": "ver-1",
                "pages": [2],
            }
        ],
    }


def test_query_source_assets_are_opt_in(monkeypatch):
    class FakeWorkflow:
        def invoke(self, _payload):
            return _fake_query_result()

    monkeypatch.setattr(main, "workflow", FakeWorkflow())

    response = client.post(
        "/query",
        headers=USER_HEADERS,
        json={"query": "where?", "corpus_ids": ["manuals_vendor_keyence"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "source_assets" not in payload
    assert "source_assets" not in payload["citations"][0]


def test_query_can_return_pdf_download_source_assets(monkeypatch):
    class FakeWorkflow:
        def invoke(self, _payload):
            return _fake_query_result()

    class FakeStore:
        def presigned_get_url(self, bucket, object_name, *, expires):
            return f"http://minio.local/{bucket}/{object_name}?signed=1"

    def fake_fetch_all(_query, _params):
        return [
            {
                "id": "doc-1",
                "tenant_id": "tenant-1",
                "current_version_id": "ver-1",
                "title": "Manual",
                "source_filename": "Manual.pdf",
                "storage_uri": "s3://manuals-originals/tenant-1/sha256/manual.pdf",
            }
        ]

    monkeypatch.setattr(main, "workflow", FakeWorkflow())
    monkeypatch.setattr(main, "ObjectStore", FakeStore)
    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)

    response = client.post(
        "/query",
        headers=USER_HEADERS,
        json={
            "query": "where?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "include_source_assets": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_assets"]["documents"][0]["pdf_download_url"] == (
        "http://minio.local/manuals-originals/tenant-1/sha256/manual.pdf?signed=1"
    )
    assert payload["source_assets"]["citation_pages"][0]["page"] == 2
    assert payload["citations"][0]["source_assets"][0]["pdf_download_url"].endswith("?signed=1")
    assert "page_image_url" not in payload["source_assets"]["citation_pages"][0]


def test_query_can_return_stored_page_and_table_image_assets(monkeypatch):
    class FakeWorkflow:
        def invoke(self, _payload):
            return _fake_query_result()

    class FakeStore:
        def presigned_get_url(self, bucket, object_name, *, expires):
            return f"http://minio.local/{bucket}/{object_name}?signed=1"

    def fake_fetch_all(_query, _params):
        return [
            {
                "id": "doc-1",
                "tenant_id": "tenant-1",
                "current_version_id": "ver-1",
                "title": "Manual",
                "source_filename": "Manual.pdf",
                "storage_uri": "s3://manuals-originals/tenant-1/sha256/manual.pdf",
                "docling_artifact_uri": "s3://manuals-artifacts/tenant-1/artifact.json",
            }
        ]

    def fake_image_assets(_store, _document):
        return {
            "page_images": [{"page": 2, "uri": "s3://manuals-artifacts/tenant-1/page-0002.png"}],
            "table_images": [
                {
                    "page": 2,
                    "table_index": 1,
                    "uri": "s3://manuals-artifacts/tenant-1/table-0001.png",
                    "bbox": {"l": 1, "t": 2, "r": 3, "b": 4},
                }
            ],
        }

    monkeypatch.setattr(main, "workflow", FakeWorkflow())
    monkeypatch.setattr(main, "ObjectStore", FakeStore)
    monkeypatch.setattr(main, "fetch_all", fake_fetch_all)
    monkeypatch.setattr(main, "_artifact_image_assets", fake_image_assets)

    response = client.post(
        "/query",
        headers=USER_HEADERS,
        json={
            "query": "where?",
            "corpus_ids": ["manuals_vendor_keyence"],
            "include_page_images": True,
            "include_table_images": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_assets"]["citation_pages"][0]["page_image_url"] == (
        "http://minio.local/manuals-artifacts/tenant-1/page-0002.png?signed=1"
    )
    assert payload["source_assets"]["tables"][0]["table_image_url"] == (
        "http://minio.local/manuals-artifacts/tenant-1/table-0001.png?signed=1"
    )
    assert payload["citations"][0]["table_assets"][0]["table_index"] == 1
