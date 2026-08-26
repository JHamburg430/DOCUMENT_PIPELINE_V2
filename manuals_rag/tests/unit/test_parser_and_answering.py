from pathlib import Path

import fitz

from manuals_rag_answering.generator import _parse_relevance_response, generate_answer, generate_answer_with_trace, judge_retrieval_relevance, prioritize_results_for_answer, validate_answer
from manuals_rag_parsers.docling_parser import (
    _classify_block,
    _docling_page_batches,
    _docling_pipeline_options,
    _docling_table_blocks,
    _docling_table_child_refs,
    _docling_text_blocks,
    _is_ignorable_block,
    _looks_like_heading,
    parse_document,
    _resolved_page_no,
)
from manuals_rag_schemas.documents import AnswerResponse, SearchResult
from manuals_rag_schemas.enums import NodeType, ParseProfile
from tests.helpers import tmp_eval_small_pdf_path


def test_classify_block_detects_spec_and_table():
    spec_type, spec_meta = _classify_block("Voltage: 24 V")
    assert spec_type == NodeType.spec
    assert spec_meta["spec_name"] == "Voltage"

    table_type, table_meta = _classify_block("Item | Value\nCurrent | 1 A")
    assert table_type == NodeType.table
    assert table_meta["table_json"]["headers"] == ["Item", "Value"]


def test_parser_ignores_docling_image_placeholders_and_asset_names():
    assert _is_ignorable_block("<!-- image -->")
    assert _is_ignorable_block("ca_en100u_dimension_01.gif")

    node_type, metadata = _classify_block("ca: en100u_dimension_01.gif")
    assert node_type == NodeType.paragraph
    assert "spec_name" not in metadata


def test_parser_does_not_treat_urls_or_page_markers_as_specs():
    node_type, metadata = _classify_block("https: //www.keyence.com")
    assert node_type == NodeType.paragraph
    assert "spec_name" not in metadata

    node_type, metadata = _classify_block("Page 1 of 8")
    assert node_type == NodeType.paragraph
    assert "spec_name" not in metadata


def test_parser_ignores_low_signal_headings():
    assert _looks_like_heading("Y") is False
    assert _looks_like_heading(":") is False
    assert _looks_like_heading("5") is False


def test_validate_answer_adds_conflict_warning_and_citation():
    answer = AnswerResponse(
        answer="Use the latest revision.",
        confidence="medium",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Setup"],
            content="First revision content",
            metadata={"chunk_type": "procedure_record"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v2",
            source_document_id="d1",
            pages=[2],
            section_path=["Setup"],
            content="Second revision content",
            metadata={"chunk_type": "procedure_record"},
        ),
    ]
    validated = validate_answer(answer, results)
    assert validated.citations
    assert any("multiple document versions" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_when_generated_answer_is_not_supported():
    answer = AnswerResponse(
        answer="Step 1: Images captured when Track Object is enabled.",
        confidence="medium",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Specify other conditions for the Defect tool as required.",
            metadata={"chunk_type": "atomic_text"},
        ),
    ]
    validated = validate_answer(answer, results)
    assert validated.answer == "Defect Tool"
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_falls_back_when_citation_quote_is_not_in_cited_chunk():
    answer = AnswerResponse(
        answer="Change the trigger signal assignment to one that can be used.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "settings-page",
                "document_id": "d1",
                "pages": [10],
                "quote_span": "Change the trigger signal assignment to one that can be used.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="settings-page",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Trigger settings"],
            content="If capture on trigger input is disabled, this setting cannot be changed.",
            metadata={"chunk_type": "atomic_text"},
        )
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "If capture on trigger input is disabled, this setting cannot be changed."
    assert validated.citations[0]["quote_span"] is None
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_does_not_repair_unsupported_citation_from_answer_overlap():
    answer = AnswerResponse(
        answer="Change to a trigger signal that can be used.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "settings-page",
                "document_id": "d1",
                "pages": [10],
                "quote_span": "Change to a trigger signal that can be used.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="settings-page",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[10],
            section_path=["Trigger settings"],
            content="If capture on trigger input is disabled, this setting cannot be changed.",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="action-row",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Corrective Action: Change to a trigger signal that can be used.",
            metadata={"chunk_type": "table_record"},
        ),
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "If capture on trigger input is disabled, this setting cannot be changed."
    assert validated.citations[0]["chunk_id"] == "settings-page"
    assert validated.citations[0]["quote_span"] is None
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_accepts_citation_quote_from_cited_chunk():
    answer = AnswerResponse(
        answer="Change to a trigger signal that can be used.",
        confidence="high",
        used_documents=[],
        citations=[
            {
                "chunk_id": "action-row",
                "document_id": "d1",
                "pages": [11],
                "quote_span": "Change to a trigger signal that can be used.",
            }
        ],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="action-row",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[11],
            section_path=["Troubleshooting"],
            content="Corrective Action: Change to a trigger signal that can be used.",
            metadata={"chunk_type": "table_record"},
        )
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == "Change to a trigger signal that can be used."
    assert not any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_accepts_short_identifier_answer_when_supported():
    answer = AnswerResponse(
        answer="CA-EN100U",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="CA-EN100U Data Sheet",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["CA-EN100U"],
            content="CA-EN100U Encoder relay unit",
            metadata={"chunk_type": "atomic_text"},
        )
    ]
    validated = validate_answer(answer, results)
    assert validated.answer == "CA-EN100U"
    assert not any("not sufficiently supported" in warning for warning in validated.warnings)


def test_validate_answer_expands_terse_structured_table_answer():
    answer = AnswerResponse(
        answer="OP-42284",
        confidence="high",
        used_documents=[],
        citations=[],
        warnings=[],
        followup_questions=[],
        insufficient_evidence=False,
    )
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Ring light options",
            document_version_id="v1",
            source_document_id="d1",
            pages=[13],
            section_path=["Options"],
            content='Part number: 19.69" OP-42284; Applicable light: CA-DRx9',
            metadata={"chunk_type": "table_record"},
        )
    ]

    validated = validate_answer(answer, results)

    assert validated.answer == 'Part number: 19.69" OP-42284; Applicable light: CA-DRx9'
    assert any("not sufficiently supported" in warning for warning in validated.warnings)


def test_parse_relevance_response_detects_missing_chunk_ids_and_normalizes_null_fields():
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect grouping function",
            metadata={"chunk_type": "section_window"},
        ),
    ]
    parsed, diagnostics = _parse_relevance_response(
        '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":null,"reason":null}]}',
        "Defect Tool",
        results,
    )
    assert parsed[0]["verdict"] == "relevant"
    assert parsed[1]["verdict"] == "relevant"
    assert parsed[1]["reason"]
    assert diagnostics["invalid_items"]
    assert diagnostics["missing_chunk_ids"] == []


def test_judge_retrieval_relevance_retries_when_chunk_coverage_is_incomplete(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect grouping function",
            metadata={"chunk_type": "section_window"},
        ),
    ]
    prompts: list[str] = []

    class FakeResponse:
        def __init__(self, body: str):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": self._body}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, _path, json):
            prompts.append(json["prompt"])
            self._calls += 1
            if self._calls == 1:
                return FakeResponse('{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."}]}')
            return FakeResponse('{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related but broader."}]}')

    monkeypatch.setattr(
        "manuals_rag_answering.generator.chat_json",
        lambda **kwargs: (
            {"items": []},
            '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."}]}' if len(prompts) == 0 else '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related but broader."}]}',
        ),
    )
    original_prompt = None

    def fake_chat_json(**kwargs):
        nonlocal original_prompt
        prompts.append(kwargs["messages"][-1]["content"])
        if len(prompts) == 1:
            return {"items": []}, '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."}]}'
        return {"items": []}, '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Direct match."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related but broader."}]}'

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    judgments = judge_retrieval_relevance("Defect Tool", results)

    assert [item["chunk_id"] for item in judgments] == ["c1", "c2"]
    assert judgments[1]["verdict"] == "potentially_relevant"
    assert len(prompts) == 2
    assert "Required chunk_ids in order" in prompts[1]


def test_answer_prioritization_excludes_wrong_model_family_table_rows(monkeypatch):
    results = [
        SearchResult(
            chunk_id="lj-x-warning",
            score=0.9,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[36],
            section_path=["LJ-X8200/LJ-X8300/LJ-X8400/LJ-X8900"],
            content="LASER RADIATION CLASS 2M LASER PRODUCT Wavelength : 405nm Output : 10mW",
            metadata={"chunk_type": "spec_record"},
        ),
        SearchResult(
            chunk_id="lj-x-table",
            score=0.8,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[36],
            section_path=["HALCON"],
            content=(
                "Column headers: LJ-X8020 > LJ-X8060 > LJ-X8080 > LJ-X8200; "
                "Row headers: Light source > Laser class; Cell value: Class 2M laser product"
            ),
            metadata={"chunk_type": "table_record"},
        ),
        SearchResult(
            chunk_id="lj-v-table",
            score=0.7,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[45],
            section_path=["135°"],
            content=(
                "Column headers: LJ-V7080/ LJ-V7080B > LJ-V7200/ LJ-V7200B; "
                "Row headers: Light source > Laser class; Cell value: Class 2"
            ),
            metadata={"chunk_type": "table_record", "product_model": "New LJ-X8000 Series"},
        ),
        SearchResult(
            chunk_id="lj-s-table",
            score=0.6,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[47],
            section_path=["135°"],
            content="Column headers: LJ-S015 > LJ-S025 > LJ-S040; Row headers: Laser light source; Cell value: 405",
            metadata={"chunk_type": "table_record", "product_model": "New LJ-X8000 Series"},
        ),
        SearchResult(
            chunk_id="lj-s-grouped-table",
            score=0.5,
            title="LJ-X8000",
            document_version_id="v1",
            source_document_id="d1",
            pages=[47],
            section_path=["135°"],
            content=(
                "Laser light source | nm(visible light) wavelength blue semiconductor laser | 405\n"
                "Laser class | 2Mlaser product | Class\n"
                "Output | 10mW"
            ),
            metadata={
                "chunk_type": "table_record",
                "product_model": "New LJ-X8000 Series",
                "table_row_group": True,
                "identifier_tokens": ["LJ-S015", "LJ-S025", "LJ-S040"],
            },
        ),
    ]

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))

    prioritized = prioritize_results_for_answer("New LJ-X8000 Series laser radiation", results)

    judgments = {item["chunk_id"]: item["verdict"] for item in prioritized["judgments"]}
    assert judgments["lj-x-table"] != "not_relevant"
    assert judgments["lj-v-table"] == "not_relevant"
    assert judgments["lj-s-table"] == "not_relevant"
    assert judgments["lj-s-grouped-table"] == "not_relevant"
    assert [result.chunk_id for result in prioritized["prioritized_results"]] == ["lj-x-warning", "lj-x-table"]


def test_generate_answer_uses_fast_model_for_relevance_and_summaries_then_answer_model(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool setup instructions",
            metadata={"chunk_type": "atomic_text"},
        ),
        SearchResult(
            chunk_id="c2",
            score=0.8,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[2],
            section_path=["Tools"],
            content="Defect grouping function details",
            metadata={"chunk_type": "section_window"},
        ),
    ]
    calls: list[dict[str, str]] = []

    def fake_chat_json(**kwargs):
        calls.append(
            {
                "model": kwargs["model"],
                "prompt": kwargs["messages"][-1]["content"],
                "num_predict": kwargs.get("num_predict"),
                "think": kwargs.get("think"),
            }
        )
        prompt = kwargs["messages"][0]["content"]
        if "You are judging whether each evidence item is relevant" in prompt:
            return {"items": []}, '{"items":[{"chunk_id":"c1","verdict":"relevant","reason":"Directly answers."},{"chunk_id":"c2","verdict":"potentially_relevant","reason":"Related context."}]}'
        if "You summarize retrieved evidence" in prompt:
            if '"chunk_id": "c1"' in kwargs["messages"][-1]["content"]:
                return {"summary": "Defect Tool setup instructions."}, '{"summary":"Defect Tool setup instructions."}'
            return {"summary": "Defect grouping function details."}, '{"summary":"Defect grouping function details."}'
        if "You compress multiple evidence summaries" in prompt:
            return {"summary": "Defect Tool setup and grouping details."}, '{"summary":"Defect Tool setup and grouping details."}'
        return {
            "answer": "Use the Defect Tool setup and grouping details.",
            "confidence": "medium",
            "used_documents": [],
            "citations": [],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        }, '{"answer":"Use the Defect Tool setup and grouping details.","confidence":"medium","used_documents":[],"citations":[],"warnings":[],"followup_questions":[],"insufficient_evidence":false}'

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer = generate_answer("How do I use defect detection?", results)

    assert answer.answer
    assert [call["model"] for call in calls[:-1]] == ["qwen3.5:4b", "qwen3.5:4b", "qwen3.5:4b"]
    assert calls[-1]["model"] == "qwen3.5:9b"
    assert calls[-1]["num_predict"] == -1
    assert calls[-1]["think"] is False


def test_generate_answer_reuses_precomputed_prioritized_results_and_summaries(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool setup instructions",
            metadata={"chunk_type": "atomic_text"},
        )
    ]
    summarized_evidence = [
        {
            "chunk_id": "c1",
            "title": "Doc",
            "pages": [1],
            "section_path": ["Tools"],
            "summary": "Defect Tool setup instructions.",
            "source_document_id": "d1",
            "document_version_id": "v1",
        }
    ]
    calls: list[str] = []

    def fail_prioritize(*args, **kwargs):
        raise AssertionError("prioritize_results_for_answer should not be called")

    def fail_summarize(*args, **kwargs):
        raise AssertionError("summarize_results_for_answer should not be called")

    def fake_chat_json(**kwargs):
        calls.append(kwargs["model"])
        return {
            "answer": "Defect Tool setup instructions.",
            "confidence": "medium",
            "used_documents": [],
            "citations": [],
            "warnings": [],
            "followup_questions": [],
            "insufficient_evidence": False,
        }, '{"answer":"Defect Tool setup instructions.","confidence":"medium","used_documents":[],"citations":[],"warnings":[],"followup_questions":[],"insufficient_evidence":false}'

    monkeypatch.setattr("manuals_rag_answering.generator.prioritize_results_for_answer", fail_prioritize)
    monkeypatch.setattr("manuals_rag_answering.generator.summarize_results_for_answer", fail_summarize)
    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer = generate_answer(
        "Defect Tool",
        results,
        prioritized_results=results,
        summarized_evidence=summarized_evidence,
    )

    assert answer.answer == "Defect Tool setup instructions."
    assert calls == ["qwen3.5:9b"]


def test_generate_answer_with_trace_exposes_summary_input_and_fallback_state(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c1",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[1],
            section_path=["Tools"],
            content="Defect Tool setup instructions",
            metadata={"chunk_type": "atomic_text"},
        )
    ]
    summarized_evidence = [
        {
            "chunk_id": "c1",
            "title": "Doc",
            "pages": [1],
            "section_path": ["Tools"],
            "summary": "Use the Defect Tool for setup.",
            "source_document_id": "d1",
            "document_version_id": "v1",
        }
    ]

    def fake_chat_json(**kwargs):
        return (
            {
                "answer": "Use the Defect Tool for setup.",
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            '{"answer":"Use the Defect Tool for setup.","confidence":"high","used_documents":[],"citations":[],"warnings":[],"followup_questions":[],"insufficient_evidence":false}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "Defect Tool",
        results,
        prioritized_results=results,
        summarized_evidence=summarized_evidence,
    )

    assert answer.answer == "Use the Defect Tool for setup."
    assert trace["final_answer"]["model"] == "qwen3.5:9b"
    assert trace["final_answer"]["num_predict"] == -1
    assert trace["final_answer"]["used_fallback"] is False
    assert trace["final_answer"]["summarized_evidence"][0]["summary"] == "Use the Defect Tool for setup."


def test_structured_evidence_uses_direct_summary_without_chunk_summary_model(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[3],
            section_path=["Settings"],
            content="Column headers: setting; Row headers: Width; Cell value: measure",
            metadata={"chunk_type": "table_record", "context_window": "Width setting selects measure."},
        )
    ]
    calls: list[str] = []

    def fake_chat_json(**kwargs):
        calls.append(kwargs["purpose"])
        if kwargs["purpose"] == "chunk_summary":
            raise AssertionError("focused table evidence should not need model summarization")
        if kwargs["purpose"] == "relevance_review":
            return (
                {"items": [{"chunk_id": "c-table", "verdict": "relevant", "reason": "Direct table match."}]},
                '{"items":[{"chunk_id":"c-table","verdict":"relevant","reason":"Direct table match."}]}',
            )
        return (
            {
                "answer": "The Width setting selects measure.",
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            '{"answer":"The Width setting selects measure.","confidence":"high",'
            '"used_documents":[],"citations":[],"warnings":[],'
            '"followup_questions":[],"insufficient_evidence":false}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace("What setting selects measure?", results)

    assert answer.answer == "The Width setting selects measure."
    assert calls == ["relevance_review", "final_answer"]
    summary = trace["final_answer"]["summarized_evidence"][0]
    assert summary["summary_source"] == "direct_evidence"
    assert summary["summary"].startswith("Column headers: setting")
    assert "Context: Width setting selects measure." in summary["summary"]


def test_validation_fallback_preserves_focused_table_cell_before_context(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[7],
            section_path=["PLC"],
            content=(
                "Column headers: 6bit > 5bit; Row headers: 0028 65.0 > "
                "Command output area; Cell value: Command Result"
            ),
            metadata={
                "chunk_type": "table_record",
                "context_window": "status Bit area | 0000 | Result Ready | Cmd Ready",
            },
        )
    ]

    def fake_chat_json(**kwargs):
        return (
            {
                "answer": "Unrelated generated answer",
                "confidence": "medium",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            (
                '{"answer":"Unrelated generated answer","confidence":"medium",'
                '"used_documents":[],"citations":[],"warnings":[],'
                '"followup_questions":[],"insufficient_evidence":false}'
            ),
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace(
        "What command 0028 65.0 Command output area 6bit value applies?",
        results,
        prioritized_results=results,
        summarized_evidence=[
            {
                "chunk_id": "c-table",
                "title": "Doc",
                "pages": [7],
                "section_path": ["PLC"],
                "summary": "The table row says Command Result.",
                "source_document_id": "d1",
                "document_version_id": "v1",
            }
        ],
    )

    assert answer.answer.startswith("Column headers: 6bit > 5bit")
    assert "Cell value: Command Result" in answer.answer
    assert "Context: status Bit area" in answer.answer
    assert trace["final_answer"]["answer_source"] == "fallback_validation"


def test_validation_support_checks_table_cell_before_context(monkeypatch):
    results = [
        SearchResult(
            chunk_id="c-table",
            score=0.9,
            title="Doc",
            document_version_id="v1",
            source_document_id="d1",
            pages=[13],
            section_path=["Ring lights"],
            content='Column headers: Part number; Cell value: 19.69" OP-42284; Row: 4; Column: 0',
            metadata={"chunk_type": "table_record", "context_window": "CA-DRR3 | 1.5W | 12VDC"},
        )
    ]

    def fake_chat_json(**kwargs):
        if kwargs["purpose"] == "relevance_review":
            return (
                {"items": [{"chunk_id": "c-table", "verdict": "relevant", "reason": "Direct table match."}]},
                '{"items":[{"chunk_id":"c-table","verdict":"relevant","reason":"Direct table match."}]}',
            )
        return (
            {
                "answer": 'The part number is 19.69" OP-42284.',
                "confidence": "high",
                "used_documents": [],
                "citations": [],
                "warnings": [],
                "followup_questions": [],
                "insufficient_evidence": False,
            },
            '{"answer":"The part number is 19.69\\" OP-42284.","confidence":"high",'
            '"used_documents":[],"citations":[],"warnings":[],'
            '"followup_questions":[],"insufficient_evidence":false}',
        )

    monkeypatch.setattr("manuals_rag_answering.generator.chat_json", fake_chat_json)

    answer, trace = generate_answer_with_trace("What 19.69 OP-42284 applies?", results)

    assert answer.answer == 'The part number is 19.69" OP-42284.'
    assert trace["final_answer"]["used_fallback"] is False


def test_docling_page_batches_cover_full_document():
    assert _docling_page_batches(10, 4) == [(1, 4), (5, 8), (9, 10)]


def test_resolved_page_no_preserves_original_range():
    assert _resolved_page_no(7, batch_start=5, batch_end=8) == 7
    assert _resolved_page_no(3, batch_start=5, batch_end=8) == 7


def test_docling_text_blocks_map_page_numbers_from_batched_export():
    exported = {
        "texts": [
            {"text": "Section A", "prov": [{"page_no": 1}]},
            {"text": "Step 1. Connect cable", "prov": [{"page_no": 2}]},
        ]
    }
    blocks = _docling_text_blocks(exported, batch_start=5, batch_end=6)
    assert blocks == [(5, "Section A"), (6, "Step 1. Connect cable")]


def test_docling_text_blocks_correct_collapsed_page_one_provenance_from_pdf_text():
    exported = {
        "texts": [
            {"text": "First page setup instructions", "prov": [{"page_no": 1}]},
            {"text": "Second page calibration steps", "prov": [{"page_no": 1}]},
            {"text": "Third page troubleshooting table", "prov": [{"page_no": 1}]},
        ]
    }
    blocks = _docling_text_blocks(
        exported,
        batch_start=7,
        batch_end=9,
        page_texts={
            7: "First page setup instructions",
            8: "Second page calibration steps",
            9: "Third page troubleshooting table",
        },
    )

    assert blocks == [
        (7, "First page setup instructions"),
        (8, "Second page calibration steps"),
        (9, "Third page troubleshooting table"),
    ]


def test_docling_table_blocks_extract_structured_cells_and_skip_child_text_refs():
    exported = {
        "texts": [
            {"text": "Heading", "prov": [{"page_no": 1}]},
            {"text": "Model", "prov": [{"page_no": 1}]},
            {"text": "LJ-X8000", "prov": [{"page_no": 1}]},
        ],
        "tables": [
            {
                "children": [{"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}],
                "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4, "coord_origin": "BOTTOMLEFT"}}],
                "data": {
                    "num_rows": 2,
                    "num_cols": 2,
                    "table_cells": [
                        {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Model", "column_header": True},
                        {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "Repeatability", "column_header": True},
                        {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "LJ-X8000", "row_header": True},
                        {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "0.3 um"},
                    ],
                },
            }
        ],
    }

    refs = _docling_table_child_refs(exported)
    assert refs == {"#/texts/1", "#/texts/2"}
    assert _docling_text_blocks(exported, batch_start=5, batch_end=5, excluded_refs=refs) == [(5, "Heading")]

    table_blocks = _docling_table_blocks(exported, batch_start=5, batch_end=5)
    assert table_blocks[0][0] == 5
    table_json = table_blocks[0][1]
    assert table_json["headers"] == ["Model", "Repeatability"]
    assert table_json["rows"] == [["LJ-X8000", "0.3 um"]]
    assert table_json["row_count"] == 1
    assert table_json["column_count"] == 2
    assert table_json["cells"][0]["column_header"] is True
    assert table_json["bbox"]["coord_origin"] == "BOTTOMLEFT"


def test_docling_pipeline_enables_tableformer_for_standard_manuals():
    options = _docling_pipeline_options(ParseProfile.standard_manual, device="cuda")

    assert options.do_table_structure is True
    assert options.table_structure_options.do_cell_matching is True
    assert options.accelerator_options.device == "cuda"


def _read_pdf(pdf_path: Path) -> bytes:
    return pdf_path.read_bytes()


def test_parse_document_preserves_page_provenance_for_multi_page_fixture():
    pdf_path = tmp_eval_small_pdf_path("AS_151019_LJ-X8000_C_689092_KA_US_2055_2.pdf")
    result = parse_document("page-provenance-version", pdf_path.name, _read_pdf(pdf_path))

    distinct_pages = {node.page_from for node in result.logical_nodes}
    expected_page_count = fitz.open(pdf_path).page_count

    assert result.page_count == expected_page_count
    assert result.page_count > 1
    assert len(distinct_pages) > 1
    assert min(distinct_pages) == 1
    assert max(distinct_pages) == result.page_count
    assert any(node.page_from == 1 for node in result.logical_nodes)
    assert any(node.page_from == result.page_count for node in result.logical_nodes)
    assert all(1 <= node.page_from <= result.page_count for node in result.logical_nodes)


def test_parse_document_docling_artifact_batches_cover_full_page_range():
    pdf_path = tmp_eval_small_pdf_path("AS_151019_LJ-X8000_C_689092_KA_US_2055_2.pdf")
    result = parse_document("artifact-batch-version", pdf_path.name, _read_pdf(pdf_path))

    artifact = result.docling_artifact
    if artifact.get("parser") == "pymupdf_fallback":
        return
    batches = artifact["batches"]
    covered_pages: list[int] = []
    for batch in batches:
        start, end = batch["page_range"]
        covered_pages.extend(range(start, end + 1))

    assert artifact["original_page_count"] == result.page_count
    assert artifact["batch_count"] == len(batches)
    assert covered_pages[0] == 1
    assert covered_pages[-1] == result.page_count
    assert covered_pages == list(range(1, result.page_count + 1))
