from manuals_rag_answering import evidence_gate


def _input(answer_text: str, passage: str, *, chunk_id: str = "source-1"):
    citation = {"chunk_id": chunk_id, "document_id": "manual-1", "pages": [1], "quote_span": None}
    answer = {
        "answer": answer_text, "confidence": "high", "citations": [citation],
        "used_documents": [{"document_id": "manual-1"}], "warnings": [],
        "insufficient_evidence": False,
    }
    observations = [{"tool": "manuals_rag", "result": {
        "answer": answer_text, "insufficient_evidence": False, "citations": [citation],
        "evidence_results": [{"chunk_id": chunk_id, "source_document_id": "manual-1", "content": passage}],
    }}]
    return answer, observations


def test_gate_rejects_wrong_torque_from_cited_manual(monkeypatch):
    answer, observations = _input(
        "The IV4-400CA M3 mounting torque is 0.2 to 0.3 N·m.",
        "IV4-400CA M3 mounting hole tightening torque: 0.5 to 0.7 N·m.",
    )
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: (_ for _ in ()).throw(AssertionError("must not call model")))
    result = evidence_gate.gate_agent_answer("What is the mounting torque?", answer, observations)
    assert result["insufficient_evidence"] is True
    assert result["citations"] == []
    assert "0.2" not in result["answer"]


def test_gate_rejects_wrong_per_unit_scope_even_when_total_appears(monkeypatch):
    answer, observations = _input(
        "One CA-E100 unit connects four cameras per CA-E100.",
        "2 color/monochrome cameras per CA-E100, up to 4 cameras via a maximum of 2 units.",
    )
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: (_ for _ in ()).throw(AssertionError("must not call model")))
    result = evidence_gate.gate_agent_answer("How many cameras can one CA-E100 connect?", answer, observations)
    assert result["insufficient_evidence"] is True
    assert result["evidence_gate"]["reason"] == "per-unit quantity is absent from cited passages"


def test_gate_rejects_one_unit_wording_when_two_units_make_the_total(monkeypatch):
    answer, observations = _input(
        "In the XG-X2802, one optional CA-E100 unit can connect four cameras.",
        "XG-X2802: 2 color/monochrome cameras per CA-E100, up to 4 cameras via a maximum of 2 units.",
    )
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: (_ for _ in ()).throw(AssertionError("must not call model")))
    result = evidence_gate.gate_agent_answer("How many cameras can one CA-E100 connect in XG-X2802?", answer, observations)
    assert result["insufficient_evidence"] is True
    assert result["evidence_gate"]["reason"] == "per-unit quantity is absent from cited passages"


def test_gate_accepts_correct_one_unit_count_with_scoped_table_row(monkeypatch):
    passage = "XG-X2802: 2 color/monochrome cameras per CA-E100, up to 4 cameras via a maximum of 2 units."
    answer, observations = _input("In the XG-X2802, one CA-E100 unit can connect 2 cameras.", passage)
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: ({
        "supported": True, "complete": True, "unsupported_claims": [], "missing_requirements": [],
        "evidence_quotes": [{"chunk_id": "source-1", "quote": passage}],
    }, ""))
    result = evidence_gate.gate_agent_answer("How many cameras can one CA-E100 connect in XG-X2802?", answer, observations)
    assert result["evidence_gate"]["status"] == "accepted"


def test_gate_rejects_incomplete_answer_despite_exact_passage(monkeypatch):
    passage = "Rated input voltage: 85 to 264 VAC, 47 to 63 Hz, 110 to 370 VDC. Rated output voltage: 24 VDC."
    answer, observations = _input("CA-U5 output voltage is 24 VDC.", passage)
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: ({
        "supported": True, "complete": False, "unsupported_claims": [],
        "missing_requirements": ["AC and DC input ranges"],
        "evidence_quotes": [{"chunk_id": "source-1", "quote": "Rated output voltage: 24 VDC"}],
    }, ""))
    result = evidence_gate.gate_agent_answer("What are the input ranges and output voltage?", answer, observations)
    assert result["insufficient_evidence"] is True
    assert result["citations"] == []


def test_gate_accepts_cited_complete_answer(monkeypatch):
    passage = "The 24 VDC supply connects to terminals 1 and 2."
    answer, observations = _input("Connect the 24 VDC supply to terminals 1 and 2.", passage)
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: ({
        "supported": True, "complete": True, "unsupported_claims": [], "missing_requirements": [],
        "evidence_quotes": [{"chunk_id": "source-1", "quote": passage}],
    }, ""))
    result = evidence_gate.gate_agent_answer("Where does the 24 VDC supply connect?", answer, observations)
    assert result["evidence_gate"]["status"] == "accepted"
    assert result["insufficient_evidence"] is False
    assert result["citations"] == answer["citations"]


def test_gate_never_accepts_a_pooled_citation_without_a_passage(monkeypatch):
    answer, observations = _input("Connect to terminal 1.", "Connect to terminal 1.")
    answer["citations"].append({"chunk_id": "unseen", "document_id": "manual-1"})
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: (_ for _ in ()).throw(AssertionError("must not call model")))
    result = evidence_gate.gate_agent_answer("Where does it connect?", answer, observations)
    assert result["insufficient_evidence"] is True
    assert result["evidence_gate"]["reason"] == "citation has no matching retrieved passage"


def test_gate_fails_closed_when_verifier_unavailable(monkeypatch):
    answer, observations = _input("Connect to terminal 1.", "Connect to terminal 1.")
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: (_ for _ in ()).throw(TimeoutError()))
    result = evidence_gate.gate_agent_answer("Where does it connect?", answer, observations)
    assert result["insufficient_evidence"] is True
    assert result["evidence_gate"]["reason"] == "verifier unavailable or malformed"


def test_gate_rejects_passage_for_different_model(monkeypatch):
    # Exact class of the live case-018 miss: XG-X2702/2502/2202 table row
    # cited for a question explicitly scoped to XG-X2802.
    answer, observations = _input(
        "One CA-E100 in the XG-X2802 connects up to four cameras.",
        "Column headers: XG-X2702 > XG-X2502 > XG-X2202 > 2 color/monochrome cameras; "
        "a maximum of four are connectable by connecting one optional CA-E100.",
    )
    monkeypatch.setattr(evidence_gate, "chat_json", lambda **_kw: (_ for _ in ()).throw(AssertionError("must not call model")))
    result = evidence_gate.gate_agent_answer("How many cameras does one CA-E100 support in XG-X2802?", answer, observations)
    assert result["insufficient_evidence"] is True
    assert result["evidence_gate"]["reason"] == "requested model is absent from cited passages"
