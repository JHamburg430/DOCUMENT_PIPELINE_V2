from pathlib import Path

from manuals_rag_evals.tmp_eval_suite import (
    TmpEvalDocSet,
    aggregate_tmp_eval_results,
    discover_tmp_eval_doc_sets,
    render_tmp_eval_markdown,
)


def test_discover_tmp_eval_doc_sets_finds_expected_directories():
    manuals_root = Path(__file__).resolve().parents[2]
    doc_sets = discover_tmp_eval_doc_sets(manuals_root)
    names = [doc_set.name for doc_set in doc_sets]
    assert "tmp_eval_docs" in names
    assert "tmp_eval_docs_medium" in names
    assert "tmp_eval_docs_small" in names
    assert all(doc_set.documents for doc_set in doc_sets)


def test_aggregate_tmp_eval_results_combines_runs():
    report = aggregate_tmp_eval_results(
        keyence_inventory=[
            "/repo/Technical_Documents/Keyence/doc-a.pdf",
            "/repo/Technical_Documents/Keyence/doc-b.pdf",
        ],
        doc_sets=[
            TmpEvalDocSet(name="tmp_eval_docs", directory="/tmp/a", documents=["/tmp/a/doc-a.pdf"]),
            TmpEvalDocSet(name="tmp_eval_docs_small", directory="/tmp/b", documents=["/tmp/b/doc-b.pdf"]),
        ],
        run_results=[
            {
                "doc_set_name": "tmp_eval_docs",
                "summary": {
                    "total_queries": 3,
                    "passed_queries": 2,
                    "failed_queries": 1,
                    "benchmark_validity_rate": 1.0,
                    "candidate_recall_rate": 0.7,
                    "metadata_document_selection_attempts": 3,
                    "metadata_document_selection_recall_rate": 1.0,
                    "metadata_document_selection_rank_1_rate": 2 / 3,
                    "by_chunk_type": {"atomic_text": {"total": 3, "passed": 2}},
                    "by_document": {"doc-a.pdf": {"total": 3, "passed": 2}},
                },
            },
            {
                "doc_set_name": "tmp_eval_docs_small",
                "summary": {
                    "total_queries": 2,
                    "passed_queries": 1,
                    "failed_queries": 1,
                    "benchmark_validity_rate": 1.0,
                    "candidate_recall_rate": 0.5,
                    "metadata_document_selection_attempts": 2,
                    "metadata_document_selection_recall_rate": 0.5,
                    "metadata_document_selection_rank_1_rate": 0.5,
                    "by_chunk_type": {"spec_record": {"total": 2, "passed": 1}},
                    "by_document": {"doc-b.pdf": {"total": 2, "passed": 1}},
                },
            },
        ],
    )
    assert report["overall"]["total_queries"] == 5
    assert report["overall"]["passed_queries"] == 3
    assert report["overall"]["pass_rate"] == 0.6
    assert report["overall"]["benchmark_validity_rate"] == 1.0
    assert report["overall"]["candidate_recall_rate"] == 0.62
    assert report["overall"]["metadata_document_selection_attempts"] == 5
    assert report["overall"]["metadata_document_selection_recall_rate"] == 0.8
    assert report["overall"]["metadata_document_selection_rank_1_rate"] == 0.6
    assert report["overall"]["by_chunk_type"]["atomic_text"]["passed"] == 2
    assert report["overall"]["by_chunk_type"]["spec_record"]["total"] == 2


def test_render_tmp_eval_markdown_includes_overview():
    report = {
        "tmp_coverage": {
            "keyence_inventory_count": 10,
            "tmp_document_count": 3,
            "tmp_documents_present_in_keyence": 3,
            "missing_from_keyence": [],
        },
        "overall": {
            "total_queries": 12,
            "passed_queries": 9,
            "failed_queries": 3,
            "pass_rate": 0.75,
            "benchmark_validity_rate": 1.0,
            "candidate_recall_rate": 0.9,
            "metadata_document_selection_recall_rate": 0.8,
            "metadata_document_selection_rank_1_rate": 0.7,
        },
        "production_readiness": {"ready": True},
        "runs": [
            {"doc_set_name": "tmp_eval_docs", "summary": {"passed_queries": 4, "total_queries": 5, "pass_rate": 0.8}}
        ],
    }
    markdown = render_tmp_eval_markdown(report)
    assert "Tmp Document Retrieval Eval" in markdown
    assert "Keyence inventory PDFs: 10" in markdown
    assert "tmp_eval_docs: 4/5 (80.00%)" in markdown
    assert "Metadata document selection recall: 80.00%" in markdown
    assert "Production ready: yes" in markdown
