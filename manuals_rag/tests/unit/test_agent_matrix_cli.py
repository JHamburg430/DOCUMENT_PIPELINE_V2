import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import pytest


def _runner():
    path = Path(__file__).resolve().parents[2] / 'scripts/benchmark/compare_agentic_retrieval.py'
    spec = importlib.util.spec_from_file_location('matrix_cli_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failure,expected', [(None, 0), (RuntimeError('failed'), 1), (SystemExit(143), 143)])
def test_matrix_cli_records_success_failure_and_termination(tmp_path, monkeypatch, failure, expected):
    module = _runner()
    dataset = tmp_path / 'input'
    dataset.write_text('{"case_id":"smoke"}\n')
    output = tmp_path / 'report.json'
    exit_file = tmp_path / 'matrix.exit'
    monkeypatch.setattr('sys.argv', ['matrix', '--dataset', str(dataset), '--corpus-id', 'test',
                                    '--output', str(output), '--exit-file', str(exit_file), '--quiet'])
    def run(args):
        if failure is not None:
            raise failure
        return {'case_count': 1, 'items': [{'langgraph': {}, 'llamaindex': {}}]}
    monkeypatch.setattr(module, 'run', run)
    if failure is not None:
        with pytest.raises(type(failure)):
            module.main()
    else:
        module.main()
    assert exit_file.read_text().strip() == str(expected)
    assert output.exists() is (expected == 0)
    launch = output.with_suffix('.launch.json')
    assert launch.exists()
    assert launch.read_text().count('"run_id"') == 1


def test_matrix_output_lock_rejects_a_second_writer(tmp_path):
    module = _runner()
    lock_path = tmp_path / 'matrix.lock'
    with module._exclusive_output_lock(lock_path, 'first'):
        with pytest.raises(RuntimeError, match='already owned'):
            with module._exclusive_output_lock(lock_path, 'second'):
                pass


def test_matrix_cli_refuses_to_overwrite_an_existing_artifact_set(tmp_path, monkeypatch):
    module = _runner()
    dataset = tmp_path / 'input'
    dataset.write_text('{"case_id":"smoke"}\n')
    output = tmp_path / 'report.json'
    output.write_text('{"immutable":true}\n')
    exit_file = tmp_path / 'matrix.exit'
    exit_file.write_text('0\n')
    monkeypatch.setattr('sys.argv', ['matrix', '--dataset', str(dataset), '--corpus-id', 'test',
                                    '--output', str(output), '--exit-file', str(exit_file), '--quiet'])

    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        module.main()

    assert output.read_text() == '{"immutable":true}\n'
    assert exit_file.read_text() == '0\n'


def test_matrix_case_concurrency_keeps_single_writer_and_dataset_order(tmp_path, monkeypatch):
    module = _runner()
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("".join(json.dumps({"case_id": case_id}) + "\n" for case_id in ("a", "b", "c")))
    output = tmp_path / "report.json"
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    writer_threads: list[int] = []
    real_atomic_write = module._atomic_write_json

    def fake_evaluate(_args, raw_case, _question_number):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03 if raw_case["case_id"] == "a" else 0.01)
        with lock:
            active -= 1
        return {
            "case_id": raw_case["case_id"],
            "agent_case_category": "single_hop",
            "baseline": {},
            "langgraph": {"agent_evaluation": {}},
            "llamaindex": {"agent_evaluation": {}},
        }

    def recording_write(path, payload):
        writer_threads.append(threading.get_ident())
        real_atomic_write(path, payload)

    monkeypatch.setattr(module, "_evaluate_case", fake_evaluate)
    monkeypatch.setattr(module, "_atomic_write_json", recording_write)
    monkeypatch.setattr(module, "_summary", lambda *_args: {})
    monkeypatch.setattr(module, "_category_summary", lambda *_args: {})
    provenance = {
        "run_id": "concurrency-test",
        "dataset": {"ordered_case_keys": ["a", "b", "c"]},
    }
    args = SimpleNamespace(
        dataset=dataset,
        limit=3,
        offset=0,
        provenance=provenance,
        case_concurrency=2,
        output=output,
        progress_jsonl=False,
        max_hops=4,
        no_llm=True,
    )

    report = module.run(args)
    partial = json.loads(output.with_suffix(".partial.json").read_text())

    assert maximum_active == 2
    assert [item["case_id"] for item in report["items"]] == ["a", "b", "c"]
    assert [item["case_id"] for item in partial["items"]] == ["a", "b", "c"]
    assert partial["completed_case_keys"] == ["a", "b", "c"]
    assert set(writer_threads) == {threading.get_ident()}


def test_case_answers_run_concurrently_and_merge_in_backend_order(monkeypatch):
    module = _runner()
    fixture = Path(__file__).resolve().parents[1] / "fixtures/heldout_retrieval_eval_production_v117.jsonl"
    raw_case = json.loads(fixture.read_text().splitlines()[0])
    baseline = module.SearchResult(
        chunk_id="baseline",
        score=1.0,
        title="Manual",
        document_version_id="version",
        source_document_id=raw_case["source_document_id"],
        pages=[1],
        section_path=["Section"],
        content="baseline evidence",
        metadata={},
    )

    def backend_output(name):
        result = baseline.model_copy(update={"chunk_id": name, "content": f"{name} evidence"})
        return {
            "elapsed_ms": 1.0,
            "sufficient": True,
            "stop_reason": "sufficient",
            "result_chunk_ids": [name],
            "result_document_ids": [raw_case["source_document_id"]],
            "results": [result.model_dump()],
            "stage_snapshots": [],
            "trace": {"cost": {}},
        }

    monkeypatch.setattr(module, "retrieve", lambda *_args, **_kwargs: [baseline])
    monkeypatch.setattr(module, "build_filters", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "score_search_results", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "score_agent_run", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module, "assess_evidence_sufficiency", lambda *_args: SimpleNamespace(sufficient=True))
    monkeypatch.setattr(
        module,
        "compare_agentic_backends",
        lambda *_args, **_kwargs: {
            "langgraph": backend_output("langgraph"),
            "llamaindex": backend_output("llamaindex"),
            "equivalent_result_chunks": False,
        },
    )

    rendezvous = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def generate_answer(_query, results):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        rendezvous.wait(timeout=2)
        with lock:
            active -= 1
        return SimpleNamespace(model_dump=lambda: {"answer": results[0].chunk_id, "citations": []})

    monkeypatch.setattr(module, "generate_answer", generate_answer)
    args = SimpleNamespace(
        corpus_id=["manuals_vendor_keyence"],
        max_hops=4,
        no_llm=True,
        backend_concurrency=1,
        progress_jsonl=False,
    )

    item = module._evaluate_case(args, raw_case, 1)

    assert maximum_active == 2
    assert list(item)[-2:] == ["langgraph", "llamaindex"]
    assert item["langgraph"]["answer"]["answer"] == "langgraph"
    assert item["llamaindex"]["answer"]["answer"] == "llamaindex"
