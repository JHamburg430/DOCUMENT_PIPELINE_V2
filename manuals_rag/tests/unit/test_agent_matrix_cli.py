import importlib.util
from pathlib import Path
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
    output = tmp_path / 'report.json'
    exit_file = tmp_path / 'matrix.exit'
    monkeypatch.setattr('sys.argv', ['matrix', '--dataset', str(tmp_path / 'input'), '--corpus-id', 'test',
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
