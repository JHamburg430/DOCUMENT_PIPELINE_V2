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
