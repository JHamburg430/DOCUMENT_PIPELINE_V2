import importlib,io,json,sys

import pytest

from manuals_rag_inference import mcp_stdio


def test_mcp_mode_runs_before_node_settings_and_single_instance(monkeypatch):
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'provis_inference_node/src'))
    path=Path(__file__).resolve().parents[2]/'scripts/run_inference_node_client.py'
    spec=spec_from_file_location('mcp_mode_node_client',path);node=module_from_spec(spec);spec.loader.exec_module(node)
    monkeypatch.setattr(sys,'argv',['PROVISInferenceNode.exe','--agent-link-mcp'])
    monkeypatch.setattr(mcp_stdio,'run_stdio',lambda:7)
    monkeypatch.setattr(node,'_managed_update_settings',lambda:pytest.fail('MCP must not initialize node/updater'))
    with pytest.raises(SystemExit) as caught:node.main()
    assert caught.value.code==7


def test_transport_failure_does_not_fall_through_to_gui(monkeypatch):
    monkeypatch.setattr(sys,'stdin',None)
    monkeypatch.setattr(mcp_stdio,'_windows_pipe',lambda *args:(_ for _ in ()).throw(OSError('no inherited pipe')))
    assert mcp_stdio.run_stdio()==2


def test_stdio_lists_tools_without_loading_owner_session(monkeypatch):
    import provis_agent_link_mcp as adapter
    stdin=io.StringIO(json.dumps({'jsonrpc':'2.0','id':1,'method':'tools/list'})+'\n')
    stdout=io.StringIO()
    monkeypatch.setattr(sys,'stdin',stdin);monkeypatch.setattr(sys,'stdout',stdout)
    assert mcp_stdio.run_stdio()==0
    result=json.loads(stdout.getvalue())
    assert len(result['result']['tools'])==3
