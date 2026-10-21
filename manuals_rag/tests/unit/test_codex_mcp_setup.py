import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from manuals_rag_inference import codex_mcp_setup as setup
from manuals_rag_inference import agent_link_session


@pytest.fixture
def plan(tmp_path):
    codex = tmp_path/'Codex app/codex.exe';codex.parent.mkdir();codex.touch()
    adapter = tmp_path/'Node app/run_agent_link_mcp.py';adapter.parent.mkdir();adapter.touch()
    return setup.setup_plan(codex=codex, executable=Path('python.exe').absolute(),
                            frozen=False, script=adapter, session_path=tmp_path/'private/owner.json')


class FakeCli:
    def __init__(self):
        self.config = None
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command,kwargs))
        assert kwargs['shell'] is False
        if command[1:3] == ['mcp','get']:
            if self.config is None:
                return SimpleNamespace(returncode=1,stdout='',stderr='Error: No MCP server named provis-agent-link found.')
            return SimpleNamespace(returncode=0,stdout=json.dumps(self.config),stderr='')
        if command[1:3] == ['mcp','add']:
            split = command.index('--');key,value=command[command.index('--env')+1].split('=',1)
            self.config={'name':setup.SERVER_NAME,'enabled':True,'transport':{'type':'stdio','command':command[split+1],'args':command[split+2:],'env':{key:value}}}
            return SimpleNamespace(returncode=0,stdout='Added',stderr='')
        rows=[{'id':1,'result':{'protocolVersion':'2024-11-05'}},
              {'id':2,'result':{'tools':[{'name':name} for name in setup.EXPECTED_TOOLS]}}]
        return SimpleNamespace(returncode=0,stdout='\n'.join(json.dumps(row) for row in rows),stderr='')


def test_exact_argv_preserves_spaces_and_session_path_without_tokens(plan):
    assert plan.command[-2:]==list(plan.server)
    assert '--env' in plan.command and setup.SESSION_ENV+'='+plan.session_path in plan.command
    assert 'mcp' in plan.display_command
    assert 'session_token' not in plan.display_command


def test_missing_installed_node_refuses_registration(tmp_path):
    codex=tmp_path/'codex.exe';codex.touch()
    node=tmp_path/'PROVISInferenceNode.exe'
    with pytest.raises(setup.SetupError,match='installed node executable'):
        setup.setup_plan(codex=codex,executable=node,frozen=True,platform='win32',session_path=tmp_path/'owner.json')


@pytest.mark.parametrize('platform',['win32','linux'])
def test_packaged_windows_and_linux_use_signed_node_without_extra_binary(tmp_path,platform):
    codex=tmp_path/'codex';codex.touch()
    node=tmp_path/'node';node.touch()
    value=setup.setup_plan(codex=codex,executable=node,frozen=True,platform=platform,session_path=tmp_path/'owner.json')
    assert value.server==(str(node),'--agent-link-mcp')


def test_inspect_does_not_add_or_start_adapter(plan):
    runner=FakeCli()
    assert setup.configuration_status(plan,runner=runner)['state']=='not_configured'
    assert len(runner.calls)==1


@pytest.mark.parametrize('approved,owner',[(False,True),(True,False),(1,True)])
def test_registration_requires_exact_approval_and_live_owner(plan,approved,owner):
    runner=FakeCli()
    with pytest.raises(setup.SetupError,match='explicit approval'):
        setup.register(plan,approved=approved,owner_active=lambda:owner,runner=runner,expected_revision='absent')
    assert not runner.calls


def test_expiry_between_preflight_and_add_prevents_persistent_change(plan):
    runner=FakeCli();checks=iter([True,False])
    with pytest.raises(setup.SetupError,match='expired'):
        setup.register(plan,approved=True,owner_active=lambda:next(checks),runner=runner,expected_revision='absent')
    assert runner.config is None


def test_registration_probes_then_adds_and_reads_back_idempotently(plan):
    runner=FakeCli()
    value=setup.register(plan,approved=True,owner_active=lambda:True,runner=runner,expected_revision='absent')
    assert value['state']=='configured'
    assert [call[0] for call in runner.calls][1]==list(plan.server)
    setup.register(plan,approved=True,owner_active=lambda:True,runner=runner)
    assert sum(command[1:3]==['mcp','add'] for command,_ in runner.calls)==1


def test_stale_configuration_and_unapproved_replacement_do_not_write(plan):
    runner=FakeCli();runner.config={'enabled':True,'transport':{'type':'stdio','command':'other-adapter','args':[]}}
    before=setup.configuration_status(plan,runner=runner)
    assert before['state']=='conflict'
    with pytest.raises(setup.SetupError,match='Replacement requires approval'):
        setup.register(plan,approved=True,owner_active=lambda:True,runner=runner,expected_revision=before['revision'])
    with pytest.raises(setup.SetupError,match='changed since'):
        setup.register(plan,approved=True,owner_active=lambda:True,runner=runner,expected_revision='stale',replace_existing=True)
    assert not any(command[1:3]==['mcp','add'] for command,_ in runner.calls)
    assert setup.register(plan,approved=True,owner_active=lambda:True,runner=runner,expected_revision=before['revision'],replace_existing=True)['state']=='configured'


def test_adapter_probe_calls_only_initialize_and_tools_list(plan):
    runner=FakeCli();setup.probe_adapter(plan,runner=runner)
    command,kwargs=runner.calls[0]
    messages=[json.loads(line) for line in kwargs['input'].splitlines()]
    assert [message['method'] for message in messages]==['initialize','notifications/initialized','tools/list']
    assert kwargs['env'][setup.SESSION_ENV]==plan.session_path
    assert kwargs['env']['PYINSTALLER_RESET_ENVIRONMENT']=='1'


def test_error_output_is_not_exposed(plan):
    runner=lambda *args,**kwargs:SimpleNamespace(returncode=1,stdout='',stderr='Bearer VERY_SECRET_TOKEN')
    with pytest.raises(setup.SetupError) as caught:setup.configuration_status(plan,runner=runner)
    assert 'VERY_SECRET_TOKEN' not in str(caught.value)


def test_invalid_adapter_response_does_not_add(plan):
    runner=FakeCli()
    original=runner.__call__
    def malformed(command,**kwargs):
        if command==list(plan.server):return SimpleNamespace(returncode=0,stdout='not protocol JSON',stderr='')
        return original(command,**kwargs)
    with pytest.raises(setup.SetupError,match='initialize/tools check'):
        setup.register(plan,approved=True,owner_active=lambda:True,runner=malformed,expected_revision='absent')
    assert runner.config is None


def test_explicit_owner_path_rejects_relative_paths_and_keeps_existing_default(tmp_path,monkeypatch):
    monkeypatch.setenv(setup.SESSION_ENV,'relative.json')
    with pytest.raises(PermissionError,match='Absolute'):agent_link_session.default_path()
    monkeypatch.setenv(setup.SESSION_ENV,str(tmp_path/'owner.json'))
    assert agent_link_session.default_path()==tmp_path/'owner.json'


def test_disabled_configuration_stays_disabled(plan):
    runner=FakeCli()
    setup.register(plan,approved=True,owner_active=lambda:True,runner=runner,expected_revision='absent')
    runner.config['enabled']=False
    current=setup.configuration_status(plan,runner=runner)
    with pytest.raises(setup.SetupError,match='disabled'):
        setup.register(plan,approved=True,owner_active=lambda:True,runner=runner,expected_revision=current['revision'])
    assert runner.config['enabled'] is False
