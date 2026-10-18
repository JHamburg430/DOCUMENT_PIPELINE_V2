"""Dedicated owner server cannot discover or publish desktop-owned threads."""
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import pytest
from manuals_rag_inference import codex_owner_store as store
from manuals_rag_inference.codex_link import CodexLink

@pytest.fixture
def owner_root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'root', lambda: tmp_path)
    return tmp_path

def test_dedicated_mode_lists_only_explicitly_created_sessions(owner_root):
    connector = CodexLink.__new__(CodexLink)
    connector.mode = 'dedicated'
    connector.rpc = lambda *args: pytest.fail('Desktop discovery must not be invoked')
    assert connector.sessions() == []
    identifier = str(uuid.uuid4())
    store.add_session(identifier, 'Owned session')
    assert connector.sessions() == [{'id': identifier, 'name': 'Owned session'}]
    with pytest.raises(PermissionError):
        connector.publish([str(uuid.uuid4())])

def test_create_session_preserves_model_and_approval_defaults_and_does_not_publish(owner_root):
    identifier = str(uuid.uuid4())
    calls = []
    connector = CodexLink.__new__(CodexLink)
    connector.mode, connector.allowed = 'dedicated', set()
    def rpc(method, params):
        calls.append((method, params))
        return {'thread': {'id': identifier}}
    connector.rpc = rpc
    assert connector.create_session('Owner test')['id'] == identifier
    assert calls == [('thread/start', {'cwd': str(owner_root / 'owner-workspace')})]
    assert connector.allowed == set()
    connector.publish([identifier])
    assert connector.allowed == {identifier}

def test_proxy_cannot_create_dedicated_sessions(owner_root):
    connector = CodexLink.__new__(CodexLink)
    connector.mode = 'proxy'
    with pytest.raises(PermissionError):
        connector.create_session()

def test_settings_require_an_explicit_known_connection_mode(owner_root):
    assert store.server_mode() == 'proxy'
    path = owner_root / 'codex-owner-settings.json'
    path.write_text(json.dumps({'server_mode': 'dedicated'}))
    assert store.server_mode() == 'dedicated'
    path.write_text(json.dumps({'server_mode': 'remote-shell'}))
    with pytest.raises(ValueError):
        store.server_mode()

def test_duplicate_server_cannot_claim_the_same_owned_sessions(owner_root):
    first = store.ServerLock()
    try:
        with pytest.raises(PermissionError):
            store.ServerLock()
    finally:
        first.close()
    second = store.ServerLock()
    second.close()

def test_bad_session_registry_is_not_imported(owner_root):
    (owner_root / 'codex-owner-sessions.json').write_text(json.dumps({'foreign-thread': 'Desktop'}))
    with pytest.raises(ValueError):
        store.owned_sessions()

def test_channel_and_dm_reuse_the_same_owned_codex_threads_after_restart(owner_root):
    channel_id, dm_id = str(uuid.uuid4()), str(uuid.uuid4())
    calls=[]
    first=CodexLink.__new__(CodexLink)
    first.mode,first.allowed='dedicated',set()
    def create(method,params):
        calls.append((method,params))
        return {'thread':{'id':channel_id if len(calls)==1 else dm_id}}
    first.rpc=create
    channel=first.ensure_conversation('channel','PROVIS Node Channel')
    dm=first.ensure_conversation('dm:Johns_Dell_V2','PROVIS DM with Johns_Dell_V2')
    assert channel['id']==channel_id and dm['id']==dm_id
    assert first.allowed=={channel_id,dm_id}
    assert [name for name,_ in calls]==['thread/start','thread/start']

    reconnected=CodexLink.__new__(CodexLink)
    reconnected.mode,reconnected.allowed='dedicated',set()
    reconnected.rpc=lambda *args: pytest.fail('A persisted conversation must not start a second Codex thread')
    assert reconnected.ensure_conversation('channel','ignored')['id']==channel_id
    assert reconnected.ensure_conversation('dm:Johns_Dell_V2','ignored')['id']==dm_id
    assert reconnected.allowed=={channel_id,dm_id}

def test_owner_runtime_never_proxies_or_lists_unrelated_desktop_threads(monkeypatch):
    from manuals_rag_inference import codex_link, agent_link_runtime
    instances=[]
    class DedicatedOnly:
        def __init__(self, *, mode):
            instances.append(mode)
            assert mode=='dedicated'
    monkeypatch.setattr(codex_link,'CodexLink',DedicatedOnly)
    runtime=agent_link_runtime.OwnerRuntime.__new__(agent_link_runtime.OwnerRuntime)
    runtime.codex=None
    assert isinstance(runtime.connect_local_codex(),DedicatedOnly)
    assert instances==['dedicated']

def test_local_agent_identity_is_distinct_and_persistent(owner_root):
    first=store.agent_identity()
    assert str(uuid.UUID(first))==first
    assert store.agent_identity()==first
    assert (owner_root/'codex-owner-agent.json').exists()

def test_new_dedicated_thread_sends_without_resuming_unpersisted_rollout():
    import threading
    connector = CodexLink.__new__(CodexLink)
    connector.mode = 'dedicated'
    connector.lock = threading.RLock();connector.turns = {};connector.allowed = {'new'}
    calls = []
    def rpc(method, params, **kwargs):
        calls.append(method)
        if method == 'thread/read': return {'thread': {'status': {'type': 'idle'}}}
        if method == 'thread/resume': pytest.fail('New loaded thread has no persisted rollout yet')
        if method == 'turn/start':
            connector.turns['new']['events']['turn'] = {'done': True, 'text': 'PROVIS_LINK_OK', 'status': 'completed'}
            return {'turn': {'id': 'turn'}}
    connector.rpc = rpc
    assert connector.send('new','fixture')['status'] == 'completed'
    assert calls == ['thread/read','turn/start']

def test_unloaded_dedicated_thread_resumes_before_sending():
    import threading
    connector = CodexLink.__new__(CodexLink)
    connector.mode = 'dedicated'
    connector.lock = threading.RLock();connector.turns = {};connector.allowed = {'saved'}
    calls = []
    def rpc(method, params, **kwargs):
        calls.append(method)
        if method == 'thread/read': return {'thread': {'status': {'type': 'notLoaded'}}}
        if method == 'thread/resume': return {'thread': {'id': 'saved'}}
        if method == 'turn/start':
            connector.turns['saved']['events']['turn'] = {'done': True, 'text': 'ok', 'status': 'completed'}
            return {'turn': {'id': 'turn'}}
    connector.rpc = rpc
    assert connector.send('saved','fixture')['status'] == 'completed'
    assert calls == ['thread/read','thread/resume','turn/start']
