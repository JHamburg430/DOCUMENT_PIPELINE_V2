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
