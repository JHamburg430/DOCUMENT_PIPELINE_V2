import time
import tkinter as tk
from types import SimpleNamespace

import pytest

from manuals_rag_inference import agent_link_ui as ui


@pytest.fixture
def panel(inference_gui_root):
    app = tk.Toplevel(inference_gui_root)
    app.withdraw()
    app.theme, app.ui_font = 'dark', 'Segoe UI'
    page = ui.OwnerPanel(app)
    yield app, page
    page.destroy()
    app.destroy()


def labels(widget):
    result = []
    for child in widget.winfo_children():
        try:
            result.append(str(child.cget('text')))
        except tk.TclError:
            pass
        result += labels(child)
    return result


def test_gate_does_not_construct_communication_controls(panel):
    app, page = panel
    page._show_private()
    assert page.private is None
    assert not hasattr(page, 'notebook')
    assert not page.update_button.winfo_manager()
    assert not any('Codex' in text or 'Send' in text for text in labels(page))
    assert 'Back to node' in labels(page)
    assert not any('Codex' in text for child in app.winfo_children() if child != page for text in labels(child))


def test_verified_owner_page_is_removed_on_signout_and_expiry(panel, monkeypatch):
    _, page = panel
    closed = []
    runtime = SimpleNamespace(close=lambda: closed.append(True), session={'expires_at_epoch': time.time()+60})
    monkeypatch.setattr(ui, 'OwnerRuntime', lambda *args: runtime)
    monkeypatch.setattr(ui, 'save_session', lambda value: None)
    page.app.updater = None
    page.emit('authenticated', (page.auth_epoch, 'https://example.test', {'session_token': 'test', 'expires_at_epoch': time.time()+60}))
    page._drain()
    assert page.runtime is runtime
    assert page.private is not None
    assert str(page.update_button.cget('state')) == 'normal'
    assert any(page.notebook.tab(tab, 'text') == 'Node Channel' for tab in page.notebook.tabs())
    assert not any(text in labels(page) for text in ('New owner session', 'Refresh remote sessions', 'Choose local Codex sessions'))
    assert len(page.conversation_tabs) == 1
    page.emit('expired', None)
    page._drain()
    assert page.private is None
    assert page.runtime is None
    assert closed
    assert not page.update_button.winfo_manager()
    assert str(page.update_button.cget('state')) == 'disabled'
    assert not page.notebook.winfo_exists()
    page.emit('workspace', {'workspace': {}, 'histories': {}}, page.auth_epoch-1)
    page._drain()
    assert page.private is None


def test_workspace_sync_builds_channel_and_online_offline_dm_tabs(panel):
    _, page = panel
    page.runtime = SimpleNamespace()
    page._show_private()
    page._apply_workspace({
        'workspace': {
            'node_id': 'Johns_Dell_V2', 'channel_id': 'channel:fixture',
            'participants': [
                {'node_id': 'Johns_Dell_V2', 'online': True},
                {'node_id': 'ProVisWork', 'online': True},
                {'node_id': 'john-System-Product-Name', 'online': False}],
            'direct_conversations': {
                'ProVisWork': 'dm:work', 'john-System-Product-Name': 'dm:linux'}},
        'histories': {
            'channel:fixture': {'conversation_id': 'channel:fixture', 'messages': [{
                'id': 'message', 'source_node': 'Johns_Dell_V2', 'body': 'synthetic update',
                'recipients': [{'node_id': 'Johns_Dell_V2', 'status': 'executing',
                                'executing_node': 'Johns_Dell_V2', 'result': None}]}]},
            'dm:work': {'conversation_id': 'dm:work', 'messages': []},
            'dm:linux': {'conversation_id': 'dm:linux', 'messages': []}}})
    current = [page.notebook.tab(tab, 'text') for tab in page.notebook.tabs()]
    assert 'Node Channel' in current
    assert any('ProVisWork' in text and '●' in text for text in current)
    assert any('john-System-Product-Name' in text and '○' in text for text in current)
    assert 'executing on Johns_Dell_V2' in page.conversation_tabs['channel:fixture']['transcript'].get('1.0', 'end')
