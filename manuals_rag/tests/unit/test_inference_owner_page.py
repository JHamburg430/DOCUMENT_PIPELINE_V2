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
        try: result.append(str(child.cget('text')))
        except tk.TclError: pass
        result += labels(child)
    return result

def test_gate_does_not_construct_communication_controls(panel):
    app, page = panel
    page._show_private()
    assert page.private is None
    assert not hasattr(page, 'message')
    assert not hasattr(page, 'targets')
    assert not page.update_button.winfo_manager()
    assert not any('Codex' in text or 'Send message' in text for text in labels(page))
    assert 'Back to node' in labels(page)
    assert not any('Codex' in text for child in app.winfo_children() if child != page for text in labels(child))

def test_verified_owner_page_is_removed_on_signout_and_expiry(panel, monkeypatch):
    _, page = panel
    closed = []
    runtime = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(ui, 'OwnerRuntime', lambda *args: runtime)
    monkeypatch.setattr(ui, 'save_session', lambda value: None)
    monkeypatch.setattr(page, '_refresh', lambda: None)
    page.app.updater = None
    page.emit('authenticated', (page.auth_epoch, 'https://example.test', {'session_token': 'test', 'expires_at_epoch': time.time()+60}))
    page._drain()
    assert page.runtime is runtime
    assert page.private is not None
    assert 'Send message' in labels(page)
    page.message.insert('1.0', 'private fixture')
    page.emit('expired', None)
    page._drain()
    assert page.private is None
    assert page.runtime is None
    assert closed
    assert not page.update_button.winfo_manager()
    assert 'Send message' not in labels(page)
    # Events arriving from the revoked capability cannot repaint its messages.
    page.emit('messages', [{'id':'old'}], page.auth_epoch-1)
    page._drain()
    assert page.private is None
