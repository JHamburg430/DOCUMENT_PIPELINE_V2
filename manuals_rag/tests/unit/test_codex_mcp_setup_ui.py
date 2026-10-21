import threading,time,tkinter as tk
from types import SimpleNamespace

import pytest

from manuals_rag_inference import codex_mcp_ui as ui
from manuals_rag_inference.codex_mcp_setup import SetupPlan


@pytest.fixture
def panel(inference_gui_root,monkeypatch,tmp_path):
    owner=tk.Toplevel(inference_gui_root);owner.withdraw()
    owner.app=SimpleNamespace(theme='dark',ui_font='Segoe UI')
    owner.runtime=SimpleNamespace(session={'expires_at_epoch':time.time()+60},stop_event=threading.Event())
    plan=SetupPlan('codex',('adapter',),str(tmp_path/'owner.json'))
    monkeypatch.setattr(ui.setup,'setup_plan',lambda:plan)
    page=ui.CodexMcpPanel(owner)
    yield owner,page
    if page.winfo_exists():page.destroy()
    owner.destroy()


def test_setup_shows_exact_command_and_does_not_modify_on_open(panel,monkeypatch):
    _,page=panel
    assert page.command.get('1.0','end-1c')==page.plan.display_command
    assert str(page.add_button.cget('state'))=='disabled'
    assert page.checked is None


def test_cancelled_action_cannot_start_registration(panel,monkeypatch):
    _,page=panel;page.checked={'state':'not_configured','revision':'absent'}
    prompts=[];operations=[]
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda title,text,**kw:prompts.append(text) or False)
    monkeypatch.setattr(page,'_work',lambda action:operations.append(action))
    page.add()
    assert page.plan.display_command in prompts[0]
    assert 'persistent' in prompts[0]
    assert not operations


def test_expired_owner_never_reaches_approval_or_registration(panel,monkeypatch):
    owner,page=panel;page.checked={'state':'not_configured','revision':'absent'}
    owner.runtime.session['expires_at_epoch']=time.time()-1
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda *a,**kw:pytest.fail('No prompt after expiry'))
    page.add()
    assert 'Verify your owner account' in page.status.get()


def test_explicit_replace_passes_checked_revision_after_approval(panel,monkeypatch):
    _,page=panel;page.checked={'state':'conflict','revision':'checked-revision'}
    prompts=[];received=[]
    monkeypatch.setattr(ui.messagebox,'askyesno',lambda title,text,**kw:prompts.append(text) or True)
    monkeypatch.setattr(ui.setup,'register',lambda plan,**kw:received.append(kw) or {'state':'configured','message':'Ready'})
    monkeypatch.setattr(page,'_work',lambda action:action())
    page.add()
    assert 'REPLACES' in prompts[0]
    assert received[0]['approved'] is True
    assert received[0]['replace_existing'] is True
    assert received[0]['expected_revision']=='checked-revision'


def test_signout_closes_setup_panel(panel):
    owner,page=panel;owner.runtime=None
    page._drain()
    assert not page.winfo_exists()
