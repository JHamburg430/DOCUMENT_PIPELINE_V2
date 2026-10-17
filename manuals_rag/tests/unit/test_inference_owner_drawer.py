import threading,time
import tkinter as tk
from types import SimpleNamespace
from manuals_rag_inference.owner_drawer import OwnerDrawer
from manuals_rag_inference.agent_link_ui import OwnerPanel

def test_drawer_only_appears_with_verified_owner_and_clears_on_expiry(inference_gui_root):
    app=tk.Toplevel(inference_gui_root);app.withdraw();app.theme='dark';app.ui_font='Segoe UI'
    app.main_body=tk.Frame(app);app.main_body.pack()
    app._position_panel=lambda:None
    drawer=app.owner_drawer=OwnerDrawer(app)
    page=app.owner_panel=OwnerPanel(app)
    try:
        assert not drawer.dock.winfo_manager()
        drawer.toggle();assert not drawer.expanded
        page.runtime=SimpleNamespace(stop_event=threading.Event(),session={'expires_at_epoch':time.time()+60},close=lambda:None)
        page._show_private();drawer.set_verified(True)
        assert drawer.dock.winfo_manager()=='pack'
        assert page.private.master is drawer.content
        assert not drawer.expanded
        drawer.toggle();assert drawer.expanded and drawer.viewport.winfo_manager()=='pack'
        drawer.toggle();assert not drawer.expanded and not drawer.viewport.winfo_manager()
        page.runtime.session['expires_at_epoch']=0
        drawer.toggle()
        assert page.runtime is None and page.private is None
        assert not drawer.dock.winfo_manager()
    finally:page.destroy();app.destroy()
