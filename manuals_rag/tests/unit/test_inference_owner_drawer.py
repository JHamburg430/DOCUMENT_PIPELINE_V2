import threading,time
import tkinter as tk
from types import SimpleNamespace
from manuals_rag_inference.owner_drawer import OwnerDrawer
from manuals_rag_inference.scrollable_body import ScrollableBody
from manuals_rag_inference.agent_link_ui import OwnerPanel

def test_drawer_only_appears_with_verified_owner_and_clears_on_expiry(inference_gui_root):
    app=tk.Toplevel(inference_gui_root);app.withdraw();app.theme='dark';app.ui_font='Segoe UI'
    app.body_scroll=ScrollableBody(app);app.body_scroll.pack(fill="both",expand=True)
    app.main_body=app.body_scroll.content
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

def test_all_owner_controls_are_reachable_at_short_window_height(inference_gui_root):
    from tkinter import ttk
    app=tk.Toplevel(inference_gui_root);app.theme='dark';app.ui_font='Segoe UI';app.geometry('760x700')
    tk.Label(app,text='PROVIS').pack(fill='x')
    app.body_scroll=ScrollableBody(app);app.body_scroll.pack(fill='both',expand=True)
    app.main_body=app.body_scroll.content
    public=tk.Frame(app.main_body,height=650);public.pack(fill='x');public.pack_propagate(False)
    app._position_panel=lambda:None
    drawer=app.owner_drawer=OwnerDrawer(app)
    page=app.owner_panel=OwnerPanel(app);page.withdraw()
    try:
        page.runtime=SimpleNamespace(stop_event=threading.Event(),session={'expires_at_epoch':time.time()+60},close=lambda:None)
        page._show_private();drawer.set_verified(True);drawer.toggle();app.update()
        canvas=app.body_scroll.canvas
        assert canvas.yview()[1]-canvas.yview()[0]<1
        canvas.yview_moveto(1);app.update()
        assert canvas.yview()[1]>.99
        assert page.notebook.winfo_rooty()+page.notebook.winfo_height()<=canvas.winfo_rooty()+canvas.winfo_height()+2
        send_button=next(iter(page.conversation_tabs.values()))['send']
        assert send_button.winfo_rootx()+send_button.winfo_width()<=canvas.winfo_rootx()+canvas.winfo_width()+2
        assert drawer.arrow.winfo_rooty()+drawer.arrow.winfo_height()<=app.winfo_rooty()+app.winfo_height()
        drawer.toggle();app.update()
        assert canvas.yview()[0]==0
    finally:page.destroy();app.destroy()
