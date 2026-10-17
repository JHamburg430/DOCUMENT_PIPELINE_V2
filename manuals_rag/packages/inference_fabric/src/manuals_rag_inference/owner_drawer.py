"""Owner-only collapsible controls; visibility follows the verified owner runtime."""
import tkinter as tk
from tkinter import ttk
from .node_theme import PALETTES

class OwnerDrawer:
    def __init__(self, app):
        self.app = app
        self.verified = self.expanded = False
        colors = PALETTES[app.theme]
        self.dock = tk.Frame(app, bg=colors['background'])
        self.arrow = ttk.Button(self.dock, text='▴', width=3, command=self.toggle)
        self.arrow.pack(side='bottom', pady=(0, 4))
        self.viewport = tk.Frame(app.main_body, bg=colors['background'])
        self.content = tk.Frame(self.viewport, bg=colors['background'], pady=12)
        self.content.pack(fill='both', expand=True)

    def set_verified(self, verified):
        self.verified = bool(verified)
        if self.verified:
            self.dock.pack(side='bottom', fill='x', before=self.app.body_scroll)
        else:
            self.expanded = False
            self.viewport.pack_forget()
            self.dock.pack_forget()
            self.arrow.configure(text='▴')
            self.app._position_panel()

    def toggle(self):
        panel = getattr(self.app, 'owner_panel', None)
        runtime = getattr(panel, 'runtime', None)
        if not self.verified or runtime is None or runtime.stop_event.is_set(): return
        import time
        if time.time() >= runtime.session['expires_at_epoch']:
            panel._sign_out();return
        self.expanded = not self.expanded
        if self.expanded:
            self.viewport.pack(side='top', fill='x')
            self.app.after_idle(lambda: self.app.body_scroll.reveal(self.viewport))
        else:
            self.viewport.pack_forget()
            self.app.body_scroll.canvas.yview_moveto(0)
        self.arrow.configure(text='▾' if self.expanded else '▴')
        self.app._position_panel()
