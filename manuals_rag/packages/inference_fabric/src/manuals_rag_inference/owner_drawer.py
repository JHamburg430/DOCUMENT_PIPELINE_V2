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
        self.viewport = tk.Frame(self.dock, bg=colors['background'])
        self.canvas = tk.Canvas(self.viewport, height=min(400, app.winfo_screenheight()//2), highlightthickness=0, bg=colors['background'])
        scrollbar = ttk.Scrollbar(self.viewport, orient='vertical', command=self.canvas.yview)
        scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.content = tk.Frame(self.canvas, bg=colors['background'], padx=18, pady=10)
        self.content_id = self.canvas.create_window((0, 0), window=self.content, anchor='nw')
        self.content.bind('<Configure>', lambda event: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfigure(self.content_id, width=event.width))

    def set_verified(self, verified):
        self.verified = bool(verified)
        if self.verified:
            self.dock.pack(side='bottom', fill='x', before=self.app.main_body)
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
            self.viewport.pack(side='top', fill='both', expand=True)
        else:
            self.viewport.pack_forget()
        self.arrow.configure(text='▾' if self.expanded else '▴')
        self.app._position_panel()
