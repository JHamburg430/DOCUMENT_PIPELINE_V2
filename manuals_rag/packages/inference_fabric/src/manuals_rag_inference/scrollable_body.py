"""One scroll region for all body content; header and owner arrow stay outside."""
import tkinter as tk
from tkinter import ttk
from .node_theme import PALETTES

class ScrollableBody(tk.Frame):
    def __init__(self, app):
        colors=PALETTES[app.theme]
        super().__init__(app, bg=colors['background'])
        self.canvas=tk.Canvas(self, highlightthickness=0, bg=colors['background'])
        self.scrollbar=ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content=tk.Frame(self.canvas, bg=colors['background'], padx=18, pady=14)
        self.content_id=self.canvas.create_window((0,0),window=self.content,anchor='nw')
        self.canvas.bind('<Configure>',self._layout)
        self.content.bind('<Configure>',self._layout)
        app.bind_all('<MouseWheel>',self._wheel,add='+')
        app.bind_all('<Button-4>',self._wheel,add='+')
        app.bind_all('<Button-5>',self._wheel,add='+')

    def _layout(self,event=None):
        width=max(1,self.canvas.winfo_width())
        height=max(self.content.winfo_reqheight(),self.canvas.winfo_height())
        self.canvas.itemconfigure(self.content_id,width=width,height=height)
        self.canvas.configure(scrollregion=(0,0,width,height))

    def _wheel(self,event):
        # Only this body, preserving native scrolling inside editable/reply text.
        if not str(event.widget).startswith(str(self)) or isinstance(event.widget,tk.Text):return
        delta=getattr(event,'delta',0)
        units=(-1 if delta>0 else 1) if delta else (-1 if getattr(event,'num',0)==4 else 1)
        self.canvas.yview_scroll(units*3,'units')
        return 'break'

    def reveal(self,widget):
        self._layout()
        height=max(1,self.content.winfo_reqheight())
        self.canvas.yview_moveto(max(0,widget.winfo_y()-8)/height)
