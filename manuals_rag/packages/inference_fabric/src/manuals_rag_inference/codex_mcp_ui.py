"""Owner-only local MCP setup dialog, with explicit command-level approval."""
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from . import codex_mcp_setup as setup
from .node_theme import PALETTES


class CodexMcpPanel(tk.Toplevel):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.title('PROVIS • Add agent communication to Codex')
        self.geometry('700x410')
        self.minsize(520, 340)
        self.events = queue.Queue()
        self.busy = False
        self.plan = None
        self.checked = None
        colors = PALETTES[owner.app.theme]
        self.configure(bg=colors['background'])
        body = tk.Frame(self, bg=colors['background'], padx=18, pady=16)
        body.pack(fill='both', expand=True)
        tk.Label(body, text='Agent communication in Codex', font=(owner.app.ui_font, 16, 'bold'),
                 bg=colors['background'], fg=colors['text']).pack(anchor='w')
        tk.Label(body, text='Adds the local adapter for this Windows or Linux user. Your verified owner sign-in still controls access. No session is published or message sent by setup.',
                 bg=colors['background'], fg=colors['muted'], wraplength=650, justify='left').pack(fill='x', pady=10)
        self.command = tk.Text(body, height=5, wrap='word', bg=colors['card'], fg=colors['text'])
        self.command.pack(fill='both', expand=True)
        self.status = tk.StringVar(value='Check setup to inspect this PC’s Codex configuration.')
        tk.Label(body, textvariable=self.status, bg=colors['background'], fg=colors['muted'],
                 wraplength=650, justify='left').pack(fill='x', pady=12)
        actions = tk.Frame(body, bg=colors['background']);actions.pack(fill='x')
        self.check_button = ttk.Button(actions, text='Check setup', command=self.check)
        self.check_button.pack(side='left')
        self.add_button = ttk.Button(actions, text='Add to Codex', command=self.add, state='disabled')
        self.add_button.pack(side='right')
        self.copy_button = ttk.Button(actions, text='Copy command', command=self.copy, state='disabled')
        self.copy_button.pack(side='right', padx=8)
        try:
            self.plan = setup.setup_plan()
            self.command.insert('1.0', self.plan.display_command)
            self.copy_button.configure(state='normal')
        except setup.SetupError as error:
            self.status.set(str(error))
            self.command.insert('1.0', 'No command is available until the missing prerequisite is resolved.')
            self.check_button.configure(state='disabled')
        self.command.configure(state='disabled')
        self.after(100, self._drain)

    def owner_active(self):
        runtime = self.owner.runtime
        stopped = getattr(runtime, 'stop_event', None)
        return bool(runtime and not (stopped and stopped.is_set()) and
                    time.time() < runtime.session.get('expires_at_epoch', 0))

    def _work(self, operation):
        if self.busy or self.plan is None:
            return
        self.busy = True
        self.check_button.configure(state='disabled')
        self.add_button.configure(state='disabled')
        def worker():
            try:
                self.events.put(operation())
            except setup.SetupError as error:
                self.events.put({'state': 'error', 'message': str(error)})
            except Exception:
                self.events.put({'state': 'error', 'message': 'Setup could not complete. Check locally before retrying; diagnostic output was withheld.'})
        threading.Thread(target=worker, name='provis-mcp-setup', daemon=True).start()

    def check(self):
        def inspect():
            result = setup.configuration_status(self.plan)
            if result['state'] == 'configured':
                probe = setup.probe_adapter(self.plan)
                result['message'] += ' ' + probe['message']
            return result
        self._work(inspect)

    def add(self):
        if not self.plan or self.busy or not self.checked or not self.owner_active():
            self.status.set('Verify your owner account before adding the adapter.')
            return
        checked = self.checked
        replacing = checked['state'] == 'conflict'
        warning = ('This REPLACES the existing provis-agent-link adapter. ' if replacing else '')
        if not messagebox.askyesno('Allow Codex agent communication?', warning +
                'This changes the persistent provis-agent-link entry in your Codex configuration. '
                'Codex will be able to use the local communication tools while your owner sign-in is valid. '
                'It does not grant remote approvals, publish sessions or send messages.\n\n'
                'Exact command:\n' + self.plan.display_command + '\n\nAdd or repair this entry now?', parent=self):
            self.status.set('Cancelled. Codex configuration was not changed.')
            return
        self._work(lambda: setup.register(self.plan, approved=True, owner_active=self.owner_active,
                                         expected_revision=checked['revision'], replace_existing=replacing))

    def copy(self):
        if self.plan:
            self.clipboard_clear();self.clipboard_append(self.plan.display_command)

    def _drain(self):
        if not self.owner_active():
            self.destroy()
            return
        try:
            while True:
                result = self.events.get_nowait()
                self.checked = result if 'revision' in result else None
                self.busy = False
                self.status.set(result['message'])
                self.check_button.configure(state='normal')
                self.add_button.configure(text='Replace in Codex' if result['state'] == 'conflict' else 'Add to Codex',
                                          state='normal' if result['state'] in ('not_configured', 'needs_update', 'conflict') else 'disabled')
        except queue.Empty:
            pass
        self.after(100, self._drain)
