"""Hidden owner surface. Rendering this window never grants server privileges."""
import queue
import threading
import tkinter as tk
from tkinter import ttk
import uuid
import webbrowser

from .agent_link_client import call
from .agent_link_session import save_session
from .agent_link_runtime import OwnerRuntime
from .entra_client import request_device_code, poll_device_token, fetch_enrollment_config
from .node_theme import PALETTES


class OwnerPanel(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title('PROVIS • Owner Agent Link')
        self.geometry('720x690')
        self.protocol('WM_DELETE_WINDOW', self.withdraw)
        self.events = queue.Queue(maxsize=30)
        self.runtime = None
        self.auth_epoch = 0
        self.code = ''
        colors = PALETTES[app.theme]
        self.configure(bg=colors['background'])
        body = tk.Frame(self, bg=colors['background'], padx=18, pady=16)
        body.pack(fill='both', expand=True)
        tk.Label(body, text='Owner Agent Link', bg=colors['background'], fg=colors['text'], font=(app.ui_font, 18, 'bold')).pack(anchor='w')
        tk.Label(body, text='Only explicitly published sessions are reachable. Existing agent approvals stay in force.', wraplength=660, justify='left', bg=colors['background'], fg=colors['muted']).pack(anchor='w', pady=(4, 12))
        login = tk.Frame(body, bg=colors['background']);login.pack(fill='x')
        self.signin = ttk.Button(login, text='Verify Microsoft account', command=self._sign_in)
        self.signin.pack(side='left')
        ttk.Button(login, text='Copy code', command=self._copy).pack(side='left', padx=8)
        ttk.Button(login, text='Sign out owner link', command=self._sign_out).pack(side='right')
        self.status = tk.StringVar(value='Sign in to verify owner access. This does not change inference enrollment.')
        tk.Label(body, textvariable=self.status, wraplength=660, justify='left', bg=colors['background'], fg=colors['muted']).pack(anchor='w', pady=10)
        controls = tk.Frame(body, bg=colors['background']);controls.pack(fill='x')
        self.local_button = ttk.Button(controls, text='Choose local Codex sessions', command=self._choose, state='disabled');self.local_button.pack(side='left')
        self.refresh = ttk.Button(controls, text='Refresh remote sessions', command=self._refresh, state='disabled');self.refresh.pack(side='right')
        self.targets = ttk.Treeview(body, columns=('node', 'session'), show='headings', height=5, style='Node.Treeview')
        self.targets.heading('node', text='PC / node');self.targets.heading('session', text='Published session')
        self.targets.column('node', width=180);self.targets.column('session', width=460)
        self.targets.pack(fill='x', pady=10)
        tk.Label(body, text='Message to selected session', bg=colors['background'], fg=colors['text']).pack(anchor='w')
        self.message = tk.Text(body, height=4, wrap='word', bg=colors['card'], fg=colors['text'], insertbackground=colors['text'], font=(app.ui_font, 10));self.message.pack(fill='x', pady=5)
        self.send_button = ttk.Button(body, text='Send message', command=self._send, state='disabled');self.send_button.pack(anchor='e')
        tk.Label(body, text='Delivery and replies', bg=colors['background'], fg=colors['text']).pack(anchor='w', pady=(10, 4))
        self.replies = tk.Text(body, height=9, wrap='word', bg=colors['card'], fg=colors['text'], font=(app.ui_font, 10), state='disabled');self.replies.pack(fill='both', expand=True)
        self.target_rows = {}
        self.after(200, self._drain)

    def emit(self, kind, value, epoch=None):
        try: self.events.put_nowait((self.auth_epoch if epoch is None else epoch, kind, value))
        except queue.Full: pass

    def _work(self, operation, success='status'):
        epoch = self.auth_epoch
        def worker():
            try: self.emit(success, operation(), epoch)
            except Exception: self.emit('status', 'Owner operation unavailable or not authorized. No message is automatically resent.', epoch)
        threading.Thread(target=worker, daemon=True).start()

    def _sign_in(self):
        self.auth_epoch += 1
        epoch = self.auth_epoch
        self.signin.configure(state='disabled')
        self.status.set('Starting owner identity verification…')
        def authenticate():
            values = self.app._resolve_values(self.app._values())
            if not call(values['gateway_url'], '', 'availability').get('enabled'):
                self.emit('status', 'Owner link is staged but is not enabled on the gateway yet.', epoch)
                return None
            device = request_device_code(tenant_id=values['entra_tenant_id'], client_id=values['entra_client_id'], scope=values['entra_scope'])
            self.emit('code', str(device['user_code']), epoch)
            url = device.get('verification_uri_complete') or device.get('verification_uri')
            if url: webbrowser.open(url)
            token = poll_device_token(tenant_id=values['entra_tenant_id'], client_id=values['entra_client_id'], device_code=device['device_code'], interval=int(device.get('interval', 5)), expires_in=int(device.get('expires_in', 900)))
            session = call(values['gateway_url'], token['access_token'], 'session', {'node_id': values['node_id']})
            return (epoch, values['gateway_url'], session)
        self._work(authenticate, 'authenticated')

    def _copy(self):
        if self.code:
            self.clipboard_clear();self.clipboard_append(self.code)

    def _sign_out(self):
        self.auth_epoch += 1
        if self.runtime: self.runtime.close();self.runtime = None
        self.code = ''
        self.status.set('Owner link signed out.')
        self.signin.configure(state='normal')
        for button in (self.local_button, self.refresh, self.send_button): button.configure(state='disabled')
        for item in self.targets.get_children(): self.targets.delete(item)
        self.replies.configure(state='normal');self.replies.delete('1.0', 'end');self.replies.configure(state='disabled')

    def _choose(self):
        if self.runtime: self._work(self.runtime.local_sessions, 'local_sessions')

    def _choose_rows(self, rows):
        dialog = tk.Toplevel(self);dialog.title('Publish local Codex sessions');dialog.geometry('640x380')
        tree = ttk.Treeview(dialog, columns=('session',), show='headings', selectmode='extended', style='Node.Treeview')
        tree.heading('session', text='Select only sessions you want this link to reach')
        tree.pack(fill='both', expand=True)
        indexed = {}
        for row in rows: indexed[tree.insert('', 'end', values=(row['name'],))] = row
        def publish():
            selected = [indexed[item] for item in tree.selection()]
            if self.runtime:
                self._work(lambda: (self.runtime.publish(selected), 'Local sessions published.')[1])
            dialog.destroy()
        ttk.Button(dialog, text='Publish selected sessions', command=publish).pack(pady=10)

    def _refresh(self):
        if self.runtime: self._work(lambda: self.runtime.request('targets')['targets'], 'targets')

    def _send(self):
        selected = self.targets.selection()
        text = self.message.get('1.0', 'end-1c').strip()
        if not self.runtime or len(selected) != 1 or not text: return
        row = self.target_rows[selected[0]]
        payload = {'id': str(uuid.uuid4()), 'node_id': row['node_id'], 'thread_id': row['thread_id'], 'message': text}
        self.send_button.configure(state='disabled')
        self.status.set('Sending '+payload['id']+'…')
        def send():
            result = self.runtime.request('send', payload)
            return 'Delivery '+result['id']+': '+result['status']
        self._work(send, 'sent')

    def _drain(self):
        try:
            while True:
                epoch, kind, value = self.events.get_nowait()
                if epoch != self.auth_epoch:
                    if kind == 'authenticated' and value:
                        _, gateway, session = value
                        threading.Thread(target=lambda: call(gateway, session['session_token'], 'logout'), daemon=True).start()
                    continue
                if kind in ('status', 'sent'):
                    self.status.set(str(value))
                    self.signin.configure(state='normal' if self.runtime is None else 'disabled')
                    if self.runtime: self.send_button.configure(state='normal')
                elif kind == 'code': self.code = value;self.status.set(value)
                elif kind == 'authenticated':
                    if value is None:
                        self.signin.configure(state='normal')
                        continue
                    epoch, gateway, session = value
                    if epoch != self.auth_epoch:
                        threading.Thread(target=lambda: call(gateway, session['session_token'], 'logout'), daemon=True).start()
                        continue
                    if self.runtime: self.runtime.close()
                    save_session({**session, 'gateway_url': gateway})
                    self.runtime = OwnerRuntime(gateway, session, lambda kind, value, epoch=epoch: self.emit(kind, value, epoch))
                    self.code = '';self.status.set('Owner verified. Choose local sessions to publish, or refresh remote sessions.')
                    for button in (self.local_button, self.refresh, self.send_button): button.configure(state='normal')
                    self._refresh()
                elif kind == 'local_sessions': self._choose_rows(value)
                elif kind == 'targets':
                    for item in self.targets.get_children(): self.targets.delete(item)
                    self.target_rows = {}
                    for row in value: self.target_rows[self.targets.insert('', 'end', values=(row['node_id'], row['name']))] = row
                elif kind == 'messages':
                    rendered = '\n\n'.join(row['id']+' • '+row['status']+' • '+row['target_node']+'\n'+row['result'] for row in value)
                    self.replies.configure(state='normal');self.replies.delete('1.0', 'end');self.replies.insert('1.0', rendered);self.replies.configure(state='disabled')
        except queue.Empty: pass
        self.after(200, self._drain)

    def stop(self):
        self.auth_epoch += 1
        if self.runtime: self.runtime.close();self.runtime = None
