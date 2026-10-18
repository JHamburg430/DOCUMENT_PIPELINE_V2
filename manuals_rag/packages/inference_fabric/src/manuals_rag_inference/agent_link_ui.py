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
        self.title('PROVIS • Private workspace')
        self.geometry('720x690')
        self.protocol('WM_DELETE_WINDOW', self._back_to_node)
        self.events = queue.Queue(maxsize=30)
        self.runtime = None
        self.auth_epoch = 0
        self.code = ''
        colors = PALETTES[app.theme]
        self.configure(bg=colors['background'])
        body = tk.Frame(self, bg=colors['background'], padx=18, pady=16)
        body.pack(fill='both', expand=True)
        ttk.Button(body, text='Back to node', command=self._back_to_node).pack(anchor='w', pady=(0, 12))
        tk.Label(body, text='Private workspace', bg=colors['background'], fg=colors['text'], font=(app.ui_font, 18, 'bold')).pack(anchor='w')
        tk.Label(body, text='Verify your Microsoft account to open your private workspace.', wraplength=660, justify='left', bg=colors['background'], fg=colors['muted']).pack(anchor='w', pady=(4, 12))
        login = tk.Frame(body, bg=colors['background']);login.pack(fill='x')
        self.signin = ttk.Button(login, text='Verify Microsoft account', command=self._sign_in)
        self.signin.pack(side='left')
        ttk.Button(login, text='Copy code', command=self._copy).pack(side='left', padx=8)
        ttk.Button(login, text='Sign out owner link', command=self._sign_out).pack(side='right')
        self.update_button = ttk.Button(login, text='App updates', command=self._updates, state='disabled')
        # Private actions are not displayed before server-verified owner sign-in.
        self.update_dialog = None
        self.status = tk.StringVar(value='Sign in to verify owner access. This does not change inference enrollment.')
        tk.Label(body, textvariable=self.status, wraplength=660, justify='left', bg=colors['background'], fg=colors['muted']).pack(anchor='w', pady=10)
        self.body = body
        self.private = None
        self.conversation_tabs = {}
        self.workspace_view = None
        self.after(200, self._drain)

    def _back_to_node(self):
        self.withdraw()
        self.app.deiconify()
        self.app.lift()

    def _show_private(self):
        if not self.runtime or self.private is not None: return
        colors = PALETTES[self.app.theme]
        host = self.app.owner_drawer.content if hasattr(self.app, 'owner_drawer') else self.body
        private = self.private = tk.Frame(host, bg=colors['background'])
        private.pack(fill='both', expand=True)
        if hasattr(self.app, 'owner_drawer'):
            actions = tk.Frame(private, bg=colors['background']);actions.pack(fill='x', pady=(0, 8))
            self.update_button = ttk.Button(actions, text='App updates', command=self._updates, state='disabled');self.update_button.pack(side='left')
            ttk.Button(actions, text='Sign out owner link', command=self._sign_out).pack(side='right')
            tk.Label(private, textvariable=self.status, bg=colors['background'], fg=colors['muted'], wraplength=680, justify='left').pack(anchor='w', pady=(0, 8))
        else:
            self.update_button.pack(side='left')
        tk.Label(private, text='Agent communication', bg=colors['background'], fg=colors['text'], font=(self.app.ui_font, 15, 'bold')).pack(anchor='w', pady=(0, 6))
        tk.Label(private, text='The Node Channel reaches the fixed workspace snapshot for each post. Direct conversations keep their own history.', bg=colors['background'], fg=colors['muted'], wraplength=680, justify='left').pack(anchor='w', pady=(0, 8))
        self.notebook = ttk.Notebook(private)
        self.notebook.pack(fill='both', expand=True)
        self.conversation_tabs = {}
        self.workspace_view = None
        self._add_conversation_tab('channel:pending', 'Node Channel', 'channel', None)

    def _add_conversation_tab(self, conversation_id, title, kind, peer):
        if conversation_id in self.conversation_tabs:
            return self.conversation_tabs[conversation_id]
        colors = PALETTES[self.app.theme]
        page = tk.Frame(self.notebook, bg=colors['background'], padx=8, pady=8)
        transcript = tk.Text(page, height=12, wrap='word', bg=colors['card'], fg=colors['text'],
                             insertbackground=colors['text'], font=(self.app.ui_font, 10), state='disabled')
        transcript.pack(fill='both', expand=True)
        status = tk.StringVar(value='Connecting to local Codex and workspace…')
        tk.Label(page, textvariable=status, bg=colors['background'], fg=colors['muted'], anchor='w').pack(fill='x', pady=(5, 2))
        composer = tk.Text(page, height=3, wrap='word', bg=colors['card'], fg=colors['text'],
                           insertbackground=colors['text'], font=(self.app.ui_font, 10))
        composer.pack(fill='x')
        send = ttk.Button(page, text='Send', command=lambda key=conversation_id: self._send_conversation(key), state='disabled')
        send.pack(anchor='e', pady=(5, 0))
        self.notebook.add(page, text=title)
        row = {'page': page, 'transcript': transcript, 'status': status, 'composer': composer,
               'send': send, 'kind': kind, 'peer': peer, 'title': title}
        self.conversation_tabs[conversation_id] = row
        return row

    def _apply_workspace(self, payload):
        view = payload.get('workspace') or {}
        self.workspace_view = view
        channel_id = view.get('channel_id')
        if channel_id:
            row = self.conversation_tabs.pop('channel:pending', None)
            if row:
                self.notebook.forget(row['page'])
                self.conversation_tabs[channel_id] = row
                row['title'] = 'Node Channel'
                self.notebook.add(row['page'], text='Node Channel')
        own_node = view.get('node_id')
        participants = {item['node_id']: item for item in view.get('participants') or []}
        for peer, conversation_id in (view.get('direct_conversations') or {}).items():
            if peer == own_node:
                continue
            participant = participants.get(peer, {'online': False})
            label = ('● ' if participant.get('online') else '○ ') + peer
            row = self.conversation_tabs.get(conversation_id)
            if row is None:
                row = self._add_conversation_tab(conversation_id, label, 'dm', peer)
            else:
                row['title'] = label
                self.notebook.tab(row['page'], text=label)
            agent = participant.get('agent_id')
            agent_label = (' · agent ' + agent[:8]) if agent else ''
            row['status'].set(('Online' if participant.get('online') else 'Offline') + ' · node ' + peer + agent_label)
        histories = payload.get('histories') or {}
        for conversation_id, history in histories.items():
            self._render_history(conversation_id, history)
        for conversation_id, row in self.conversation_tabs.items():
            row['send'].configure(state='normal' if self.runtime and conversation_id != 'channel:pending' else 'disabled')
        self.status.set('Local Codex connected. Workspace conversations are synchronized automatically.')

    def _render_history(self, conversation_id, history):
        row = self.conversation_tabs.get(conversation_id)
        if not row:
            return
        transcript = row['transcript']
        messages = []
        for message in history.get('messages') or []:
            messages.append(f"[{message['source_node']}] {message['body']}")
            for recipient in message.get('recipients') or []:
                state = recipient.get('status', 'queued')
                executor = recipient.get('executing_node')
                agent = recipient.get('agent_id')
                detail = f" · agent {agent[:8]} executing on {executor}" if executor and agent else (f" · executing on {executor}" if executor else '')
                messages.append(f"  {recipient['node_id']}: {state}{detail}")
                result = recipient.get('result')
                if result and result != '[NO_RESPONSE]':
                    messages.append(f"  ↳ {recipient['node_id']}: {result}")
        transcript.configure(state='normal')
        transcript.delete('1.0', 'end')
        transcript.insert('1.0', '\n\n'.join(messages) or 'No messages yet.')
        transcript.configure(state='disabled')

    def _send_conversation(self, conversation_id):
        row = self.conversation_tabs.get(conversation_id)
        if not self.runtime or not row:
            return
        message = row['composer'].get('1.0', 'end-1c').strip()
        if not message:
            return
        row['send'].configure(state='disabled')
        row['status'].set('Queueing message…')
        kind, peer = row['kind'], row['peer']
        self._work(lambda: self.runtime.post(kind, peer, message), 'posted')
        row['composer'].delete('1.0', 'end')

    def _hide_private(self):
        if self.update_button.winfo_exists(): self.update_button.pack_forget()
        if hasattr(self.app, 'owner_drawer'): self.app.owner_drawer.set_verified(False)
        if self.private is not None:
            self.private.destroy()
            self.private = None
        self.conversation_tabs = {}
        self.workspace_view = None

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
        if self.update_dialog and self.update_dialog.winfo_exists(): self.update_dialog.destroy()
        self.update_dialog = None
        if self.runtime: self.runtime.close();self.runtime = None
        self.code = ''
        self.status.set('Owner link signed out.')
        self.signin.configure(state='normal')
        self._hide_private()

    def _updates(self):
        if not self.runtime: return
        if self.update_dialog and self.update_dialog.winfo_exists(): self.update_dialog.lift();return
        self.update_dialog = UpdatePanel(self)

    def _drain(self):
        try:
            while True:
                epoch, kind, value = self.events.get_nowait()
                if epoch != self.auth_epoch:
                    if kind == 'authenticated' and value:
                        _, gateway, session = value
                        threading.Thread(target=lambda: call(gateway, session['session_token'], 'logout'), daemon=True).start()
                    continue
                if kind == 'expired':
                    self._sign_out()
                    self.status.set('Owner access expired. Verify your Microsoft account again.')
                elif kind == 'status':
                    self.status.set(str(value))
                    self.signin.configure(state='normal' if self.runtime is None else 'disabled')
                elif kind == 'posted':
                    self._render_history(value.get('conversation_id'), value)
                    for row in self.conversation_tabs.values():
                        row['send'].configure(state='normal')
                        row['status'].set('Message saved; recipient delivery is tracked below.')
                elif kind == 'workspace':
                    self._apply_workspace(value)
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
                    self.runtime.external_busy = lambda: bool(self.app.updater and self.app.updater.applying)
                    self._show_private()
                    self.code = '';self.status.set('Owner verified. Connecting to local Codex and discovering workspace nodes.')
                    if hasattr(self.app, 'owner_drawer'):
                        self.app.owner_drawer.set_verified(True)
                        self.withdraw()
                        self.app.deiconify();self.app.lift()
                elif kind == 'update_data':
                    if self.update_dialog and self.update_dialog.winfo_exists(): self.update_dialog.render(value)
                elif kind == 'update_requested':
                    self.status.set('App update '+value['id']+': '+value['status'])
                    if self.update_dialog: self.update_dialog.refresh()
        except queue.Empty: pass
        self.after(200, self._drain)

    def stop(self):
        self._hide_private()
        self.auth_epoch += 1
        if self.runtime: self.runtime.close();self.runtime = None


class UpdatePanel(tk.Toplevel):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.title('PROVIS • Remote app updates')
        self.geometry('830x620')
        colors=PALETTES[owner.app.theme]
        self.configure(bg=colors['background'])
        tk.Label(self,text='Node app updates',bg=colors['background'],fg=colors['text'],font=(owner.app.ui_font,18,'bold')).pack(anchor='w',padx=18,pady=12)
        tk.Label(self,text='Signed PROVIS releases only. Active work finishes before restart. The previous app stays available for recovery.',wraplength=790,justify='left',bg=colors['background'],fg=colors['muted']).pack(anchor='w',padx=18)
        self.nodes=ttk.Treeview(self,columns=('node','version','platform','enabled'),show='headings',height=5,style='Node.Treeview',selectmode='browse')
        for key,label in zip(('node','version','platform','enabled'),('PC / node','Current version','Platform','Updater ready')): self.nodes.heading(key,text=label);self.nodes.column(key,width=190)
        self.nodes.pack(fill='x',padx=18,pady=10)
        self.releases=ttk.Treeview(self,columns=('version','platform','publisher'),show='headings',height=4,style='Node.Treeview',selectmode='browse')
        for key in ('version','platform','publisher'): self.releases.heading(key,text=key.title())
        self.releases.pack(fill='x',padx=18,pady=8)
        row=tk.Frame(self,bg=colors['background']);row.pack(fill='x',padx=18)
        ttk.Button(row,text='Refresh',command=self.refresh).pack(side='left')
        ttk.Button(row,text='Update selected PC',command=self.send).pack(side='right')
        self.history=tk.Text(self,height=8,state='disabled',bg=colors['card'],fg=colors['text'],wrap='word')
        self.history.pack(fill='both',expand=True,padx=18,pady=12)
        self.node_rows={};self.release_rows={}
        self.refresh()
        self.after(10000,self._tick)
    def _tick(self):
        if self.winfo_exists():
            self.refresh();self.after(10000,self._tick)
    def refresh(self):
        runtime=self.owner.runtime
        if runtime:
            self.owner._work(lambda:(runtime.request('update-nodes'),runtime.request('update-releases'),runtime.request('update-status')),'update_data')
    def render(self,data):
        nodes,releases,status=data
        for tree in (self.nodes,self.releases):
            for item in tree.get_children(): tree.delete(item)
        self.node_rows={};self.release_rows={}
        for node in nodes['nodes']:
            key=self.nodes.insert('', 'end',values=(node['node_id'],node['version'] or 'Bootstrap needed',node['platform'],'Yes' if node['updates_enabled'] else 'No'))
            self.node_rows[key]=node
        for release in releases['releases']:
            m=release['manifest'];key=self.releases.insert('', 'end',values=(m['version'],m['platform'],release['publisher']))
            self.release_rows[key]=m
        self.history.configure(state='normal');self.history.delete('1.0','end')
        self.history.insert('1.0','\n'.join(row['node_id']+' • '+row['status']+' • '+row['detail'] for row in status['updates']) or 'No app updates requested.')
        self.history.configure(state='disabled')
    def send(self):
        from tkinter import messagebox
        a,b=self.nodes.selection(),self.releases.selection()
        if not self.owner.runtime or len(a)!=1 or len(b)!=1: return
        node,release=self.node_rows[a[0]],self.release_rows[b[0]]
        if not node['updates_enabled'] or node['platform']!=release['platform']:
            messagebox.showinfo('App update','Select a compatible PC with a trusted updater bootstrap.',parent=self);return
        if not messagebox.askyesno('Update node app', 'Update '+node['node_id']+' from '+node['version']+' to '+release['version']+'? Active work finishes before the app restarts.',parent=self): return
        payload={'id':str(uuid.uuid4()),'node_id':node['node_id'],'release_id':release['release_id']}
        runtime=self.owner.runtime
        self.owner._work(lambda:runtime.request('request-update',payload),'update_requested')
