"""Owner connector: existing proxy or explicitly selected dedicated local server.

Dedicated mode publishes only sessions created through its explicit UI action.
Model, approval and sandbox defaults are not overridden.
No shell commands from a remote message are executed by this connector.
"""
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path


def codex_executable():
    candidates = [shutil.which('codex')]
    if os.name == 'nt':
        candidates.insert(0, str(Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/OpenAI/Codex/bin/codex.exe'))
    for value in candidates:
        if value and Path(value).is_file(): return value
    raise RuntimeError('Codex is not installed for this user')


class CodexLink:
    def __init__(self, mode=None):
        from .codex_owner_store import server_mode, ServerLock
        self.mode = server_mode() if mode is None else mode
        if self.mode not in ('proxy', 'dedicated'):
            raise ValueError('Explicit supported Codex connection mode required')
        self.server_lock = None
        self.process = None
        self.lock = threading.RLock()
        self.pending = {}
        self.counter = 0
        self.turns = {}
        self.allowed = set()
        command = [codex_executable(), 'app-server', '--stdio' if self.mode == 'dedicated' else 'proxy']
        socket_path = os.getenv('PROVIS_CODEX_CONTROL_SOCKET', '').strip()
        if socket_path and self.mode == 'proxy': command += ['--sock', socket_path]
        try:
            if self.mode == 'dedicated': self.server_lock = ServerLock()
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except Exception:
            self.close()
            raise
        threading.Thread(target=self._read, daemon=True, name='provis-codex-link').start()
        try:
            self.rpc('initialize', {'clientInfo': {'name': 'provis_agent_link', 'title': 'PROVIS Owner Agent Link', 'version': '1.0.0'}})
            self._write({'method': 'initialized', 'params': {}})
        except Exception:
            self.close()
            raise

    def _write(self, value):
        raw = (json.dumps(value, separators=(',', ':')) + '\n').encode()
        if len(raw) > 65536: raise ValueError('Bounded Codex request required')
        with self.lock:
            self.process.stdin.write(raw);self.process.stdin.flush()

    def rpc(self, method, params, timeout=15):
        with self.lock:
            self.counter += 1;identifier = self.counter
            response = queue.Queue(maxsize=1);self.pending[identifier] = response
        try:
            self._write({'id': identifier, 'method': method, 'params': params})
            result = response.get(timeout=timeout)
            if 'error' in result: raise RuntimeError('Codex request was not accepted')
            return result['result']
        finally:
            with self.lock: self.pending.pop(identifier, None)

    def _read(self):
        try:
            while True:
                raw = self.process.stdout.readline(2_000_001)
                if not raw or len(raw) > 2_000_000: break
                value = json.loads(raw)
                if 'method' in value and 'id' in value:
                    with self.lock:
                        owned = (value.get('params') or {}).get('threadId') in self.turns
                    if not owned: continue
                    # Never approve anything remotely or change the target's safeguards.
                    self._write({'id': value['id'], 'error': {'code': -32000, 'message': 'Complete approvals in the local Codex app.'}})
                    with self.lock:
                        state = self.turns[(value.get('params') or {})['threadId']]
                        turn_id = (value.get('params') or {}).get('turnId')
                        if turn_id:
                            state['blocked_turns'].add(turn_id)
                    continue
                if 'id' in value:
                    with self.lock: pending = self.pending.get(value['id'])
                    if pending is not None: pending.put_nowait(value)
                    continue
                params = value.get('params') or {}
                thread = params.get('threadId')
                with self.lock:
                    state = self.turns.get(thread)
                    if state is None: continue
                    turn_id = params.get('turnId') or (params.get('turn') or {}).get('id')
                    if not isinstance(turn_id, str): continue
                    if turn_id not in state['events'] and len(state['events']) >= 4: continue
                    turn_event = state['events'].setdefault(turn_id, {'text': '', 'status': 'failed', 'done': False})
                    if value.get('method') == 'item/completed':
                        item = params.get('item') or {}
                        if item.get('type') == 'agentMessage':
                            turn_event['text'] = str(item.get('text') or '')[:20000]
                    if value.get('method') == 'turn/completed':
                        turn_event['status'] = (params.get('turn') or {}).get('status', 'failed')
                        turn_event['done'] = True
                    state['changed'].notify_all()
        except Exception:
            pass
        finally:
            with self.lock:
                for pending in self.pending.values():
                    if pending.empty(): pending.put_nowait({'error': 'connection_closed'})
                for state in self.turns.values():
                    state['connection_closed'] = True
                    state['changed'].notify_all()

    def sessions(self):
        if getattr(self, 'mode', 'proxy') == 'dedicated':
            from .codex_owner_store import owned_sessions
            return [{'id': identifier, 'name': name} for identifier, name in owned_sessions().items()]
        result = self.rpc('thread/list', {'limit': 60, 'archived': False, 'sourceKinds': ['cli', 'vscode', 'appServer', 'exec', 'unknown']})
        return [{'id': str(row['id']), 'name': str(row.get('name') or row.get('id'))[:160]} for row in result.get('data', [])]

    def create_session(self, name='PROVIS owner session'):
        if self.mode != 'dedicated':
            raise PermissionError('Dedicated owner server required')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
            raise ValueError('Bounded session name required')
        from .codex_owner_store import root, owned_sessions, add_session
        if len(owned_sessions()) >= 60:
            raise ValueError('Owner session capacity reached')
        workspace = root() / 'owner-workspace'
        workspace.mkdir(mode=0o700, exist_ok=True)
        # Start an empty owned session, preserving the user's normal model and
        # approval defaults. No inference or forwarded message is sent here.
        result = self.rpc('thread/start', {'cwd': str(workspace)})
        identifier = result['thread']['id']
        add_session(identifier, name.strip())
        return {'id': identifier, 'name': name.strip()}

    def publish(self, threads):
        available = {row['id'] for row in self.sessions()}
        if not set(threads).issubset(available): raise PermissionError('Select an existing local session')
        self.allowed = set(threads)

    def ready_threads(self):
        ready = []
        for thread in tuple(self.allowed):
            metadata = self.rpc('thread/read', {'threadId': thread})['thread']
            if (metadata.get('status') or {}).get('type') in ('idle', 'notLoaded'):
                ready.append(thread)
        return ready

    def send(self, thread, message):
        if thread not in self.allowed: raise PermissionError('Session not published locally')
        metadata = self.rpc('thread/read', {'threadId': thread})['thread']
        if (metadata.get('status') or {}).get('type') == 'active':
            return {'status': 'blocked', 'result': 'Target session is busy; no message was injected.'}
        self.rpc('thread/resume', {'threadId': thread, 'excludeTurns': True})
        state = {'events': {}, 'blocked_turns': set(), 'changed': threading.Condition(self.lock)}
        with self.lock: self.turns[thread] = state
        try:
            # User-visible input, not a system/developer instruction or hidden authority.
            forwarded = 'Forwarded agent-to-agent message via PROVIS Agent Link. Existing permissions and approval requirements still apply.\n\n' + message
            started = self.rpc('turn/start', {'threadId': thread, 'input': [{'type': 'text', 'text': forwarded, 'text_elements': []}]})
            turn_id = started['turn']['id']
            deadline = time.monotonic() + 540
            with self.lock:
                while not state['events'].get(turn_id, {}).get('done'):
                    if state.get('connection_closed'):
                        return {'status': 'uncertain', 'result': 'The control connection closed. Inspect the target session before resending.'}
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {'status': 'uncertain', 'result': 'No completion received. Inspect the target session before resending.'}
                    state['changed'].wait(remaining)
                completed = state['events'][turn_id]
            if turn_id in state['blocked_turns']:
                return {'status': 'blocked', 'result': 'The target requires a local approval. No remote approval was granted.'}
            return {'status': 'completed' if completed['status'] == 'completed' else 'failed',
                    'result': completed['text'] or 'The session ended without a text reply.'}
        except Exception:
            return {'status': 'uncertain', 'result': 'Delivery could not be confirmed. Inspect the target session before resending.'}
        finally:
            with self.lock: self.turns.pop(thread, None)

    def close(self):
        # Stop only the connector-owned proxy/dedicated child, never the desktop.
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=5)
        if getattr(self, 'server_lock', None) is not None:
            self.server_lock.close(); self.server_lock = None
