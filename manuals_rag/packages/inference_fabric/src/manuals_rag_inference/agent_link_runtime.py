"""Explicitly enabled owner connector. No autonomous outbound agent messages."""
import threading
import time
from .agent_link_client import call
from .agent_link_session import clear_session


class OwnerRuntime:
    def __init__(self, gateway, session, emit):
        self.gateway, self.session, self.emit = gateway, session, emit
        self.stop_event = threading.Event()
        self.threads = []
        self.codex = None
        self.busy = threading.Event()
        threading.Thread(target=self._poll, daemon=True, name='provis-owner-mailbox').start()

    def request(self, action, payload=None):
        if self.stop_event.is_set() or time.time() >= self.session['expires_at_epoch']:
            raise PermissionError('Owner sign-in expired')
        return call(self.gateway, self.session['session_token'], action, payload)

    def local_sessions(self):
        if self.codex is None:
            from .codex_link import CodexLink
            self.codex = CodexLink()
        return self.codex.sessions()

    def publish(self, rows):
        if self.codex is None: self.local_sessions()
        self.codex.publish([row['id'] for row in rows])
        self.request('publish', {'threads': rows})
        self.threads = list(rows)

    def _poll(self):
        heartbeat = 0
        while not self.stop_event.is_set():
            try:
                if time.time() >= self.session['expires_at_epoch']:
                    self.emit('status', 'Owner sign-in expired. Sign in again to reconnect agent sessions.')
                    self.close();return
                if time.monotonic() - heartbeat > 10:
                    if self.threads: self.request('publish', {'threads': list(self.threads)})
                    self.emit('messages', self.request('messages')['messages'])
                    heartbeat = time.monotonic()
                if self.threads and not self.busy.is_set():
                    ready = self.codex.ready_threads()
                    job = self.request('poll', {'ready_thread_ids': ready}).get('job') if ready else None
                    if job:
                        self.busy.set()
                        threading.Thread(target=self._deliver, args=(job,), daemon=True, name='provis-owner-delivery').start()
            except Exception:
                self.emit('status', 'Owner link unavailable; retrying connection. No message is resent automatically.')
            self.stop_event.wait(3)

    def _deliver(self, job):
        try:
            result = self.codex.send(job['thread_id'], job['message'])
            self.request('complete', {'id': job['id'], 'claim': job['claim'], **result})
        except Exception:
            self.emit('status', 'Delivery is uncertain. Inspect the target session before resending.')
        finally: self.busy.clear()

    def close(self):
        if self.stop_event.is_set(): return
        self.stop_event.set()
        if self.codex: self.codex.close()
        clear_session()
        def logout():
            try: call(self.gateway, self.session['session_token'], 'logout')
            except Exception: pass
        threading.Thread(target=logout, daemon=True).start()
