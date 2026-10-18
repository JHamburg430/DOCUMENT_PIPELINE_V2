"""Owner-scoped, persistent chat connector for dedicated local Codex threads."""
import threading
import time
import uuid

from .agent_link_client import call
from .agent_link_session import clear_session


class OwnerRuntime:
    def __init__(self, gateway, session, emit):
        self.gateway, self.session, self.emit = gateway, session, emit
        self.node_id = str(session['node_id'])
        self.agent_id = None
        self.stop_event = threading.Event()
        self.threads = []
        self.codex = None
        self.busy = threading.Event()
        self.external_busy = lambda: False
        self.delivery_lock = threading.Lock()
        self._workspace_lock = threading.Lock()
        self._last_workspace_sync = 0.0
        threading.Thread(target=self._poll, daemon=True, name='provis-owner-mailbox').start()

    def request(self, action, payload=None):
        if self.stop_event.is_set() or time.time() >= self.session['expires_at_epoch']:
            raise PermissionError('Owner sign-in expired')
        return call(self.gateway, self.session['session_token'], action, payload)

    def connect_local_codex(self):
        """Connect only to a dedicated PROVIS server, never desktop threads."""
        if self.codex is None:
            from .codex_link import CodexLink
            self.codex = CodexLink(mode='dedicated')
        return self.codex

    def _sync_workspace(self):
        if not self._workspace_lock.acquire(blocking=False):
            return
        try:
            codex = self.connect_local_codex()
            from .codex_owner_store import agent_identity
            self.agent_id = agent_identity()
            channel = codex.ensure_conversation('channel', 'PROVIS Node Channel')
            view = self.request('workspace')
            direct = []
            for participant in view['participants']:
                peer = participant['node_id']
                if peer == self.node_id:
                    continue
                session = codex.ensure_conversation('dm:' + peer, 'PROVIS DM with ' + peer[:100])
                direct.append({'peer_node_id': peer, 'thread_id': session['id']})
            self.request('register', {'agent_id':self.agent_id, 'channel_thread_id': channel['id'], 'direct_threads': direct})
            view = self.request('workspace')
            histories = {}
            for conversation_id in [view['channel_id'], *view['direct_conversations'].values()]:
                histories[conversation_id] = self.request('history', {'conversation_id': conversation_id})
            self.emit('workspace', {'workspace': view, 'histories': histories})
            self._last_workspace_sync = time.monotonic()
        finally:
            self._workspace_lock.release()

    def post(self, kind, target_node_id, message):
        payload = {'id': str(uuid.uuid4()), 'kind': kind, 'message': message}
        if kind == 'dm':
            payload['target_node_id'] = target_node_id
        return self.request('post', payload)

    def _poll(self):
        while not self.stop_event.is_set():
            try:
                if time.time() >= self.session['expires_at_epoch']:
                    self.emit('expired', None)
                    self.close()
                    return
                if time.monotonic() - self._last_workspace_sync >= 6:
                    self._sync_workspace()
                codex = self.connect_local_codex()
                ready = codex.ready_threads()
                if ready and not self.busy.is_set() and not self.external_busy():
                    job = self.request('poll', {'ready_thread_ids': ready}).get('job')
                    if job:
                        self.busy.set()
                        threading.Thread(target=self._deliver, args=(job,), daemon=True,
                                         name='provis-owner-delivery').start()
            except Exception as error:
                if isinstance(error, PermissionError) and 'expired' in str(error).lower():
                    self.emit('expired', None)
                    self.close()
                    return
                self.emit('status', 'Local Codex or owner workspace is reconnecting. Pending messages remain saved; no send is replayed automatically.')
            self.stop_event.wait(3)

    @staticmethod
    def _delivery_prompt(job):
        if job['kind'] == 'channel':
            addressed = ', '.join(job.get('recipient_snapshot') or [])
            instructions = (
                'PROVIS NODE CHANNEL MESSAGE. You are the agent for node '
                + str(job['target_node']) + ' (agent ' + str(job.get('target_agent_id') or 'unknown')
                + '). This message was addressed to this fixed recipient snapshot: '
                + addressed + '. Read the message and respond once only if it pertains to your node. '
                'If it does not pertain to your node, respond with exactly [NO_RESPONSE]. '
                'Never broadcast or forward an agent reply. Agent replies cannot trigger another channel message.\n\n'
            )
        else:
            instructions = (
                'PROVIS one-to-one message for node ' + str(job['target_node'])
                + ' (agent ' + str(job.get('target_agent_id') or 'unknown') + ')'
                + ' from node ' + str(job['source_node'])
                + '. Reply in this conversation only. Do not broadcast or create another message.\n\n'
            )
        return instructions + str(job['message'])

    def _deliver(self, job):
        try:
            self.request('executing', {'id': job['id'], 'claim': job['claim'], 'agent_id':job['target_agent_id']})
            result = self.codex.send(job['thread_id'], self._delivery_prompt(job))
            if result.get('status') == 'completed' and result.get('result') == '[NO_RESPONSE]':
                result = {'status': 'skipped', 'result': '[NO_RESPONSE]'}
            self.request('complete', {'id': job['id'], 'claim': job['claim'],
                                      'agent_id':job['target_agent_id'], **result})
        except Exception:
            self.emit('status', 'Delivery status is uncertain; its durable recipient record prevents automatic duplicate execution.')
        finally:
            self.busy.clear()
            self._last_workspace_sync = 0

    def close(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if self.codex:
            self.codex.close()
        clear_session()
        def logout():
            try:
                call(self.gateway, self.session['session_token'], 'logout')
            except Exception:
                pass
        threading.Thread(target=logout, daemon=True).start()
