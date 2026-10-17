"""Owner-only agent mailbox, isolated from inference credentials and routing.

Explicitly published sessions only. Claims are never automatically redelivered:
an interrupted delivery becomes uncertain, avoiding duplicate agent actions.
"""
import hashlib
import json
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class AgentLink:
    def __init__(self, validator, enrollment, *, owner_object_id, path, clock=time.time):
        self.validator, self.enrollment = validator, enrollment
        self.owner = str(uuid.UUID(owner_object_id))
        self.clock = clock
        self.lock = threading.RLock()
        self.sessions = {}
        self.targets = {}
        self.updates = None
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        if str(path) != ':memory:': Path(path).chmod(0o600)
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, source TEXT, target TEXT, thread TEXT, message TEXT, status TEXT, result TEXT, created REAL, claimed REAL, claim_hash TEXT, message_hash TEXT)')
        self.db.commit()

    def session(self, bearer, payload):
        identity = self.validator.validate(bearer)
        if identity.get('object_id') != self.owner:
            raise PermissionError('Owner account required')
        node_id = payload.get('node_id')
        record = next((node for node in self.enrollment.nodes() if node.node_id == node_id), None)
        if record is None or record.enrollment_method != 'entra' or record.identity_subject != identity['subject'] or record.identity_tenant != identity['tenant_id']:
            raise PermissionError('Owned Entra-enrolled node required')
        expiry = min(float(identity.get('expires_at_epoch', 0)), self.clock() + 3600)
        if expiry <= self.clock():
            raise PermissionError('Fresh owner sign-in required')
        token = secrets.token_urlsafe(48)
        with self.lock:
            self._cleanup()
            if len(self.sessions) >= 32:
                raise ValueError('Owner session limit reached')
            self.sessions[self._hash(token)] = (node_id, expiry, self._hash(record.secret))
        return {'session_token': token, 'expires_at_epoch': expiry, 'node_id': node_id}

    @staticmethod
    def _hash(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def _cleanup(self):
        now = self.clock()
        self.sessions = {key: value for key, value in self.sessions.items() if value[1] > now}
        self.targets = {key: value for key, value in self.targets.items() if value['lease_until'] > now}
        self.db.execute("UPDATE jobs SET status='uncertain', result='Delivery interrupted; inspect the target session before resending.' WHERE status='claimed' AND claimed<?", (now - 600,))
        self.db.execute('DELETE FROM jobs WHERE created<?', (now - 86400,))
        self.db.commit()

    def dispatch(self, action, bearer, payload):
        if type(payload) is not dict:
            raise ValueError('Object required')
        if action == 'session':
            return self.session(bearer, payload)
        with self.lock:
            self._cleanup()
            session = self.sessions.get(self._hash(bearer))
            if session is None:
                raise PermissionError('Owner sign-in required')
            node_id, expiry, enrollment_pin = session
            record = next((node for node in self.enrollment.nodes() if node.node_id == node_id), None)
            if record is None or self._hash(record.secret) != enrollment_pin:
                raise PermissionError('Current enrollment required')
            if action in ('update-nodes', 'update-releases', 'request-update', 'update-status'):
                if self.updates is None: raise PermissionError('Remote updates are not configured')
                return self.updates.owner(action, node_id, payload)
            if action == 'publish':
                threads = payload.get('threads')
                if type(threads) is not list or len(threads) > 16:
                    raise ValueError('Select at most 16 sessions')
                validated = []
                for item in threads:
                    if type(item) is not dict or type(item.get('id')) is not str or not 1 <= len(item['id']) <= 200 or type(item.get('name')) is not str or len(item['name']) > 160:
                        raise ValueError('Bounded session metadata required')
                    validated.append((item['id'], item['name']))
                for key in list(self.targets):
                    if key[0] == node_id: del self.targets[key]
                for thread, name in validated:
                    self.targets[(node_id, thread)] = {'node_id': node_id, 'thread_id': thread, 'name': name, 'lease_until': min(expiry, self.clock() + 30), 'owner_session': self._hash(bearer)}
                return {'published': len(validated)}
            if action == 'targets':
                return {'targets': [{key: row[key] for key in ('node_id', 'thread_id', 'name')} for row in self.targets.values()]}
            if action == 'send':
                identifier = str(uuid.UUID(payload.get('id', '')))
                target, thread, message = payload.get('node_id'), payload.get('thread_id'), payload.get('message')
                if type(message) is not str or not message.strip() or len(message) > 12000:
                    raise ValueError('Message must contain 1–12000 characters')
                if type(target) is not str or type(thread) is not str or (target, thread) not in self.targets:
                    raise PermissionError('Published owner session required')
                existing = self.db.execute('SELECT source,target,thread,message_hash,status FROM jobs WHERE id=?', (identifier,)).fetchone()
                if existing:
                    if existing[:4] != (node_id, target, thread, self._hash(message)):
                        raise PermissionError('Message identifier already used')
                    return {'id': identifier, 'status': existing[4]}
                if self.db.execute('SELECT count(*) FROM jobs').fetchone()[0] >= 256:
                    raise ValueError('Message capacity reached')
                self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)', (identifier, node_id, target, thread, message, 'queued', '', self.clock(), 0, '', self._hash(message)))
                self.db.commit()
                return {'id': identifier, 'status': 'queued'}
            if action == 'poll':
                ready = payload.get('ready_thread_ids')
                if ready is not None and (type(ready) is not list or len(ready) > 16 or any(type(item) is not str for item in ready)):
                    raise ValueError('Bounded ready-session list required')
                rows = self.db.execute("SELECT id,thread,message FROM jobs WHERE target=? AND status='queued' ORDER BY created LIMIT 256", (node_id,)).fetchall()
                for identifier, thread, message in rows:
                    if (node_id, thread) not in self.targets: continue
                    if ready is not None and thread not in ready: continue
                    claim = secrets.token_urlsafe(32)
                    self.db.execute("UPDATE jobs SET status='claimed', claimed=?, claim_hash=? WHERE id=? AND status='queued'", (self.clock(), self._hash(claim), identifier))
                    self.db.commit()
                    return {'job': {'id': identifier, 'thread_id': thread, 'message': message, 'claim': claim}}
                return {'job': None}
            if action == 'complete':
                identifier, claim, status, result = (payload.get(key) for key in ('id', 'claim', 'status', 'result'))
                if type(claim) is not str or status not in ('completed', 'blocked', 'failed', 'uncertain') or type(result) is not str or len(result) > 20000:
                    raise ValueError('Bounded completion required')
                row = self.db.execute('SELECT target,status,claim_hash FROM jobs WHERE id=?', (identifier,)).fetchone()
                if row is None or row[0] != node_id or row[1] != 'claimed' or not secrets.compare_digest(row[2], self._hash(claim)):
                    raise PermissionError('Exact delivery claim required')
                self.db.execute('UPDATE jobs SET status=?,result=?,message=? WHERE id=?', (status, result, '', identifier))
                self.db.commit()
                return {'id': identifier, 'status': status}
            if action == 'messages':
                rows = self.db.execute('SELECT id,source,target,thread,status,result,created FROM jobs WHERE source=? OR target=? ORDER BY created DESC LIMIT 30', (node_id, node_id)).fetchall()
                messages = []
                for row in rows:
                    candidate = dict(zip(('id', 'source_node', 'target_node', 'thread_id', 'status', 'result', 'created_at'), row))
                    if len(json.dumps({'messages': messages + [candidate]}).encode()) > 700000: break
                    messages.append(candidate)
                return {'messages': messages}
            if action == 'logout':
                self.sessions.pop(self._hash(bearer), None)
                for key in list(self.targets):
                    if key[0] == node_id and self.targets[key]['owner_session'] == self._hash(bearer): del self.targets[key]
                return {'signed_out': True}
            raise ValueError('Unsupported owner operation')
