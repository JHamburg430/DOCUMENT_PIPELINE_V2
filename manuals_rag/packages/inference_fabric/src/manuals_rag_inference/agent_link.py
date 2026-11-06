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

from .node_channel_protocol import (
    CLAIM_TIMEOUT_SECONDS,
    HEARTBEAT_SECONDS,
    LEASE_SECONDS,
    PROTOCOL_ID,
)


class AgentLink:
    def __init__(self, validator, enrollment, *, owner_object_id, path, clock=time.time):
        self.validator, self.enrollment = validator, enrollment
        self.owner = str(uuid.UUID(owner_object_id))
        self.clock = clock
        self.lock = threading.RLock()
        self.sessions = {}
        self.targets = {}
        self.updates = None
        self.diagnostic = None  # separate explicit runtime opt-in
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        if str(path) != ':memory:': Path(path).chmod(0o600)
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, source TEXT, target TEXT, thread TEXT, message TEXT, status TEXT, result TEXT, created REAL, claimed REAL, claim_hash TEXT, message_hash TEXT)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS workspace_nodes (
            node_id TEXT PRIMARY KEY, workspace TEXT NOT NULL, channel_thread TEXT NOT NULL,
            direct_threads TEXT NOT NULL, agent_id TEXT NOT NULL DEFAULT '', owner_session TEXT NOT NULL, lease_until REAL NOT NULL,
            updated REAL NOT NULL, published_sessions TEXT NOT NULL DEFAULT '[]')''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, workspace TEXT NOT NULL, kind TEXT NOT NULL,
            left_node TEXT NOT NULL, right_node TEXT NOT NULL, created REAL NOT NULL)''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS conversation_messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, source_node TEXT NOT NULL,
            body TEXT NOT NULL, created REAL NOT NULL, body_hash TEXT NOT NULL)''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS conversation_recipients (
            message_id TEXT NOT NULL, node_id TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE,
            agent_id TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, executing_node TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '',
            updated REAL NOT NULL, PRIMARY KEY(message_id,node_id))''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS node_channel_audit (
            id TEXT PRIMARY KEY, event TEXT NOT NULL, actor_node TEXT NOT NULL,
            target_node TEXT NOT NULL, message_id TEXT NOT NULL, status TEXT NOT NULL,
            occurred REAL NOT NULL, detail_hash TEXT NOT NULL)''')
        # Forward-compatible additions for a mailbox created by an earlier
        # development build of this conversation branch.
        for table, column in (('workspace_nodes','agent_id'),('conversation_recipients','agent_id'),
                              ('workspace_nodes','published_sessions')):
            if column not in {row[1] for row in self.db.execute(f'PRAGMA table_info({table})')}:
                default = "'[]'" if column == 'published_sessions' else "''"
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT NOT NULL DEFAULT {default}")
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
        workspace = self._workspace(identity)
        with self.lock:
            self._cleanup()
            if len(self.sessions) >= 32:
                raise ValueError('Owner session limit reached')
            self.sessions[self._hash(token)] = (node_id, expiry, self._hash(record.secret), workspace)
        return {'session_token': token, 'expires_at_epoch': expiry, 'node_id': node_id,
                'node_channel_protocol': PROTOCOL_ID}

    @staticmethod
    def _hash(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def _cleanup(self):
        now = self.clock()
        self.sessions = {key: value for key, value in self.sessions.items() if value[1] > now}
        self.targets = {key: value for key, value in self.targets.items() if value['lease_until'] > now}
        expired_nodes = self.db.execute(
            'SELECT node_id FROM workspace_nodes WHERE lease_until>0 AND lease_until<=?',
            (now,),
        ).fetchall()
        for (node_id,) in expired_nodes:
            self._audit('lease_expired', node_id, node_id, status='expired')
        self.db.execute(
            'UPDATE workspace_nodes SET lease_until=-lease_until WHERE lease_until>0 AND lease_until<=?',
            (now,),
        )
        expired = [row[0] for row in self.db.execute(
            "SELECT id FROM jobs WHERE status IN ('claimed','executing') AND claimed<?",
            (now - CLAIM_TIMEOUT_SECONDS,),
        ).fetchall()]
        for identifier in expired:
            self.db.execute("UPDATE jobs SET status='uncertain', result='Delivery interrupted; inspect the target session before resending.' WHERE id=?", (identifier,))
            self.db.execute("UPDATE conversation_recipients SET status='uncertain',result='Delivery interrupted; inspect the target session before resending.',updated=? WHERE job_id=? AND status IN ('admitted','executing')", (now, identifier))
        # Keep queued work until its owner node reconnects. Completed and
        # uncertain delivery envelopes may age out while chat history remains.
        self.db.execute("DELETE FROM jobs WHERE created<? AND status NOT IN ('queued','claimed','executing')", (now - 86400,))
        self.db.commit()

    def _audit(self, event, actor_node, target_node='', message_id='', *, status='ok', detail=''):
        """Append a sanitized receipt; raw messages and credentials are never stored here."""
        self.db.execute(
            'INSERT INTO node_channel_audit VALUES(?,?,?,?,?,?,?,?)',
            (str(uuid.uuid4()), str(event)[:60], str(actor_node)[:120],
             str(target_node)[:120], str(message_id)[:80], str(status)[:40],
             self.clock(), self._hash(str(detail)) if detail else ''),
        )

    @staticmethod
    def _workspace(identity):
        raw = str(identity.get('tenant_id') or '') + '\0' + str(identity.get('subject') or '')
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _conversation_id(workspace, kind, source, target=''):
        if kind == 'channel':
            material = 'channel\0' + workspace
            prefix = 'channel'
        elif kind == 'dm':
            left, right = sorted((source, target))
            material = 'dm\0' + workspace + '\0' + left + '\0' + right
            prefix = 'dm'
        else:
            raise ValueError('Supported conversation required')
        return prefix + ':' + hashlib.sha256(material.encode()).hexdigest()

    @staticmethod
    def _node_id(value):
        if type(value) is not str or not 1 <= len(value) <= 120:
            raise ValueError('Bounded node identity required')
        return value

    @staticmethod
    def _thread_id(value):
        if type(value) is not str or not 1 <= len(value) <= 200:
            raise ValueError('Bounded local conversation identity required')
        return value

    def _enrolled_workspace_nodes(self, workspace):
        # Owner workspaces are tied to the Entra subject and tenant, not to
        # inference admission or the device's network status.
        sessions = {row[0]: row for row in self.db.execute(
            'SELECT node_id,workspace,channel_thread,direct_threads,agent_id,owner_session,lease_until,updated,published_sessions FROM workspace_nodes WHERE workspace=?',
            (workspace,)).fetchall()}
        enrolled = []
        for record in self.enrollment.nodes():
            if record.enrollment_method != 'entra':
                continue
            identity = {'subject': record.identity_subject, 'tenant_id': record.identity_tenant}
            if self._workspace(identity) != workspace:
                continue
            row = sessions.get(record.node_id)
            online = False
            if row and row[5]:
                active = self.sessions.get(row[5])
                online = bool(active and active[1] > self.clock() and active[0] == record.node_id and active[3] == workspace and row[6] > self.clock())
            enrolled.append({
                'node_id': record.node_id,
                'agent_id': row[4] if row and row[4] else None,
                'online': online,
                'current': online,
                'last_seen': row[7] if row else None,
                'lease_expires_at': max(0, row[6]) if row else None,
                'node_channel_protocol': PROTOCOL_ID if row else None,
                'codex_sessions': json.loads(row[8]) if row and online else [],
            })
        return sorted(enrolled, key=lambda item: item['node_id'].casefold())

    def _register(self, node_id, workspace, bearer, payload, expiry):
        agent_id = payload.get('agent_id')
        if type(agent_id) is not str or str(uuid.UUID(agent_id)) != agent_id:
            raise ValueError('Stable local agent identity required')
        channel_thread = self._thread_id(payload.get('channel_thread_id'))
        direct = payload.get('direct_threads', [])
        if type(direct) is not list or len(direct) > 60:
            raise ValueError('Bounded direct conversation registry required')
        normalized = {}
        for item in direct:
            if type(item) is not dict:
                raise ValueError('Direct conversation metadata required')
            peer = self._node_id(item.get('peer_node_id'))
            thread = self._thread_id(item.get('thread_id'))
            if peer == node_id:
                raise ValueError('Direct conversation must identify another node')
            normalized[peer] = thread
        allowed = {row['node_id'] for row in self._enrolled_workspace_nodes(workspace)}
        if set(normalized) - allowed:
            raise PermissionError('Direct conversation peer is outside this owner workspace')
        published = payload.get('published_sessions', [])
        if type(published) is not list or len(published) > 16:
            raise ValueError('Select at most 16 existing Codex sessions')
        published_ids = set()
        normalized_published = []
        for item in published:
            if (type(item) is not dict or set(item) != {'id', 'name'} or
                    type(item.get('id')) is not str or not 1 <= len(item['id']) <= 200 or
                    type(item.get('name')) is not str or not 1 <= len(item['name']) <= 160):
                raise ValueError('Bounded published session metadata required')
            if item['id'] == channel_thread or item['id'] in normalized.values() or item['id'] in published_ids:
                raise ValueError('Published session identities must be distinct')
            published_ids.add(item['id'])
            normalized_published.append({'id': item['id'], 'name': item['name']})
        lease_until = min(expiry, self.clock() + LEASE_SECONDS)
        packed = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        packed_published = json.dumps(normalized_published, sort_keys=True, separators=(',', ':'))
        owner_hash = bearer
        self.db.execute("UPDATE conversation_recipients SET agent_id=?,updated=? WHERE node_id=? AND status='queued'",
                        (agent_id, self.clock(), node_id))
        self.db.execute('''INSERT INTO workspace_nodes(node_id,workspace,channel_thread,direct_threads,agent_id,owner_session,lease_until,updated,published_sessions)
            VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(node_id) DO UPDATE SET workspace=excluded.workspace,
            channel_thread=excluded.channel_thread,direct_threads=excluded.direct_threads,
            agent_id=excluded.agent_id,owner_session=excluded.owner_session,lease_until=excluded.lease_until,
            updated=excluded.updated,published_sessions=excluded.published_sessions''',
            (node_id, workspace, channel_thread, packed, agent_id, owner_hash, lease_until, self.clock(), packed_published))
        for key in list(self.targets):
            if key[0] == node_id:
                del self.targets[key]
        self.targets[(node_id, channel_thread)] = {'node_id': node_id, 'thread_id': channel_thread,
                                                   'name': 'Node Channel', 'lease_until': lease_until,
                                                   'owner_session': owner_hash}
        for peer, thread in normalized.items():
            self.targets[(node_id, thread)] = {'node_id': node_id, 'thread_id': thread,
                                               'name': 'DM with ' + peer, 'lease_until': lease_until,
                                               'owner_session': owner_hash}
        for item in normalized_published:
            self.targets[(node_id, item['id'])] = {'node_id': node_id, 'thread_id': item['id'],
                                                  'name': item['name'], 'lease_until': lease_until,
                                                  'owner_session': owner_hash}
        # Bind pending deliveries to the recipient's stable local Codex
        # conversation as it becomes available, without changing their
        # original recipient snapshot.
        channel_rows = self.db.execute("SELECT id FROM jobs WHERE target=? AND thread='' AND status='queued' AND id IN (SELECT job_id FROM conversation_recipients WHERE status='queued' AND message_id IN (SELECT id FROM conversation_messages WHERE conversation_id LIKE 'channel:%'))", (node_id,)).fetchall()
        for (job_id,) in channel_rows:
            self.db.execute('UPDATE jobs SET thread=? WHERE id=?', (channel_thread, job_id))
        for peer, thread in normalized.items():
            conversation = self._conversation_id(workspace, 'dm', node_id, peer)
            self.db.execute('INSERT OR IGNORE INTO conversations VALUES(?,?,?,?,?,?)',
                            (conversation, workspace, 'dm', *sorted((node_id, peer)), self.clock()))
            rows = self.db.execute("SELECT j.id FROM jobs j JOIN conversation_recipients r ON r.job_id=j.id WHERE j.target=? AND j.thread='' AND j.status='queued' AND r.status='queued' AND r.message_id IN (SELECT id FROM conversation_messages WHERE conversation_id=?)", (node_id, conversation)).fetchall()
            for (job_id,) in rows:
                self.db.execute('UPDATE jobs SET thread=? WHERE id=?', (thread, job_id))
        self.db.commit()
        self._audit('registered', node_id, node_id, status='current')
        self.db.commit()
        return {'registered': True, 'node_id': node_id, 'lease_until': lease_until,
                'heartbeat_after_seconds': HEARTBEAT_SECONDS,
                'node_channel_protocol': PROTOCOL_ID}

    def _heartbeat(self, node_id, workspace, bearer, payload, expiry):
        if set(payload) != {'agent_id'}:
            raise ValueError('Exact heartbeat fields required')
        agent_id = payload.get('agent_id')
        row = self.db.execute(
            'SELECT agent_id,owner_session FROM workspace_nodes WHERE node_id=? AND workspace=?',
            (node_id, workspace),
        ).fetchone()
        if (row is None or row[0] != agent_id or row[1] != bearer or
                type(agent_id) is not str or str(uuid.UUID(agent_id)) != agent_id):
            raise PermissionError('Current registered node session required')
        lease_until = min(expiry, self.clock() + LEASE_SECONDS)
        self.db.execute(
            'UPDATE workspace_nodes SET lease_until=?,updated=? WHERE node_id=? AND workspace=? AND owner_session=?',
            (lease_until, self.clock(), node_id, workspace, bearer),
        )
        for target in self.targets.values():
            if target['node_id'] == node_id and target['owner_session'] == bearer:
                target['lease_until'] = lease_until
        self._audit('heartbeat', node_id, node_id, status='current')
        self.db.commit()
        return {'current': True, 'node_id': node_id, 'lease_until': lease_until,
                'heartbeat_after_seconds': HEARTBEAT_SECONDS,
                'node_channel_protocol': PROTOCOL_ID}

    def _unregister(self, node_id, workspace, bearer, *, reason='owner_stop'):
        changed = self.db.execute(
            'UPDATE workspace_nodes SET lease_until=0,updated=? WHERE node_id=? AND workspace=? AND owner_session=?',
            (self.clock(), node_id, workspace, bearer),
        ).rowcount
        for key in list(self.targets):
            if key[0] == node_id and self.targets[key]['owner_session'] == bearer:
                del self.targets[key]
        if changed:
            self._audit('unregistered', node_id, node_id, status='offline', detail=reason)
        self.db.commit()
        return {'unregistered': bool(changed), 'node_id': node_id,
                'node_channel_protocol': PROTOCOL_ID}

    def _workspace_view(self, node_id, workspace):
        participants = self._enrolled_workspace_nodes(workspace)
        rows = {row[0]: row for row in self.db.execute(
            'SELECT node_id,channel_thread,direct_threads,agent_id FROM workspace_nodes WHERE workspace=?', (workspace,)).fetchall()}
        for peer in participants:
            if peer['node_id'] == node_id:
                continue
            conversation = self._conversation_id(workspace, 'dm', node_id, peer['node_id'])
            self.db.execute('INSERT OR IGNORE INTO conversations VALUES(?,?,?,?,?,?)',
                            (conversation, workspace, 'dm', *sorted((node_id, peer['node_id'])), self.clock()))
        self.db.commit()
        channel_id = self._conversation_id(workspace, 'channel', node_id)
        self.db.execute('INSERT OR IGNORE INTO conversations VALUES(?,?,?,?,?,?)',
                        (channel_id, workspace, 'channel', '', '', self.clock()))
        self.db.commit()
        return {'node_id': node_id, 'participants': participants, 'channel_id': channel_id,
                'node_channel_protocol': PROTOCOL_ID,
                'heartbeat_seconds': HEARTBEAT_SECONDS, 'lease_seconds': LEASE_SECONDS,
                'direct_conversations': {peer['node_id']: self._conversation_id(workspace, 'dm', node_id, peer['node_id'])
                                         for peer in participants if peer['node_id'] != node_id}}

    def _post(self, node_id, workspace, payload):
        identifier = str(uuid.UUID(payload.get('id', '')))
        kind = payload.get('kind')
        target = self._node_id(payload.get('target_node_id')) if kind in ('channel', 'dm') else ''
        body = payload.get('message')
        target_session = payload.get('target_session_id')
        if target_session is not None and (type(target_session) is not str or not 1 <= len(target_session) <= 200):
            raise ValueError('Bounded target session identity required')
        if payload.get('origin') not in ('owner_ui', 'owner_mcp') or payload.get('hop_count') != 0:
            raise PermissionError('Only a fresh owner-authored, zero-hop message may be posted')
        if type(body) is not str or not body.strip() or len(body) > 12000:
            raise ValueError('Message must contain 1-12000 characters')
        stripped = body.strip()
        lowered = stripped.casefold()
        if stripped.startswith('@'):
            raise ValueError('Do not embed an address; select exactly one PC destination')
        if ('provis node channel message' in lowered or
                'forwarded agent-to-agent message via provis agent link' in lowered):
            raise PermissionError('Forwarded delivery envelopes cannot be posted back into Node Channel')
        if target == node_id:
            raise ValueError('A node cannot address itself')
        conversation = self._conversation_id(workspace, kind, node_id, target)
        if kind == 'channel':
            participants = {row['node_id']: row for row in self._enrolled_workspace_nodes(workspace)}
            if target not in participants or not participants[target]['online']:
                raise PermissionError('Addressed node must be connected in this owner workspace')
            recipients = [target]
        elif kind == 'dm':
            enrolled = {row['node_id'] for row in self._enrolled_workspace_nodes(workspace)}
            if target not in enrolled:
                raise PermissionError('Direct conversation peer is outside this owner workspace')
            recipients = [target]
        else:
            raise ValueError('Supported conversation required')
        if target_session is not None:
            target_row = next((row for row in self._enrolled_workspace_nodes(workspace)
                               if row['node_id'] == target), None)
            if not target_row or not target_row['online']:
                raise PermissionError('Target Codex session is not currently connected')
            if target_session not in {item['id'] for item in target_row['codex_sessions']}:
                raise PermissionError('Exact locally published Codex session required')
        # The visible address is part of every immutable agent-message
        # envelope as well as its routing metadata. A receiver can therefore
        # apply an exact, model-independent address check before invoking
        # Codex, regardless of which conversation view initiated the send.
        body = '@' + target + ': ' + stripped
        if not recipients:
            raise ValueError('No registered recipients in this conversation')
        if self.db.execute('SELECT count(*) FROM jobs').fetchone()[0] + len(set(recipients)) > 512:
            raise ValueError('Conversation delivery capacity reached')
        digest = self._hash(body)
        prior = self.db.execute('SELECT conversation_id,source_node,body_hash FROM conversation_messages WHERE id=?', (identifier,)).fetchone()
        if prior:
            if prior != (conversation, node_id, digest):
                raise PermissionError('Message identifier already used')
            return self._conversation_snapshot(conversation)
        endpoints = sorted((node_id, target)) if target else ('', '')
        self.db.execute('INSERT INTO conversations VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING',
                        (conversation, workspace, kind, *endpoints, self.clock()))
        self.db.execute('INSERT INTO conversation_messages VALUES(?,?,?,?,?,?)',
                        (identifier, conversation, node_id, body, self.clock(), digest))
        for recipient in sorted(set(recipients)):
            recipient_row = self.db.execute('SELECT channel_thread,direct_threads,agent_id FROM workspace_nodes WHERE node_id=? AND workspace=?', (recipient, workspace)).fetchone()
            thread = ''
            if recipient_row:
                if target_session is not None:
                    thread = target_session
                elif kind == 'channel':
                    thread = recipient_row[0]
                else:
                    thread = json.loads(recipient_row[1]).get(node_id, '')
            # Addressed messages have exactly one recipient, so the durable
            # delivery ID remains the caller-visible message ID.
            job_id = identifier
            self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (job_id, node_id, recipient, thread, body, 'queued', '', self.clock(), 0, '', digest))
            recipient_agent = recipient_row[2] if recipient_row else ''
            self.db.execute('INSERT INTO conversation_recipients(message_id,node_id,job_id,agent_id,status,updated) VALUES(?,?,?,?,?,?)',
                            (identifier, recipient, job_id, recipient_agent, 'queued', self.clock()))
        self.db.commit()
        self._audit('message_posted', node_id, target, identifier, status='queued', detail=body)
        self.db.commit()
        return self._conversation_snapshot(conversation)

    def _conversation_snapshot(self, conversation):
        messages = []
        rows = self.db.execute('SELECT id,source_node,body,created FROM conversation_messages WHERE conversation_id=? ORDER BY created,id', (conversation,)).fetchall()
        for message_id, source, body, created in rows:
            recipients = self.db.execute('SELECT node_id,agent_id,status,executing_node,result FROM conversation_recipients WHERE message_id=? ORDER BY node_id', (message_id,)).fetchall()
            messages.append({'id': message_id, 'source_node': source, 'body': body, 'created_at': created,
                             'recipients': [{'node_id': node, 'agent_id':agent or None, 'status': status,
                                             'executing_node': executing or None, 'result': result or None}
                                            for node, agent, status, executing, result in recipients]})
        return {'conversation_id': conversation, 'messages': messages}

    def _authorized_session(self, bearer):
        self._cleanup()
        session = self.sessions.get(self._hash(bearer))
        if session is None: raise PermissionError('Owner sign-in required')
        node_id, expiry, enrollment_pin, workspace = session
        record = next((node for node in self.enrollment.nodes() if node.node_id == node_id), None)
        if (record is None or self._hash(record.secret) != enrollment_pin or
                record.enrollment_method != 'entra' or
                self._workspace({'tenant_id':record.identity_tenant,'subject':record.identity_subject}) != workspace):
            raise PermissionError('Current enrollment required')
        return session

    def dispatch(self, action, bearer, payload, *, cancelled=None):
        if type(payload) is not dict:
            raise ValueError('Object required')
        if action == 'session':
            return self.session(bearer, payload)
        if action == 'diagnostic-inference':
            with self.lock:
                source, _, _, workspace = self._authorized_session(bearer)
                if self.diagnostic is None: raise PermissionError('Owner diagnostics disabled')
            result = self.diagnostic.run(workspace,source,payload,cancelled=cancelled)
            with self.lock:
                if self._authorized_session(bearer)[3] != workspace:
                    raise PermissionError('Owner workspace changed')
            return result
        with self.lock:
            node_id, expiry, enrollment_pin, workspace = self._authorized_session(bearer)
            if action in ('update-nodes', 'update-releases', 'request-update', 'update-status'):
                if self.updates is None: raise PermissionError('Remote updates are not configured')
                return self.updates.owner(action, node_id, payload)
            if action == 'register':
                return self._register(node_id, workspace, self._hash(bearer), payload, expiry)
            if action == 'heartbeat':
                return self._heartbeat(node_id, workspace, self._hash(bearer), payload, expiry)
            if action == 'unregister':
                if payload and set(payload) != {'reason'}:
                    raise ValueError('Exact unregister fields required')
                reason = payload.get('reason', 'owner_stop')
                if reason not in ('owner_stop', 'owner_logout', 'runtime_replace'):
                    raise ValueError('Supported unregister reason required')
                return self._unregister(node_id, workspace, self._hash(bearer), reason=reason)
            if action == 'workspace':
                return self._workspace_view(node_id, workspace)
            if action == 'post':
                return self._post(node_id, workspace, payload)
            if action == 'history':
                conversation = payload.get('conversation_id')
                row = self.db.execute('SELECT workspace,kind,left_node,right_node FROM conversations WHERE id=?', (conversation,)).fetchone()
                if not row or row[0] != workspace or (row[1] == 'dm' and node_id not in row[2:]):
                    raise PermissionError('Conversation outside this owner workspace')
                return self._conversation_snapshot(conversation)
            if action == 'audit':
                if payload:
                    raise ValueError('Audit listing takes no fields')
                rows = self.db.execute(
                    'SELECT id,event,actor_node,target_node,message_id,status,occurred,detail_hash '
                    'FROM node_channel_audit WHERE actor_node=? OR target_node=? '
                    'ORDER BY occurred DESC,id DESC LIMIT 100',
                    (node_id, node_id),
                ).fetchall()
                return {'node_channel_protocol': PROTOCOL_ID, 'receipts': [dict(zip(
                    ('id', 'event', 'actor_node', 'target_node', 'message_id',
                     'status', 'occurred_at', 'detail_hash'), row)) for row in rows]}
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
                    self.db.execute("UPDATE conversation_recipients SET status='admitted', updated=? WHERE job_id=? AND status='queued'", (self.clock(), identifier))
                    self._audit('delivery_claimed', node_id, node_id, identifier, status='admitted')
                    self.db.commit()
                    row = self.db.execute('SELECT message_id,agent_id FROM conversation_recipients WHERE job_id=?', (identifier,)).fetchone()
                    message = self.db.execute('SELECT conversation_id,source_node,body FROM conversation_messages WHERE id=?', (row[0],)).fetchone() if row else None
                    return {'job': {'id': identifier, 'message_id': row[0] if row else identifier,
                                    'conversation_id': message[0] if message else '', 'source_node': message[1] if message else '',
                                    'kind': self.db.execute('SELECT kind FROM conversations WHERE id=?', (message[0],)).fetchone()[0] if message else 'dm',
                                    'thread_id': thread, 'target_node': node_id, 'message': message[2] if message else message,
                                    'target_agent_id': row[1] if row else '',
                                    'recipient_snapshot': [r[0] for r in self.db.execute('SELECT node_id FROM conversation_recipients WHERE message_id=? ORDER BY node_id', (row[0],)).fetchall()] if row else [],
                                    'claim': claim}}
                return {'job': None}
            if action == 'executing':
                identifier, claim = payload.get('id'), payload.get('claim')
                row = self.db.execute('SELECT target,status,claim_hash FROM jobs WHERE id=?', (identifier,)).fetchone()
                if row is None or row[0] != node_id or row[1] != 'claimed' or not secrets.compare_digest(row[2], self._hash(claim or '')):
                    raise PermissionError('Exact delivery claim required')
                recipient = self.db.execute('SELECT agent_id,status FROM conversation_recipients WHERE job_id=?', (identifier,)).fetchone()
                if recipient is None or recipient[0] != payload.get('agent_id') or recipient[1] != 'admitted':
                    raise PermissionError('Exact local agent identity required')
                self.db.execute("UPDATE jobs SET status='executing' WHERE id=?", (identifier,))
                self.db.execute("UPDATE conversation_recipients SET status='executing',executing_node=?,updated=? WHERE job_id=? AND status='admitted'", (node_id, self.clock(), identifier))
                self._audit('delivery_executing', node_id, node_id, identifier, status='executing')
                self.db.commit()
                return {'id': identifier, 'status': 'executing', 'executing_node': node_id}
            if action == 'complete':
                identifier, claim, status, result = (payload.get(key) for key in ('id', 'claim', 'status', 'result'))
                if type(claim) is not str or status not in ('completed', 'blocked', 'failed', 'uncertain', 'skipped') or type(result) is not str or len(result) > 20000:
                    raise ValueError('Bounded completion required')
                row = self.db.execute('SELECT target,status,claim_hash FROM jobs WHERE id=?', (identifier,)).fetchone()
                if row is None or row[0] != node_id or row[1] not in ('claimed', 'executing') or not secrets.compare_digest(row[2], self._hash(claim)):
                    raise PermissionError('Exact delivery claim required')
                recipient = self.db.execute('SELECT agent_id,status FROM conversation_recipients WHERE job_id=?', (identifier,)).fetchone()
                if recipient is not None and (recipient[0] != payload.get('agent_id') or recipient[1] not in ('admitted','executing')):
                    raise PermissionError('Exact local agent identity required')
                public_status = 'replied' if status == 'completed' else status
                self.db.execute('UPDATE jobs SET status=?,result=?,message=? WHERE id=?', (public_status, result, '', identifier))
                self.db.execute('UPDATE conversation_recipients SET status=?,result=?,executing_node=?,updated=? WHERE job_id=? AND status IN (\'admitted\',\'executing\')',
                                (public_status, result, node_id, self.clock(), identifier))
                self._audit('delivery_completed', node_id, node_id, identifier,
                            status=public_status, detail=result)
                self.db.commit()
                return {'id': identifier, 'status': public_status}
            if action == 'messages':
                rows = self.db.execute('SELECT id,source,target,thread,status,result,created FROM jobs WHERE source=? OR target=? ORDER BY created DESC LIMIT 30', (node_id, node_id)).fetchall()
                messages = []
                for row in rows:
                    candidate = dict(zip(('id', 'source_node', 'target_node', 'thread_id', 'status', 'result', 'created_at'), row))
                    if len(json.dumps({'messages': messages + [candidate]}).encode()) > 700000: break
                    messages.append(candidate)
                return {'messages': messages}
            if action == 'logout':
                self._unregister(node_id, workspace, self._hash(bearer), reason='owner_logout')
                self.sessions.pop(self._hash(bearer), None)
                self.db.commit()
                return {'signed_out': True, 'node_channel_protocol': PROTOCOL_ID}
            raise ValueError('Unsupported owner operation')
