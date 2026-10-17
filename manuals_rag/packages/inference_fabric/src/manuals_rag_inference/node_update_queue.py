"""Owner-authorized update requests; node credentials may only claim/report self."""
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
import uuid
from .node_release import verify_release

TERMINAL = ('healthy', 'rolled_back', 'failed', 'interrupted')
STAGES = ('claimed', 'downloading', 'verified', 'waiting_for_idle', 'restarting')


class NodeUpdateQueue:
    def __init__(self, catalog, enrollment, scheduler, db, lock, clock=time.time):
        self.catalog, self.enrollment, self.scheduler = catalog, enrollment, scheduler
        self.db, self.lock, self.clock = db, lock, clock
        self.db.execute('CREATE TABLE IF NOT EXISTS app_updates (id TEXT PRIMARY KEY, source TEXT, target TEXT, release TEXT, status TEXT, detail TEXT, created REAL, changed REAL, claim_hash TEXT)')
        self.db.commit()

    def inventory(self):
        snapshots = {node.node_id: node for node in self.scheduler.snapshots()}
        return [{'node_id': record.node_id, 'version': snapshots[record.node_id].node_app_version if record.node_id in snapshots else '',
                 'platform': snapshots[record.node_id].node_app_platform if record.node_id in snapshots else '',
                 'updates_enabled': snapshots[record.node_id].remote_updates_enabled if record.node_id in snapshots else False,
                 'last_seen': snapshots[record.node_id].observed_at_epoch if record.node_id in snapshots else 0} for record in self.enrollment.nodes()]

    def owner(self, action, source_node, payload):
        with self.lock:
            if action == 'update-nodes': return {'nodes': self.inventory()}
            if action == 'update-releases': return {'releases': self.catalog.list()}
            if action == 'request-update':
                identifier = str(uuid.UUID(payload.get('id', '')))
                target, release = payload.get('node_id'), payload.get('release_id')
                document, _ = self.catalog.get(release)
                node = next((node for node in self.inventory() if node['node_id'] == target), None)
                if node is None or node['updates_enabled'] is not True or self.clock()-node['last_seen'] > 30:
                    raise PermissionError('Node needs a trusted updater bootstrap first')
                old = self.db.execute('SELECT source,target,release,status FROM app_updates WHERE id=?', (identifier,)).fetchone()
                if old:
                    if old[:3] != (source_node, target, release): raise PermissionError('Update identifier already used')
                    return {'id': identifier, 'status': old[3]}
                verify_release(document, self.catalog.keys, platform=node['platform'], current_version=node['version'])
                active = self.db.execute("SELECT id FROM app_updates WHERE target=? AND status NOT IN ('healthy','rolled_back','failed','interrupted')", (target,)).fetchone()
                if active: raise ValueError('An update is already pending for this node')
                if self.db.execute('SELECT count(*) FROM app_updates').fetchone()[0] >= 256:
                    raise ValueError('Update history capacity reached')
                now = self.clock()
                self.db.execute('INSERT INTO app_updates VALUES (?,?,?,?,?,?,?,?,?)', (identifier, source_node, target, release, 'queued', '', now, now, ''))
                self.db.commit()
                return {'id': identifier, 'status': 'queued'}
            if action == 'update-status':
                self.db.execute("UPDATE app_updates SET status='interrupted',detail='No completion received; inspect the node before retrying.' WHERE status IN ('claimed','downloading','verified','waiting_for_idle','restarting') AND changed<?", (self.clock()-1200,))
                self.db.commit()
                rows = self.db.execute('SELECT id,target,release,status,detail,changed FROM app_updates ORDER BY created DESC LIMIT 64').fetchall()
                return {'updates': [dict(zip(('id','node_id','release_id','status','detail','changed_at'), row)) for row in rows]}
            raise ValueError('Unsupported owner update action')

    def node(self, action, raw, signature):
        body = json.loads(raw)
        node_id, issued, nonce = body.get('node_id'), body.get('issued_at'), body.get('nonce')
        record = next((node for node in self.enrollment.nodes() if node.node_id == node_id), None)
        if record is None or not hmac.compare_digest(hmac.new(record.secret.encode(), raw, hashlib.sha256).hexdigest(), signature):
            raise PermissionError('Node authentication required')
        if type(issued) not in (int,float) or not math.isfinite(issued) or abs(self.clock()-issued)>60 or not isinstance(nonce,str) or not 1 <= len(nonce) <= 100:
            raise PermissionError('Fresh node request required')
        self.enrollment.claim_inference_request(node_id, 'app-update:'+nonce, issued)
        with self.lock:
            if action == 'poll':
                row = self.db.execute("SELECT id,release FROM app_updates WHERE target=? AND status='queued' ORDER BY created LIMIT 1", (node_id,)).fetchone()
                if row is None: return {'update': None}
                document, _ = self.catalog.get(row[1])
                claim = secrets.token_urlsafe(32)
                self.db.execute("UPDATE app_updates SET status='claimed',changed=?,claim_hash=? WHERE id=? AND status='queued'", (self.clock(), hashlib.sha256(claim.encode()).hexdigest(), row[0]))
                self.db.commit()
                return {'update': {'id': row[0], 'release': document, 'claim': claim}}
            if action == 'report':
                identifier, claim, stage = body.get('id'), body.get('claim'), body.get('status')
                if not isinstance(claim,str) or stage not in STAGES+TERMINAL: raise ValueError('Fixed update status required')
                row = self.db.execute('SELECT target,status,claim_hash FROM app_updates WHERE id=?', (identifier,)).fetchone()
                if row is None or row[0] != node_id or not hmac.compare_digest(row[2], hashlib.sha256(claim.encode()).hexdigest()):
                    raise PermissionError('Exact node update claim required')
                if row[1] in TERMINAL:
                    if row[1] != stage: raise PermissionError('Update already completed')
                    return {'id': identifier, 'status': stage}
                if stage in STAGES and (row[1] == 'queued' or STAGES.index(stage) < STAGES.index(row[1])):
                    raise PermissionError('Update state cannot go backwards')
                detail = {'healthy':'New app started and reconnected.', 'rolled_back':'New app failed startup; previous app restored.',
                          'failed':'Update was not applied.', 'interrupted':'Inspect the node before retrying.'}.get(stage, '')
                self.db.execute('UPDATE app_updates SET status=?,detail=?,changed=? WHERE id=?', (stage, detail, self.clock(), identifier))
                self.db.commit()
                return {'id': identifier, 'status': stage}
        raise ValueError('Unsupported node update action')
