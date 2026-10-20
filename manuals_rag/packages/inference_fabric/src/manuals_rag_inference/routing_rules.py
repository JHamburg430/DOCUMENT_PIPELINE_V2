"""Versioned owner routing preferences; never an enrollment authorization source."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class RoutingRules:
    def __init__(self, path, applications, node_ids, *, clock=time.time):
        self.path = Path(path)
        self.applications = applications
        self.node_ids = node_ids
        self.clock = clock
        self.lock = threading.RLock()
        self.previews = {}
        self.document = {'format': 1, 'revision': 0, 'rules': {'applications': {}, 'nodes': {}}, 'history': []}
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink() or self.path.stat().st_size > 2_000_000:
                raise ValueError('Private bounded routing rules required')
            self.document = json.loads(self.path.read_text())
            if (set(self.document) != {'format', 'revision', 'rules', 'history'} or
                    self.document['format'] != 1 or type(self.document['revision']) is not int or
                    self.document['revision'] < 0 or type(self.document['history']) is not list or
                    len(self.document['history']) > 20):
                raise ValueError('Invalid routing rules store')
            self.validate(self.document['rules'], check_nodes=False)

    def validate(self, rules, *, check_nodes=True):
        if type(rules) is not dict or set(rules) != {'applications', 'nodes'} or len(canonical(rules)) > 65536:
            raise ValueError('Bounded application and node rules required')
        if any(type(rules[key]) is not dict for key in ('applications', 'nodes')):
            raise ValueError('Rule maps required')
        known_nodes = set(self.node_ids())
        def node(value):
            if (type(value) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', value) or
                    (check_nodes and value not in known_nodes)):
                raise ValueError('An enrolled node is required')
        def priority(value):
            if type(value) is not int or not -100 <= value <= 100:
                raise ValueError('Worker priority must be between -100 and 100')
        if len(rules['nodes']) > 128 or len(rules['applications']) > 32:
            raise ValueError('Rule limit exceeded')
        for name, rule in rules['nodes'].items():
            node(name)
            if type(rule) is not dict or set(rule) != {'enabled', 'priority'} or type(rule['enabled']) is not bool:
                raise ValueError('Node enabled state and priority required')
            priority(rule['priority'])
        for app, rule in rules['applications'].items():
            if app not in self.applications():
                raise ValueError('A configured application is required')
            if type(rule) is not dict or set(rule) != {'enabled', 'membership', 'nodes', 'excluded_nodes', 'preferred_nodes', 'fallback', 'node_priorities'}:
                raise ValueError('Exact application routing rule required')
            if type(rule['enabled']) is not bool or rule['membership'] not in ('automatic', 'selected') or rule['fallback'] not in ('any_eligible', 'preferred_only'):
                raise ValueError('Invalid membership or fallback policy')
            for key in ('nodes', 'excluded_nodes', 'preferred_nodes'):
                values = rule[key]
                if type(values) is not list or len(values) > 128 or any(type(v) is not str for v in values) or len(set(values)) != len(values):
                    raise ValueError('Unique bounded node list required')
                for value in values:
                    node(value)
            if type(rule['node_priorities']) is not dict or len(rule['node_priorities']) > 128:
                raise ValueError('Bounded node priorities required')
            for name, value in rule['node_priorities'].items():
                node(name)
                priority(value)
            if rule['fallback'] == 'preferred_only' and not rule['preferred_nodes']:
                raise ValueError('Choose a preferred node or allow eligible fallbacks')
        normalized = copy.deepcopy(rules)
        normalized['nodes'] = {n: r for n, r in normalized['nodes'].items() if r != {'enabled': True, 'priority': 0}}
        for app, rule in list(normalized['applications'].items()):
            rule['node_priorities'] = {n: p for n, p in rule['node_priorities'].items() if p}
            if rule == {'enabled': True, 'membership': 'automatic', 'nodes': [], 'excluded_nodes': [],
                        'preferred_nodes': [], 'fallback': 'any_eligible', 'node_priorities': {}}:
                del normalized['applications'][app]
        return normalized

    def state(self):
        with self.lock:
            return copy.deepcopy(self.document)

    def allowed_node_ids(self, application, base_ids, *, rules=None):
        if rules is None:
            rules = self.state()['rules']
        app = rules['applications'].get(application, {})
        if app.get('enabled', True) is not True:
            return ()
        result = set(base_ids)
        if app.get('membership', 'automatic') == 'selected':
            result.intersection_update(app['nodes'])
        result.difference_update(app.get('excluded_nodes', []))
        if app.get('fallback', 'any_eligible') == 'preferred_only':
            result.intersection_update(app['preferred_nodes'])
        return tuple(sorted(n for n in result if rules['nodes'].get(n, {}).get('enabled', True)))

    def priorities(self, application, members):
        rules = self.state()['rules']
        app = rules['applications'].get(application, {})
        preferred = app.get('preferred_nodes', [])
        return {n: rules['nodes'].get(n, {}).get('priority', 0) + app.get('node_priorities', {}).get(n, 0)
                + (1000 if n in preferred else 0) for n in members}

    def preview(self, rules, revision, actor):
        normalized = self.validate(rules)
        with self.lock:
            if type(revision) is not int or revision != self.document['revision']:
                raise FileExistsError('Rules changed; refresh before saving')
            now = self.clock()
            self.previews = {k: v for k, v in self.previews.items() if v['expires'] > now}
            if len(self.previews) >= 32:
                raise ValueError('Too many pending previews')
            identifier = secrets.token_urlsafe(24)
            self.previews[identifier] = {'actor': actor, 'revision': revision, 'rules': canonical(normalized), 'expires': now + 300}
            changes = []
            for section in ('applications', 'nodes'):
                old = self.document['rules'][section]
                for name in sorted(set(old) | set(normalized[section])):
                    if old.get(name) != normalized[section].get(name):
                        changes.append({'section': section, 'name': name, 'before': old.get(name), 'after': normalized[section].get(name)})
            return {'preview_id': identifier, 'revision': revision, 'changes': changes, 'expires_in': 300}

    def save(self, rules, revision, preview_id, actor, *, audit_actor='owner'):
        normalized = self.validate(rules)
        with self.lock:
            preview = self.previews.get(preview_id)
            if (not preview or preview['actor'] != actor or preview['expires'] <= self.clock() or
                    preview['rules'] != canonical(normalized) or preview['revision'] != revision):
                raise PermissionError('A fresh matching owner preview is required')
            if type(revision) is not int or revision != self.document['revision']:
                raise FileExistsError('Rules changed; refresh before saving')
            old = self.document
            history = (old['history'] + [{'revision': old['revision'], 'rules': old['rules'],
                                         'changed_at': self.clock(), 'changed_by': audit_actor}])[-20:]
            document = {'format': 1, 'revision': revision + 1, 'rules': normalized, 'history': history}
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.path.is_symlink():
                raise PermissionError('Private regular rules file required')
            temp = self.path.with_name(self.path.name + '.' + secrets.token_hex(8) + '.tmp')
            with temp.open('xb') as stream:
                temp.chmod(0o600)
                stream.write(canonical(document))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
            if os.name != 'nt':
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            self.document = document
            self.previews.clear()
            return {'revision': document['revision'], 'sha256': hashlib.sha256(canonical(document)).hexdigest(), 'saved': True}
