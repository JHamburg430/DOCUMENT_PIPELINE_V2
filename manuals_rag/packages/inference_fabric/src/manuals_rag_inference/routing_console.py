"""Tailnet routing console with Microsoft owner authentication and reviewed saves."""
import hashlib
from http.cookies import SimpleCookie
import json
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener
from urllib.error import HTTPError

from .node_updater import NoRedirect


COOKIE = '__Secure-provis-routing'
FLOW = '__Secure-provis-routing-flow'
PREFIX = '/v1/routing/'


class OwnerRequired(PermissionError):
    pass


def microsoft_form(url, values):
    request = Request(url, data=urlencode(values).encode(), headers={'Content-Type': 'application/x-www-form-urlencoded'})
    try:
        response = build_opener(NoRedirect()).open(request, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError('Authentication response limit exceeded')
    result = json.loads(raw)
    if type(result) is not dict:
        raise ValueError('Authentication response invalid')
    return result


class RoutingConsole:
    def __init__(self, gateway, origin, *, form=microsoft_form, clock=time.time):
        address = urlsplit(origin)
        if (address.scheme != 'https' or not address.hostname or address.username or address.password or
                address.path or address.query or address.fragment):
            raise ValueError('Exact HTTPS routing-console origin required')
        if gateway.agent_link is None or gateway.routing_rules is None:
            raise ValueError('Owner authentication and persistent routing rules required')
        self.gateway, self.origin, self.form, self.clock = gateway, origin, form, clock
        self.flows = {}
        self.starts = []
        self.lock = threading.RLock()

    def _cookie(self, handler, name):
        raw = handler.headers.get('Cookie', '')
        if len(raw) > 8192:
            raise OwnerRequired('Owner sign-in required')
        cookies = SimpleCookie()
        cookies.load(raw)
        return cookies[name].value if name in cookies else ''

    def _owner(self, handler):
        token = self._cookie(handler, COOKIE)
        if not token:
            raise OwnerRequired('Owner sign-in required')
        link = self.gateway.agent_link
        try:
            with link.lock:
                link._authorized_session(token)
        except PermissionError:
            raise OwnerRequired('Owner sign-in expired') from None
        return token

    def _send(self, handler, status, payload, *, cookie=None, html=False):
        nonce = secrets.token_urlsafe(18)
        raw = payload.replace('__NONCE__', nonce).encode() if html else json.dumps(payload, allow_nan=False).encode()
        handler.send_response(status)
        handler.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
        handler.send_header('Content-Length', str(len(raw)))
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('X-Content-Type-Options', 'nosniff')
        handler.send_header('X-Frame-Options', 'DENY')
        handler.send_header('Referrer-Policy', 'no-referrer')
        handler.send_header('Content-Security-Policy', "default-src 'self'; script-src 'nonce-" + nonce + "'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        if cookie:
            name, value, age = cookie
            handler.send_header('Set-Cookie', f'{name}={value}; Path={PREFIX}; Max-Age={int(age)}; Secure; HttpOnly; SameSite=Strict')
        handler.end_headers()
        handler.wfile.write(raw)

    def _start(self, handler):
        now = self.clock()
        with self.lock:
            self.flows = {k: v for k, v in self.flows.items() if v['expires'] > now}
            old = self._cookie(handler, FLOW)
            self.flows.pop(old, None)
            self.starts = [stamp for stamp in self.starts if now - stamp < 60]
            if len(self.flows) >= 4 or len(self.starts) >= 12:
                raise ValueError('Please wait before starting another sign-in')
            self.starts.append(now)
            identifier = secrets.token_urlsafe(32)
            self.flows[identifier] = {'expires': now + 30, 'polling': True}
        g = self.gateway
        try:
            tenant = g.entra_validator.config.tenant_id
            result = self.form(f'https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode',
                               {'client_id': g.entra_client_id, 'scope': g.entra_scope})
            code = result.get('user_code', '')
            if not re.fullmatch(r'[A-Z0-9-]{6,32}', code) or not result.get('device_code'):
                raise ValueError('Microsoft sign-in could not be started')
            ttl = max(1, min(600, int(result.get('expires_in', 600))))
            interval = max(1, min(30, int(result.get('interval', 5))))
            with self.lock:
                self.flows[identifier] = {'device_code': result['device_code'], 'expires': now + ttl,
                                           'interval': interval, 'next_poll': now + interval, 'polling': False}
            return {'status': 'pending', 'user_code': code, 'verification_uri': 'https://microsoft.com/devicelogin', 'retry_after': interval}, (FLOW, identifier, ttl)
        except Exception:
            with self.lock:
                self.flows.pop(identifier, None)
            raise

    def _poll(self, handler):
        identifier = self._cookie(handler, FLOW)
        with self.lock:
            flow = self.flows.get(identifier)
            now = self.clock()
            if flow is None or flow['expires'] <= now:
                self.flows.pop(identifier, None)
                raise OwnerRequired('Microsoft sign-in expired')
            if flow['polling'] or now < flow['next_poll']:
                return {'status': 'pending', 'retry_after': flow['interval']}, None
            flow['polling'] = True
            flow['next_poll'] = now + flow['interval']
        g = self.gateway
        try:
            tenant = g.entra_validator.config.tenant_id
            result = self.form(f'https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token',
                               {'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                                'client_id': g.entra_client_id, 'device_code': flow['device_code']})
            if result.get('error') in ('authorization_pending', 'slow_down'):
                with self.lock:
                    if result['error'] == 'slow_down':
                        flow['interval'] = min(30, flow['interval'] + 5)
                        flow['next_poll'] = self.clock() + flow['interval']
                return {'status': 'pending', 'retry_after': flow['interval']}, None
            access_token = result.get('access_token')
            if not access_token:
                raise OwnerRequired('Microsoft sign-in did not complete')
            identity = g.entra_validator.validate(access_token)
            if identity.get('object_id') != g.agent_link.owner:
                raise OwnerRequired('Only the configured owner may manage routing')
            records = [n for n in g.enrollment.nodes() if n.enrollment_method == 'entra'
                       and n.identity_tenant == identity['tenant_id'] and n.identity_subject == identity['subject']]
            if not records:
                raise OwnerRequired('Current company enrollment required')
            session = g.agent_link.session(access_token, {'node_id': sorted(records, key=lambda n: n.node_id)[0].node_id})
            age = max(1, min(3600, int(session['expires_at_epoch'] - self.clock())))
            with self.lock:
                self.flows.pop(identifier, None)
            return {'status': 'verified'}, (COOKIE, session['session_token'], age)
        except OwnerRequired:
            with self.lock:
                self.flows.pop(identifier, None)
            raise
        finally:
            with self.lock:
                flow['polling'] = False

    def state(self):
        g = self.gateway
        document = g.routing_rules.state()
        snapshots = {n.node_id: n for n in g.scheduler.snapshots()}
        nodes = []
        for record in g.enrollment.nodes():
            snapshot = snapshots.get(record.node_id)
            fresh = snapshot is not None and 0 <= self.clock() - snapshot.observed_at_epoch <= g.scheduler.policy.heartbeat_ttl_seconds
            nodes.append({'node_id': record.node_id, 'company_enrolled': record.enrollment_method == 'entra',
                          'applications': list(record.allowed_applications or g.node_default_applications),
                          'healthy': bool(fresh and snapshot.healthy),
                          'inference_enabled': bool(fresh and snapshot.inference_enabled),
                          'version': snapshot.node_app_version if snapshot else '',
                          'active_requests': snapshot.active_requests if snapshot else 0,
                          'queued_requests': snapshot.queued_requests if snapshot else 0,
                          'gpus': [{'name': gpu.name, 'total_vram_mb': gpu.total_vram_mb} for gpu in snapshot.gpus] if snapshot else []})
        applications = sorted(set(g.application_tokens) | set(g.node_default_applications))
        return {'revision': document['revision'], 'rules': document['rules'], 'history': document['history'],
                'nodes': nodes, 'applications': applications,
                'membership_mode': 'automatic' if g.enrolled_worker_admission is not None else 'static',
                'eligible_workers': {app: list(g.authorized_worker_ids(app)) for app in applications}}

    def handle(self, handler):
        path = handler.path
        if path not in ('/routing', '/routing/') and not path.startswith(PREFIX):
            return False
        try:
            if handler.command == 'GET' and path in ('/routing', '/routing/'):
                self._send(handler, 200, Path(__file__).with_name('routing_console.html').read_text(encoding='utf-8'), html=True)
                return True
            if handler.command == 'GET' and path == PREFIX + 'state':
                self._owner(handler)
                self._send(handler, 200, self.state())
                return True
            if handler.command != 'POST':
                self._send(handler, 404, {'error': 'Route not found'})
                return True
            if handler.headers.get('Origin') != self.origin:
                raise PermissionError('Routing changes require the approved console origin')
            if handler.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                raise ValueError('JSON request required')
            length = int(handler.headers.get('Content-Length') or 0)
            if not 0 < length <= 70000:
                raise ValueError('Bounded request body required')
            payload = json.loads(handler.rfile.read(length))
            if type(payload) is not dict:
                raise ValueError('Object required')
            cookie = None
            if path == PREFIX + 'login/start' and payload == {}:
                result, cookie = self._start(handler)
            elif path == PREFIX + 'login/poll' and payload == {}:
                result, cookie = self._poll(handler)
            else:
                token = self._owner(handler)
                actor = hashlib.sha256(token.encode()).hexdigest()
                if path == PREFIX + 'logout' and payload == {}:
                    self.gateway.agent_link.dispatch('logout', token, {})
                    result, cookie = {'signed_out': True}, (COOKIE, '', 0)
                elif path == PREFIX + 'preview' and set(payload) == {'rules', 'expected_revision'}:
                    result = self.gateway.routing_rules.preview(payload['rules'], payload['expected_revision'], actor)
                    g = self.gateway
                    result['effective_workers'] = {}
                    for app in sorted(set(g.application_tokens) | set(g.node_default_applications)):
                        base = (g.enrolled_worker_admission.allowed_node_ids(g, app) if g.enrolled_worker_admission is not None
                                else tuple(g.application_worker_allowlist.get(app, ())))
                        result['effective_workers'][app] = list(g.routing_rules.allowed_node_ids(app, base, rules=payload['rules']))
                elif path == PREFIX + 'save' and set(payload) == {'rules', 'expected_revision', 'preview_id'}:
                    result = self.gateway.routing_rules.save(payload['rules'], payload['expected_revision'], payload['preview_id'], actor,
                                                              audit_actor=self.gateway.agent_link.owner)
                else:
                    raise ValueError('Unsupported routing action')
            self._send(handler, 200, result, cookie=cookie)
        except OwnerRequired as error:
            self._send(handler, 401, {'error': str(error)})
        except PermissionError as error:
            self._send(handler, 403, {'error': str(error)})
        except FileExistsError as error:
            self._send(handler, 409, {'error': str(error)})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._send(handler, 400, {'error': 'Invalid routing request or sign-in response'})
        except Exception:
            self._send(handler, 503, {'error': 'Routing console temporarily unavailable'})
        return True
