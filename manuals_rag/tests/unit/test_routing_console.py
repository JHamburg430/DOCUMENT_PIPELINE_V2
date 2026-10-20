import copy
import dataclasses
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from manuals_rag_inference.gateway import AuthorizationError, create_gateway_handler
from manuals_rag_inference.models import InferenceRequest
from manuals_rag_inference.routing_console import COOKIE, FLOW, RoutingConsole
from manuals_rag_inference.routing_rules import RoutingRules
from test_enrolled_worker_admission import enable
from test_owner_diagnostic import diag, APPLICATION, MODEL, DELL
from test_inference_node_broker_gateway import serving

ORIGIN = 'https://fixture.ts.net:8445'


def app_rule(**updates):
    rule = dict(enabled=True, membership='automatic', nodes=[], excluded_nodes=[], preferred_nodes=[],
                fallback='any_eligible', node_priorities={})
    rule.update(updates)
    return rule


@pytest.fixture
def console(diag, tmp_path):
    g = enable(diag)
    validator = g.agent_link.validator.validate
    g.entra_validator.validate = validator
    g.entra_client_id = 'public-client'
    g.entra_scope = 'api://fixture/Node.Enroll'
    g.application_tokens = {APPLICATION: 'PRIVATE_APPLICATION_TOKEN'}
    g.routing_rules = RoutingRules(tmp_path / 'routing.json', lambda: {APPLICATION}, lambda: [n.node_id for n in diag.records])
    clock = [time.time()]
    calls = []
    def form(url, values):
        calls.append((url, values))
        return ({'device_code': 'PRIVATE_DEVICE_CODE', 'user_code': 'ABCD1234', 'interval': 5, 'expires_in': 600}
                if url.endswith('/devicecode') else {'access_token': 'owner'})
    g.routing_console = RoutingConsole(g, ORIGIN, form=form, clock=lambda: clock[0])
    diag.clock, diag.auth_calls = clock, calls
    return diag


def commit(console, rules):
    r = console.g.routing_rules
    revision = r.state()['revision']
    preview = r.preview(rules, revision, 'owner-session')
    return r.save(rules, revision, preview['preview_id'], 'owner-session')


def request():
    return InferenceRequest('fixture-routing-policy', APPLICATION, 'fixture', MODEL, 1000, 100)


def test_rules_persist_and_restore_as_a_new_revision(console):
    r = console.g.routing_rules
    rules = {'applications': {}, 'nodes': {DELL: {'enabled': False, 'priority': 0}}}
    commit(console, rules)
    assert DELL not in console.g.authorized_worker_ids(APPLICATION)
    reloaded = RoutingRules(r.path, r.applications, r.node_ids)
    assert reloaded.state()['revision'] == 1
    assert reloaded.state()['rules'] == rules
    old = r.state()['history'][-1]['rules']
    commit(console, old)
    assert r.state()['revision'] == 2
    assert DELL in console.g.authorized_worker_ids(APPLICATION)
    assert len(r.state()['history']) == 2


def test_priorities_change_worker_selection_and_client_priority_map_is_ignored(console):
    g = console.g
    commit(console, {'applications': {APPLICATION: app_rule(node_priorities={'other': 50})}, 'nodes': {}})
    req = dataclasses.replace(request(), worker_priorities={DELL: 9999999})
    assert g.admit(req).node_id == 'other'
    assert g._authoritative_request(req).worker_priorities[DELL] == 0


def test_preferred_workers_fall_back_only_before_dispatch(console):
    g = console.g
    rule = app_rule(preferred_nodes=[DELL])
    commit(console, {'applications': {APPLICATION: rule}, 'nodes': {}})
    assert g.admit(request()).node_id == DELL
    current = next(n for n in g.scheduler.snapshots() if n.node_id == DELL)
    g.scheduler.upsert_node(dataclasses.replace(current, healthy=False))
    assert g.admit(request()).node_id == 'other'
    rule['fallback'] = 'preferred_only'
    commit(console, {'applications': {APPLICATION: rule}, 'nodes': {}})
    assert not g.admit(request()).admitted


def test_rules_can_narrow_but_cannot_authorize_a_foreign_or_revoked_node(console):
    g = console.g
    commit(console, {'applications': {APPLICATION: app_rule(membership='selected', nodes=[DELL])}, 'nodes': {}})
    console.records[1].identity_tenant = 'foreign'
    with pytest.raises(AuthorizationError):
        g.admit(request())


def test_pausing_an_application_blocks_new_work(console):
    commit(console, {'applications': {APPLICATION: app_rule(enabled=False)}, 'nodes': {}})
    with pytest.raises(AuthorizationError):
        console.g.admit(request())


def test_preview_is_bound_to_exact_rules_owner_revision_and_expiry(console):
    r = console.g.routing_rules
    rules = {'applications': {}, 'nodes': {DELL: {'enabled': False, 'priority': 0}}}
    p = r.preview(rules, 0, 'owner-A')
    for actor, candidate in [('owner-B', rules), ('owner-A', {'applications': {}, 'nodes': {}})]:
        with pytest.raises(PermissionError):
            r.save(candidate, 0, p['preview_id'], actor)
    r.save(rules, 0, p['preview_id'], 'owner-A')
    with pytest.raises(PermissionError):
        r.save(rules, 0, p['preview_id'], 'owner-A')
    with pytest.raises(FileExistsError):
        r.preview(rules, 0, 'owner-A')
    p = r.preview(rules, 1, 'owner-A')
    r.previews[p['preview_id']]['expires'] = 0
    with pytest.raises(PermissionError):
        r.save(rules, 1, p['preview_id'], 'owner-A')


@pytest.mark.parametrize('rules', [
    {'applications': {}, 'nodes': {}, 'gateway_token': 'x'},
    {'applications': {'unknown': app_rule()}, 'nodes': {}},
    {'applications': {}, 'nodes': {'unregistered': {'enabled': True, 'priority': 1}}},
    {'applications': {}, 'nodes': {DELL: {'enabled': True, 'priority': True}}},
    {'applications': {}, 'nodes': {DELL: {'enabled': True, 'priority': 101}}},
    {'applications': {APPLICATION: app_rule(nodes=[{}])}, 'nodes': {}},
    {'applications': {APPLICATION: app_rule(fallback='preferred_only')}, 'nodes': {}},
    {'applications': {APPLICATION: app_rule(endpoint='http://untrusted')}, 'nodes': {}},
])
def test_invalid_rules_cannot_change_persistent_state(console, rules):
    r = console.g.routing_rules
    with pytest.raises((ValueError, TypeError)):
        r.preview(rules, 0, 'owner')
    assert r.state()['revision'] == 0 and not r.path.exists()


def http(base, route, payload=None, *, cookie='', origin=ORIGIN):
    headers = {'Cookie': cookie}
    if payload is not None:
        headers.update({'Content-Type': 'application/json', 'Origin': origin})
    req = Request(base + route, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        response = urlopen(req, timeout=3)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read()
        value = json.loads(raw) if 'application/json' in response.headers.get('Content-Type', '') else raw.decode()
        return response.status, value, response.headers


def test_http_owner_cookie_csrf_preview_save_and_revocation(console):
    g = console.g
    with serving(create_gateway_handler(g)) as base:
        status, html, headers = http(base, '/routing')
        assert status == 200 and 'Compute Control' in html and '__NONCE__' not in html
        assert "script-src 'nonce-" in headers['Content-Security-Policy']
        assert http(base, '/v1/routing/state')[0] == 401
        cookie = COOKIE + '=' + console.token
        assert http(base, '/v1/routing/state', cookie=cookie)[0] == 200
        rules = {'applications': {}, 'nodes': {DELL: {'enabled': False, 'priority': 0}}}
        body = {'rules': rules, 'expected_revision': 0}
        assert http(base, '/v1/routing/preview', body, cookie=cookie, origin='https://evil.example')[0] == 403
        status, preview, _ = http(base, '/v1/routing/preview', body, cookie=cookie)
        assert status == 200 and DELL not in preview['effective_workers'][APPLICATION]
        assert http(base, '/v1/routing/save', dict(body, preview_id=preview['preview_id']), cookie=cookie)[0] == 200
        assert g.routing_rules.state()['revision'] == 1
        console.records[0].secret = 'rotated'
        assert http(base, '/v1/routing/state', cookie=cookie)[0] == 401


def test_microsoft_flow_never_returns_private_authentication_material(console):
    with serving(create_gateway_handler(console.g)) as base:
        status, start, headers = http(base, '/v1/routing/login/start', {})
        assert status == 200 and start['user_code'] == 'ABCD1234'
        cookie = headers['Set-Cookie'].split(';')[0]
        assert FLOW in cookie and all(flag in headers['Set-Cookie'] for flag in ['Secure', 'HttpOnly', 'SameSite=Strict'])
        assert 'PRIVATE_DEVICE_CODE' not in json.dumps(start)
        assert http(base, '/v1/routing/login/poll', {})[0] == 401
        assert http(base, '/v1/routing/login/poll', {}, cookie=cookie)[1]['status'] == 'pending'
        assert len(console.auth_calls) == 1
        console.clock[0] += 6
        status, result, headers = http(base, '/v1/routing/login/poll', {}, cookie=cookie)
        assert status == 200 and result == {'status': 'verified'}
        assert COOKIE in headers['Set-Cookie']
        session_cookie = headers['Set-Cookie'].split(';')[0]
        status, state, _ = http(base, '/v1/routing/state', cookie=session_cookie)
        assert status == 200
        text = json.dumps(state)
        for private in ['PRIVATE_APPLICATION_TOKEN', 'secret-source', 'secret-'+DELL, 'PRIVATE_DEVICE_CODE', 'session_token']:
            assert private not in text


def test_non_owner_microsoft_sign_in_is_rejected(console):
    console.g.routing_console.form = lambda url, values: ({'device_code': 'private', 'user_code': 'ABC123', 'interval': 1} if url.endswith('/devicecode') else {'access_token': 'employee'})
    with serving(create_gateway_handler(console.g)) as base:
        _, _, headers = http(base, '/v1/routing/login/start', {})
        cookie = headers['Set-Cookie'].split(';')[0]
        console.clock[0] += 2
        assert http(base, '/v1/routing/login/poll', {}, cookie=cookie)[0] == 401


def test_invalid_origin_or_missing_owner_configuration_refused(console):
    with pytest.raises(ValueError):
        RoutingConsole(console.g, 'http://fixture.ts.net')
    console.g.agent_link = None
    with pytest.raises(ValueError):
        RoutingConsole(console.g, ORIGIN)


def test_owner_worker_check_uses_existing_exact_target_and_cleanup(console):
    with serving(create_gateway_handler(console.g)) as base:
        body = {'target_node_id': DELL, 'model': MODEL}
        assert http(base, '/v1/routing/diagnostic', body)[0] == 401
        cookie = COOKIE + '=' + console.token
        assert http(base, '/v1/routing/diagnostic', body, cookie=cookie, origin='https://evil.example')[0] == 403
        status, result, _ = http(base, '/v1/routing/diagnostic', body, cookie=cookie)
        assert status == 200 and result['receipt']['node_id'] == DELL
        assert len(console.calls) == 1 and not console.g.scheduler._reservations
        assert http(base, '/v1/routing/diagnostic', dict(body, allow_model_pull=True), cookie=cookie)[0] == 400
        assert len(console.calls) == 1


def remembered_cookie(console, base):
    _, _, headers = http(base, '/v1/routing/login/start', {})
    flow = headers['Set-Cookie'].split(';')[0]
    console.clock[0] += 6
    status, _, headers = http(base, '/v1/routing/login/poll', {}, cookie=flow)
    assert status == 200 and 'Max-Age=34560000' in headers['Set-Cookie']
    return headers['Set-Cookie'].split(';')[0]


def test_remembered_pc_survives_token_expiry_and_gateway_restart(console, tmp_path):
    import sqlite3
    link = console.g.agent_link
    path = tmp_path / 'owner.sqlite'
    link.db.backup(sqlite3.connect(path))
    link.db.close()
    link.db = sqlite3.connect(path, check_same_thread=False)
    with serving(create_gateway_handler(console.g)) as base:
        cookie = remembered_cookie(console, base)
        token = cookie.split('=', 1)[1]
        assert token not in str(link.db.execute('SELECT * FROM routing_devices').fetchall())
        assert link._hash(token) not in link.sessions
        console.clock[0] += 86400 * 90
        link.sessions.clear()
        link.db.close()
        link.db = sqlite3.connect(path, check_same_thread=False)
        console.g.routing_console = RoutingConsole(console.g, ORIGIN, clock=lambda: console.clock[0])
        status, _, headers = http(base, '/v1/routing/state', cookie=cookie)
        assert status == 200 and 'Max-Age=34560000' in headers['Set-Cookie']
        assert http(base, '/v1/routing/diagnostic', {'target_node_id': DELL, 'model': MODEL}, cookie=cookie)[0] == 200
        assert not link.sessions
        assert http(base, '/v1/routing/logout', {}, cookie=cookie)[0] == 200
        assert http(base, '/v1/routing/state', cookie=cookie)[0] == 401
        console.g.routing_console = RoutingConsole(console.g, ORIGIN)
        assert http(base, '/v1/routing/state', cookie=cookie)[0] == 401


@pytest.mark.parametrize('change', ['secret', 'subject', 'tenant', 'owner', 'removed'])
def test_remembered_pc_does_not_bypass_revocation(console, change):
    with serving(create_gateway_handler(console.g)) as base:
        cookie = remembered_cookie(console, base)
        record = next(n for n in console.records if n.node_id == 'Johns_Dell_V2')
        # Login selects the lexically first owned enrolled record.
        if change == 'secret': record.secret = 'replaced'
        elif change == 'subject': record.identity_subject = 'different'
        elif change == 'tenant': record.identity_tenant = 'different'
        elif change == 'owner': console.g.agent_link.owner = 'different'
        else: console.records.remove(record)
        assert http(base, '/v1/routing/state', cookie=cookie)[0] == 401
