import dataclasses
import time
import uuid
from types import SimpleNamespace
import pytest

from manuals_rag_inference.agent_link import AgentLink
from manuals_rag_inference.agent_link_session import save_session, load_session

OWNER = 'feb94da3-80b4-4f51-b6aa-3688b9b8e41e'


@pytest.fixture
def link(tmp_path):
    nodes = [SimpleNamespace(node_id=node, secret='secret-'+node, identity_subject='john', identity_tenant='tenant', enrollment_method='entra') for node in ('work', 'john')]
    class Validator:
        def validate(self, token):
            return {'subject': 'john', 'tenant_id': 'tenant', 'object_id': OWNER if token == 'owner-token' else 'another-employee', 'expires_at_epoch': 5000}
    now = [1000.0]
    service = AgentLink(Validator(), SimpleNamespace(nodes=lambda: nodes), owner_object_id=OWNER, path=tmp_path/'link.db', clock=lambda: now[0])
    service.test_clock = now
    service.test_nodes = nodes
    yield service
    service.db.close()


def login(link, node): return link.session('owner-token', {'node_id': node})['session_token']


def publish(link, token): return link.dispatch('publish', token, {'threads': [{'id': 'selected-thread', 'name': 'Explicit owner session'}]})


def send(link, token, identifier=None):
    return link.dispatch('send', token, {'id': identifier or str(uuid.uuid4()), 'node_id': 'john', 'thread_id': 'selected-thread', 'message': 'Synthetic status question'})


def test_other_employee_and_inference_credentials_are_not_owner_authority(link):
    for token in ('employee-token', 'operator-token', 'node-secret', 'application-token'):
        with pytest.raises(PermissionError): link.session(token, {'node_id': 'work'})
        with pytest.raises(PermissionError): link.dispatch('targets', token, {})
    link.test_nodes[0].identity_subject = 'other-user'
    with pytest.raises(PermissionError): login(link, 'work')


def test_only_explicit_targets_and_claimed_node_can_receive_or_complete(link):
    source, receiver = login(link, 'work'), login(link, 'john')
    with pytest.raises(PermissionError): send(link, source)
    publish(link, receiver)
    job_id = send(link, source)['id']
    assert link.dispatch('poll', source, {})['job'] is None
    job = link.dispatch('poll', receiver, {})['job']
    assert job['id'] == job_id
    assert link.dispatch('poll', receiver, {})['job'] is None
    with pytest.raises(PermissionError): link.dispatch('complete', source, {**job, 'status': 'completed', 'result': 'reply'})
    link.dispatch('complete', receiver, {**job, 'status': 'completed', 'result': 'Synthetic reply'})
    assert send(link, source, job_id)['status'] == 'completed'
    assert link.db.execute('SELECT message FROM jobs WHERE id=?', (job_id,)).fetchone()[0] == ''
    assert link.dispatch('messages', source, {})['messages'][0]['result'] == 'Synthetic reply'


def test_expired_revoked_and_logged_out_sessions_fail_closed(link):
    source, receiver = login(link, 'work'), login(link, 'john')
    publish(link, receiver)
    link.test_clock[0] += 31
    assert link.dispatch('targets', source, {}) == {'targets': []}
    publish(link, receiver)
    link.test_nodes[1].secret = 'rotated'
    with pytest.raises(PermissionError): publish(link, receiver)
    link.dispatch('logout', source, {})
    with pytest.raises(PermissionError): link.dispatch('targets', source, {})
    source = login(link, 'work')
    link.test_clock[0] = 6000
    with pytest.raises(PermissionError): link.dispatch('targets', source, {})


def test_uncertain_claims_are_never_automatically_replayed(link):
    source, receiver = login(link, 'work'), login(link, 'john')
    publish(link, receiver)
    identifier = send(link, source)['id']
    link.dispatch('poll', receiver, {})
    link.test_clock[0] += 601
    publish(link, receiver)
    assert link.dispatch('poll', receiver, {})['job'] is None
    assert link.dispatch('messages', source, {})['messages'][0]['status'] == 'uncertain'
    assert send(link, source, identifier)['status'] == 'uncertain'


def test_owner_capability_is_private_and_expiring(tmp_path):
    path = tmp_path/'owner.json'
    value = {'session_token': 'fixture-secret', 'expires_at_epoch': time.time()+60, 'gateway_url': 'https://fixture.invalid'}
    save_session(value, path)
    assert load_session(path) == value
    import os
    if os.name == 'nt': assert 'fixture-secret' not in path.read_text()
    value['expires_at_epoch'] = 1
    save_session(value, path)
    with pytest.raises(PermissionError): load_session(path)


def test_busy_sessions_stay_queued_without_injection(link):
    source, receiver = login(link, 'work'), login(link, 'john')
    publish(link, receiver)
    send(link, source)
    assert link.dispatch('poll', receiver, {'ready_thread_ids': []})['job'] is None
    assert link.dispatch('messages', source, {})['messages'][0]['status'] == 'queued'
    assert link.dispatch('poll', receiver, {'ready_thread_ids': ['selected-thread']})['job']


def test_http_route_checks_owner_identity_and_never_accepts_node_credentials(link):
    import json
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from test_inference_node_broker_gateway import serving
    from manuals_rag_inference.gateway import create_gateway_handler
    gateway = SimpleNamespace(agent_link=link)
    with serving(create_gateway_handler(gateway)) as url:
        for token in ('application-token', 'node-secret', 'employee-token'):
            request = Request(url+'/v1/agent-link/session', data=b'{"node_id":"work"}', headers={'Authorization': 'Bearer '+token})
            with pytest.raises(HTTPError) as denied: urlopen(request, timeout=3)
            assert denied.value.code == 403
        request = Request(url+'/v1/agent-link/session', data=b'{"node_id":"work"}', headers={'Authorization': 'Bearer owner-token'})
        with urlopen(request, timeout=3) as response: assert json.load(response)['session_token']


def test_codex_connector_cannot_target_an_unpublished_or_busy_thread():
    from manuals_rag_inference.codex_link import CodexLink
    connector = CodexLink.__new__(CodexLink)
    connector.allowed = {'selected'}
    calls = []
    def rpc(method, params, **kwargs):
        calls.append((method, params))
        return {'thread': {'status': {'type': 'active'}}}
    connector.rpc = rpc
    with pytest.raises(PermissionError): connector.send('unpublished', 'fixture')
    assert calls == []
    assert connector.send('selected', 'fixture')['status'] == 'blocked'
    assert [item[0] for item in calls] == ['thread/read']


def test_codex_reply_is_bound_to_the_started_turn_without_policy_overrides():
    import threading
    from manuals_rag_inference.codex_link import CodexLink
    connector = CodexLink.__new__(CodexLink)
    connector.lock = threading.RLock();connector.turns = {};connector.allowed = {'selected'}
    calls = []
    def rpc(method, params, **kwargs):
        calls.append((method, params))
        if method == 'thread/read': return {'thread': {'status': {'type': 'idle'}}}
        if method == 'thread/resume': return {'thread': {'id': 'selected'}}
        if method == 'turn/start':
            connector.turns['selected']['events']['unrelated-turn'] = {'done': True, 'text': 'private unrelated reply', 'status': 'completed'}
            connector.turns['selected']['events']['this-turn'] = {'done': True, 'text': 'Synthetic reply', 'status': 'completed'}
            return {'turn': {'id': 'this-turn'}}
    connector.rpc = rpc
    assert connector.send('selected', 'Synthetic request') == {'status': 'completed', 'result': 'Synthetic reply'}
    assert calls[1] == ('thread/resume', {'threadId': 'selected', 'excludeTurns': True})
    assert set(calls[2][1]) == {'threadId', 'input'}
