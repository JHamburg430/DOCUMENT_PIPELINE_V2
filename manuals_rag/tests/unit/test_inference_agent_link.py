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
    assert send(link, source, job_id)['status'] == 'replied'
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


def _workspace_service(path, node_ids, now=None):
    records = [SimpleNamespace(node_id=node, secret='secret-'+node,
        identity_subject='john', identity_tenant='tenant', enrollment_method='entra')
        for node in node_ids]
    class Validator:
        def validate(self, token):
            return {'subject':'john','tenant_id':'tenant','object_id':OWNER,
                    'expires_at_epoch':500000}
    clock = now or [1000.0]
    service = AgentLink(Validator(), SimpleNamespace(nodes=lambda: records),
                        owner_object_id=OWNER, path=path, clock=lambda:clock[0])
    service.test_nodes = records
    service.test_clock = clock
    return service


def _register_workspace_node(service, node, peer_ids):
    token = login(service, node)
    direct = [{'peer_node_id':peer,'thread_id':f'dm-{node}-{peer}'} for peer in peer_ids if peer != node]
    service.dispatch('register', token, {'agent_id':str(uuid.uuid5(uuid.NAMESPACE_DNS,'agent:'+node)),
        'channel_thread_id':'channel-'+node,'direct_threads':direct})
    return token


def test_channel_fanout_has_three_durable_recipients_and_one_reply_each_after_reconnect(tmp_path):
    nodes = ['node-a','node-b','node-c']
    path = tmp_path/'channel.db'
    service = _workspace_service(path, nodes)
    tokens = {node:_register_workspace_node(service,node,nodes) for node in nodes}
    for node, token in tokens.items():
        participant = next(row for row in service.dispatch('workspace',token,{})['participants'] if row['node_id']==node)
        assert participant['online'] is True
        assert participant['agent_id']==str(uuid.uuid5(uuid.NAMESPACE_DNS,'agent:'+node))
    message_id = str(uuid.uuid4())
    original = {'id':message_id,'kind':'channel','message':'Status for the three-node test'}
    posted = service.dispatch('post',tokens['node-a'],original)
    snapshot = posted['messages'][0]
    assert [row['node_id'] for row in snapshot['recipients']] == nodes
    for node in nodes:
        job = service.dispatch('poll',tokens[node],{'ready_thread_ids':['channel-'+node]})['job']
        assert job and job['target_node']==node and job['kind']=='channel'
        assert job['recipient_snapshot']==nodes
        assert job['target_agent_id']==str(uuid.uuid5(uuid.NAMESPACE_DNS,'agent:'+node))
        claim={'id':job['id'],'claim':job['claim'],'agent_id':job['target_agent_id']}
        assert service.dispatch('executing',tokens[node],claim)['executing_node']==node
        service.dispatch('complete',tokens[node],{'id':job['id'],'claim':job['claim'],
            'agent_id':job['target_agent_id'],'status':'completed','result':'One test reply from '+node})
        assert service.dispatch('poll',tokens[node],{'ready_thread_ids':['channel-'+node]})['job'] is None
        with pytest.raises(PermissionError):
            service.dispatch('complete',tokens[node],{'id':job['id'],'claim':job['claim'],
                'agent_id':job['target_agent_id'],'status':'completed','result':'Duplicate reply'})
    history = service.dispatch('history',tokens['node-a'],{'conversation_id':posted['conversation_id']})
    assert len(history['messages'])==1  # Agent replies are records, never new broadcasts.
    recipients = history['messages'][0]['recipients']
    assert len(recipients)==3
    assert {row['status'] for row in recipients}=={'replied'}
    assert {row['executing_node'] for row in recipients}==set(nodes)
    assert {row['agent_id'] for row in recipients}=={str(uuid.uuid5(uuid.NAMESPACE_DNS,'agent:'+node)) for node in nodes}
    service.db.close()

    # A gateway restart drops owner capabilities but keeps durable messages,
    # recipient snapshots, claims and replies. Retrying the same id is inert.
    service = _workspace_service(path, nodes)
    tokens = {node:_register_workspace_node(service,node,nodes) for node in nodes}
    replay = service.dispatch('post',tokens['node-a'],original)
    assert replay['conversation_id']==posted['conversation_id']
    assert len(replay['messages'][0]['recipients'])==3
    assert all(row['status']=='replied' for row in replay['messages'][0]['recipients'])
    assert all(service.dispatch('poll',tokens[node],{'ready_thread_ids':['channel-'+node]})['job'] is None for node in nodes)
    service.db.close()


def test_newly_joined_node_is_not_added_to_an_existing_broadcast_snapshot(tmp_path):
    service = _workspace_service(tmp_path/'join.db',['node-a','node-b'])
    first = _register_workspace_node(service,'node-a',['node-a','node-b'])
    second = _register_workspace_node(service,'node-b',['node-a','node-b'])
    posted = service.dispatch('post',first,{'id':str(uuid.uuid4()),'kind':'channel','message':'pre-join broadcast'})
    assert {row['node_id'] for row in posted['messages'][0]['recipients']}=={'node-a','node-b'}
    service.test_nodes.append(SimpleNamespace(node_id='node-c',secret='secret-node-c',
        identity_subject='john',identity_tenant='tenant',enrollment_method='entra'))
    service.test_clock[0] += 86401
    third = login(service,'node-c')
    service.dispatch('register',third,{'agent_id':str(uuid.uuid5(uuid.NAMESPACE_DNS,'agent:node-c')),
        'channel_thread_id':'channel-node-c','direct_threads':[
        {'peer_node_id':'node-a','thread_id':'dm-node-c-node-a'},
        {'peer_node_id':'node-b','thread_id':'dm-node-c-node-b'}]})
    second = _register_workspace_node(service,'node-b',['node-a','node-b','node-c'])
    assert service.dispatch('poll',third,{'ready_thread_ids':['channel-node-c']})['job'] is None
    assert service.dispatch('poll',second,{'ready_thread_ids':['channel-node-b']})['job'] is not None
    assert len(service.dispatch('history',second,{'conversation_id':posted['conversation_id']})['messages'][0]['recipients'])==2
    service.db.close()


def test_job_2a8c9425_is_delivered_on_dell_while_sender_identity_stays_distinct(tmp_path):
    source_node, dell_node = 'john-System-Product-Name','Johns_Dell_V2'
    service = _workspace_service(tmp_path/'dell-attribution.db',[source_node,dell_node])
    source = _register_workspace_node(service,source_node,[source_node,dell_node])
    dell = _register_workspace_node(service,dell_node,[source_node,dell_node])
    identifier='2a8c9425-4993-4b20-a52f-d4d1296c746c'
    posted=service.dispatch('post',source,{'id':identifier,'kind':'dm','target_node_id':dell_node,
        'message':'Synthetic identity attribution regression'})
    job=service.dispatch('poll',dell,{'ready_thread_ids':['dm-Johns_Dell_V2-john-System-Product-Name']})['job']
    assert job['id']==identifier
    assert job['source_node']==source_node
    assert job['target_node']==dell_node
    assert job['thread_id']=='dm-Johns_Dell_V2-john-System-Product-Name'
    assert job['target_agent_id']==str(uuid.uuid5(uuid.NAMESPACE_DNS,'agent:'+dell_node))
    assert [row['node_id'] for row in posted['messages'][0]['recipients']]==[dell_node]
    service.db.close()
