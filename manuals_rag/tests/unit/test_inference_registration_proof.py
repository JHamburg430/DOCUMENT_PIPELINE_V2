"""Adversarial protocol checks for both packaged registration implementations."""
import hashlib
import hmac
import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


@pytest.fixture(params=['manuals_rag_inference', 'provis_inference_node'])
def protocol(request):
    package = request.param
    gateway_module = importlib.import_module(package + '.gateway')
    proof_module = importlib.import_module(package + '.registration_proof')
    enrollment = importlib.import_module(package + '.enrollment').EnrollmentManager()
    class Validator:
        def validate(self, token):
            if token in ('owner-token-a', 'owner-token-b'):
                return {'subject': 'owner', 'tenant_id': 'tenant'}
            if token == 'other-owner':
                return {'subject': 'other', 'tenant_id': 'tenant'}
            if token == 'other-tenant':
                return {'subject': 'owner', 'tenant_id': 'other'}
            raise PermissionError('invalid or expired fixture token')
    gateway = gateway_module.InferenceGateway(enrollment=enrollment, entra_validator=Validator())
    registration = gateway.enroll_node_with_entra(access_token='owner-token-a', node_id='fixture-node',
        endpoint='http://127.0.0.1:11434', agent_url='http://127.0.0.1:8765', tailnet_name='fixture')
    now = [100.0]
    gateway._registration_challenges = proof_module.RegistrationChallenges(clock=lambda: now[0])
    return gateway, enrollment, registration, now, proof_module, gateway_module, package


def issue(p, token='owner-token-a'):
    return p[0].create_node_registration_challenge_with_entra(access_token=token, node_id='fixture-node')


def arguments(p, nonce, token='owner-token-a', secret=None):
    key = (secret or p[2]['shared_secret']).encode()
    signature = hmac.new(key, p[4].registration_proof_bytes('fixture-node', nonce, token), hashlib.sha256).hexdigest()
    return dict(access_token=token, node_id='fixture-node', challenge=nonce, signature=signature)


def test_replay_denied_and_fresh_signin_keeps_existing_registration(protocol):
    p = protocol
    before = p[1].nodes()
    nonce = issue(p)
    args = arguments(p, nonce)
    assert p[0].verify_node_registration_with_entra(**args)
    assert not p[0].verify_node_registration_with_entra(**args)
    assert p[0].verify_node_registration_with_entra(**arguments(p, issue(p)))
    assert p[1].nodes() == before
    assert p[0].node_secrets['fixture-node'] == p[2]['shared_secret'].encode()


def test_expiration_and_gateway_restart_never_restore_old_proof(protocol):
    p = protocol
    old = arguments(p, issue(p))
    p[3][0] += p[4].CHALLENGE_TTL_SECONDS
    assert not p[0].verify_node_registration_with_entra(**old)
    pending = arguments(p, issue(p))
    replacement = p[5].InferenceGateway(enrollment=p[1], entra_validator=p[0].entra_validator)
    assert not replacement.verify_node_registration_with_entra(**pending)
    assert not replacement.verify_node_registration_with_entra(**old)


@pytest.mark.parametrize('token', ['owner-token-b', 'other-owner', 'other-tenant', 'expired-token'])
def test_challenge_bound_to_exact_token_and_current_owner(protocol, token):
    p = protocol
    nonce = issue(p)
    # Even another valid bearer for the same owner cannot reuse this nonce.
    assert not p[0].verify_node_registration_with_entra(**arguments(p, nonce, token))
    assert p[0].verify_node_registration_with_entra(**arguments(p, nonce))


def test_unissued_and_other_purpose_proofs_rejected(protocol):
    p = protocol
    assert not p[0].verify_node_registration_with_entra(**arguments(p, 'A' * 43))
    nonce = issue(p)
    args = arguments(p, nonce)
    legacy = json.dumps({'node_id': 'fixture-node', 'challenge': nonce}, sort_keys=True, separators=(',', ':')).encode()
    args['signature'] = hmac.new(p[2]['shared_secret'].encode(), legacy, hashlib.sha256).hexdigest()
    assert not p[0].verify_node_registration_with_entra(**args)
    assert p[0].verify_node_registration_with_entra(**arguments(p, nonce))


def test_revocation_and_secret_replacement_deny_issued_proof(protocol):
    p = protocol
    nonce = issue(p)
    args = arguments(p, nonce)
    p[0].node_secrets['fixture-node'] = b'wrong-current-secret'
    assert not p[0].verify_node_registration_with_entra(**args)
    assert issue(p) is None
    p[0].node_secrets['fixture-node'] = p[2]['shared_secret'].encode()
    p[1].revoke('fixture-node')
    assert not p[0].verify_node_registration_with_entra(**args)
    assert issue(p) is None


def test_simultaneous_replay_has_exactly_one_winner(protocol):
    p = protocol
    args = arguments(p, issue(p))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: p[0].verify_node_registration_with_entra(**args), range(16)))
    assert sum(results) == 1


def test_challenge_cache_bounded_without_eviction_or_token_storage(protocol):
    p = protocol
    cache = p[4].RegistrationChallenges(clock=lambda: p[3][0], max_pending=2, max_per_node=1)
    def create(node):
        return cache.issue(node_id=node, subject='owner', tenant='tenant', access_token='sensitive-fixture-token')
    first = create('one')
    assert first and create('one') is None
    assert create('two') and create('three') is None
    assert len(cache._pending) == 2
    assert 'sensitive-fixture-token' not in repr(cache._pending)
    p[3][0] += p[4].CHALLENGE_TTL_SECONDS
    assert create('three') and first not in cache._pending


def test_real_http_client_handshake_and_replay_denial(protocol):
    p = protocol
    server = ThreadingHTTPServer(('127.0.0.1', 0), p[5].create_gateway_handler(p[0]))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = importlib.import_module(p[6] + '.entra_client')
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        assert client.verify_node_registration_with_entra(gateway_url=base, access_token='owner-token-a',
            node_id='fixture-node', shared_secret=p[2]['shared_secret'])
        nonce = issue(p)
        args = arguments(p, nonce)
        request = Request(base + '/v1/nodes/verify-registration/entra', method='POST',
            headers={'Authorization': 'Bearer owner-token-a', 'Content-Type': 'application/json'},
            data=json.dumps({key: args[key] for key in ('node_id', 'challenge', 'signature')}).encode())
        with urlopen(request, timeout=5) as response:
            assert json.load(response) == {'status': 'accepted'}
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 403
        assert json.loads(error.value.read()) == {'status': 'denied'}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
