import dataclasses
from types import SimpleNamespace

import pytest

from manuals_rag_inference.models import InferenceRequest
from manuals_rag_inference.worker_admission import configure_worker_admission
from test_owner_diagnostic import diag, run, APPLICATION, MODEL, DELL


def enable(d):
    g = d.g
    g.require_entra_enrollment = True
    g.require_node_agents = True
    g.require_application_identity = True
    g.entra_validator = SimpleNamespace(config=SimpleNamespace(tenant_id='tenant'))
    g.scheduler.policy = dataclasses.replace(g.scheduler.policy, require_application_allowlist=True)
    configure_worker_admission(g, 'entra_enrolled')
    return g


def test_existing_and_future_company_workers_join_without_static_edits(diag):
    g = enable(diag)
    g.application_worker_allowlist[APPLICATION] = ['source']
    assert DELL in g.authorized_worker_ids(APPLICATION)
    record = SimpleNamespace(**vars(diag.records[1]))
    record.node_id = 'new-company-worker'
    record.secret = 'new-node-secret'
    record.agent_url = 'http://new-company-worker'
    diag.records.append(record)
    assert record.node_id not in g.authorized_worker_ids(APPLICATION)  # not registered in transport yet
    g.node_secrets[record.node_id] = record.secret.encode()
    g.node_agent_urls[record.node_id] = record.agent_url
    assert record.node_id in g.authorized_worker_ids(APPLICATION)
    assert g.application_worker_allowlist[APPLICATION] == ['source']


@pytest.mark.parametrize('field,value', [
    ('enrollment_method', 'pairing'), ('identity_tenant', 'another-tenant'),
    ('identity_tenant', None), ('identity_subject', ''), ('identity_subject', ' '),
    ('allowed_applications', ('unapproved',)),
])
def test_unapproved_records_cannot_join_even_when_static_list_and_heartbeat_allow(diag, field, value):
    g = enable(diag)
    setattr(diag.records[1], field, value)
    assert DELL not in g.authorized_worker_ids(APPLICATION)
    with pytest.raises(PermissionError):
        run(diag)
    assert not diag.calls


@pytest.mark.parametrize('cache', ['secret', 'endpoint'])
def test_replaced_transport_requires_current_registration(diag, cache):
    g = enable(diag)
    if cache == 'secret':
        g.node_secrets[DELL] = b'stale-secret'
    else:
        g.node_agent_urls[DELL] = 'http://stale-listener'
    assert DELL not in g.authorized_worker_ids(APPLICATION)


def test_enrollment_revocation_removes_membership_with_cached_heartbeat_remaining(diag):
    g = enable(diag)
    diag.records.remove(diag.records[1])
    assert DELL in {n.node_id for n in g.scheduler.snapshots()}
    assert DELL not in g.authorized_worker_ids(APPLICATION)
    with pytest.raises(PermissionError):
        run(diag)
    assert not diag.calls


def test_empty_pool_does_not_fall_back_to_static_or_caller_nodes(diag):
    g = enable(diag)
    diag.records.clear()
    request = InferenceRequest('fixture-empty-pool', APPLICATION, 'fixture', MODEL, 1000, 100,
                               allowed_node_ids=(DELL,))
    from manuals_rag_inference.gateway import AuthorizationError
    with pytest.raises(AuthorizationError):
        g.admit(request)


def test_caller_cannot_override_central_membership_and_health_still_gates_admission(diag):
    g = enable(diag)
    request = InferenceRequest('fixture-member-policy', APPLICATION, 'fixture', MODEL, 1000, 100,
                               allowed_node_ids=('unregistered-worker',))
    assert 'unregistered-worker' not in g._authoritative_request(request).allowed_node_ids
    for n in g.scheduler.snapshots():
        g.scheduler.upsert_node(dataclasses.replace(n, healthy=False))
    assert not g.admit(request).admitted


def test_legacy_application_defaults_are_server_owned(diag):
    g = enable(diag)
    diag.records[1].allowed_applications = ()
    assert DELL in g.authorized_worker_ids(APPLICATION)
    g.node_default_applications = ('different-application',)
    assert DELL not in g.authorized_worker_ids(APPLICATION)


def test_owner_diagnostic_uses_same_dynamic_policy_without_widening_target(diag):
    g = enable(diag)
    g.application_worker_allowlist[APPLICATION] = ['source']
    result = run(diag)
    assert result['receipt']['target_node_id'] == DELL
    assert result['receipt']['node_id'] == DELL
    assert len(diag.calls) == 1
    assert not g.scheduler._reservations


def test_static_mode_stays_explicit_and_invalid_modes_fail_closed(diag):
    g = diag.g
    g.application_worker_allowlist[APPLICATION] = ['source']
    configure_worker_admission(g, 'static')
    assert g.authorized_worker_ids(APPLICATION) == ('source',)
    with pytest.raises(ValueError):
        configure_worker_admission(g, 'anything')
    with pytest.raises(ValueError):
        configure_worker_admission(g, 'entra_enrolled')


@pytest.mark.parametrize('flag', ['require_entra_enrollment', 'require_node_agents', 'require_application_identity'])
def test_automatic_mode_refuses_weakened_authentication(diag, flag):
    g = enable(diag)
    setattr(g, flag, False)
    with pytest.raises(ValueError):
        configure_worker_admission(g, 'entra_enrolled')


def test_automatic_mode_requires_scheduler_restrictions_and_pinned_tenant(diag):
    g = enable(diag)
    g.scheduler.policy = dataclasses.replace(g.scheduler.policy, require_application_allowlist=False)
    with pytest.raises(ValueError):
        configure_worker_admission(g, 'entra_enrolled')
    g.scheduler.policy = dataclasses.replace(g.scheduler.policy, require_application_allowlist=True)
    g.entra_validator.config.tenant_id = ''
    with pytest.raises(ValueError):
        configure_worker_admission(g, 'entra_enrolled')
