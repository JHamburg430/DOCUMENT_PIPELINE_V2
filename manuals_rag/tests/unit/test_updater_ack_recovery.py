"""Crash/restart and actual queue idempotency; fixture credentials only."""
import json
from types import SimpleNamespace
import pytest
from manuals_rag_inference import node_updater as u
from test_inference_node_updates import release, queue, plan_fixture, enqueue, signed_node


@pytest.fixture
def committed(tmp_path, release):
    config,path,plan,settings=plan_fixture(tmp_path,release)
    config.write_text(json.dumps({'node_id':'john','gateway_url':plan['gateway'],'shared_secret':'fixture'}))
    u.private_json(path/'healthy.json',{'nonce':plan['probe_nonce'],'version':release.manifest['version'],'node_id':'john'})
    u.private_json(u.root_for(config)/'active.json',{'release':plan['release']})
    return config,path,plan,{**settings,'version':release.manifest['version']},release.manifest


class Clock:
    def __init__(self):self.now=1000.;self.delays=[]
    def __call__(self):return self.now
    def sleep(self,delay):self.delays.append(delay);self.now+=delay


def deliver(fixture, send, clock):
    config,path,plan,settings,m=fixture
    return u.deliver_healthy_ack(path/'plan.json',plan,m,config,send=send,clock=clock,sleep=clock.sleep)


def test_crash_after_pointer_commit_recovers_same_claim_without_launch_or_rollback(committed,monkeypatch):
    config,path,plan,settings,m=committed
    before=config.read_bytes();pointer=(u.root_for(config)/'active.json').read_bytes()
    attempts=[]
    def crash(*args,**kw):attempts.append(args[3].copy());raise KeyboardInterrupt()
    clock=Clock()
    with pytest.raises(KeyboardInterrupt):deliver(committed,crash,clock)
    assert json.loads((path/'healthy-ack.json').read_text())['attempts']==1
    clock.now+=2
    def success(*args,**kw):attempts.append(args[3].copy());return {'id':plan['id'],'status':'healthy'}
    monkeypatch.setattr(u,'request',success)
    assert u.recover_healthy_ack(config,settings,'john',clock=clock,sleep=clock.sleep)=='acknowledged'
    assert attempts[0]==attempts[1]
    assert config.read_bytes()==before and (u.root_for(config)/'active.json').read_bytes()==pointer
    assert u.recover_healthy_ack(config,settings,'john')=='acknowledged'
    assert len(attempts)==2


def test_missing_outbox_after_commit_is_reconstructed(committed):
    config,path,plan,settings,m=committed
    assert not (path/'healthy-ack.json').exists()
    assert deliver(committed,lambda *a,**k:{'id':plan['id'],'status':'healthy'},Clock())=='acknowledged'


@pytest.mark.parametrize('code',[400,401,403,404,409,422,302])
def test_permanent_failure_stops_across_restarts(committed,code,monkeypatch):
    calls=[]
    def rejected(*a,**k):calls.append(True);raise u.HTTPError('https://fixture.invalid',code,'fixture',{},None)
    clock=Clock();assert deliver(committed,rejected,clock)=='blocked'
    config,path,plan,settings,m=committed
    monkeypatch.setattr(u,'request',lambda *a,**k:pytest.fail('permanent denial retried'))
    assert u.recover_healthy_ack(config,settings,'john')=='blocked'
    assert calls==[True] and clock.delays==[]


@pytest.mark.parametrize('error',[u.URLError('reset'),TimeoutError(),ConnectionResetError(),u.HTTPError('https://fixture.invalid',503,'fixture',{},None),u.HTTPError('https://fixture.invalid',429,'fixture',{},None)])
def test_transient_retry_has_bounded_backoff_and_exact_payload(committed,error):
    calls=[];plan=committed[2];clock=Clock()
    def send(*a,**k):
        calls.append(a[3].copy())
        if len(calls)<3:raise error
        return {'id':plan['id'],'status':'healthy'}
    assert deliver(committed,send,clock)=='acknowledged'
    assert len(calls)==3 and all(p==calls[0] for p in calls)
    assert clock.delays==[1,2]


def test_attempt_budget_persists_and_never_loops_indefinitely(committed):
    calls=[];clock=Clock()
    def send(*a,**k):calls.append(True);raise u.URLError('offline')
    assert deliver(committed,send,clock)=='pending'
    assert deliver(committed,send,clock)=='exhausted'
    assert deliver(committed,send,clock)=='exhausted'
    assert len(calls)==u.ACK_LIMIT and sum(clock.delays)<=31


def test_old_or_invalid_outbox_cannot_claim_success(committed):
    config,path,plan,settings,m=committed;clock=Clock()
    state=u._ack_state(path/'plan.json',plan,clock=clock)
    clock.now+=u.ACK_LIFETIME+1
    assert deliver(committed,lambda *a,**k:pytest.fail('expired acknowledgement sent'),clock)=='exhausted'
    state['id']='wrong';u.private_json(path/'healthy-ack.json',state)
    with pytest.raises(PermissionError):deliver(committed,lambda *a,**k:None,clock)


def test_bad_response_is_not_success(committed):
    assert deliver(committed,lambda *a,**k:{'id':'wrong','status':'healthy'},Clock())=='blocked'


def test_recovery_requires_signed_active_artifact_and_matching_receipt(committed,monkeypatch):
    config,path,plan,settings,m=committed
    monkeypatch.setattr(u,'request',lambda *a,**k:pytest.fail('unauthorized ack'))
    with pytest.raises(PermissionError):u.recover_healthy_ack(config,settings,'other')
    (path/m['filename']).write_bytes(b'changed')
    with pytest.raises(PermissionError):u.recover_healthy_ack(config,settings,'john')


def test_committed_helper_never_relaunches_or_rolls_back_on_ack_error(committed,monkeypatch):
    config,path,plan,settings,m=committed
    monkeypatch.setattr(u,'request',lambda *a,**k:(_ for _ in ()).throw(u.HTTPError('https://fixture.invalid',403,'denied',{},None)))
    monkeypatch.setattr(u,'_pid_running',lambda _:pytest.fail('parent waited after commit'))
    assert u.helper(path/'plan.json',config,settings,launch=lambda *a,**k:pytest.fail('healthy release relaunched'))
    assert json.loads((u.root_for(config)/'active.json').read_text())['release']==plan['release']


def test_real_queue_lost_healthy_response_replays_idempotently(committed,queue,release):
    config,path,plan,settings,m=committed
    enqueue(queue,release);update=signed_node(queue,'poll')['update']
    plan.update(id=update['id'],claim=update['claim']);u.private_json(path/'plan.json',plan)
    calls=[]
    def send(gateway,secret,action,payload,**kw):
        assert action=='report';calls.append(payload.copy())
        result=signed_node(queue,'report',id=payload['id'],claim=payload['claim'],status='healthy')
        if len(calls)==1:raise u.URLError('response lost after commit')
        return result
    assert deliver(committed,send,Clock())=='acknowledged'
    assert calls[0]==calls[1]
    assert queue.link.db.execute('SELECT count(*) FROM app_updates').fetchone()[0]==1
    assert queue.updates.owner('update-status','work',{})['updates'][0]['status']=='healthy'


def test_worker_recovers_before_polling_and_never_polls_when_blocked(monkeypatch):
    worker=u.UpdateWorker.__new__(u.UpdateWorker)
    calls=[];waits=iter([False,True])
    worker.stop=SimpleNamespace(wait=lambda _:next(waits));worker.agent=SimpleNamespace(inference_enabled=True,node_id='john',heartbeat_accepted=SimpleNamespace(is_set=lambda:True),_record_activity=lambda *a:None)
    worker.config_path='fixture';worker.settings={};worker.ack_checked=False;worker.ack_status=None;worker.probe_plan=None
    monkeypatch.setattr(u,'recover_healthy_ack',lambda *a:calls.append('recover') or 'blocked')
    monkeypatch.setattr(u,'request',lambda *a,**k:pytest.fail('polled while acknowledgement blocked'))
    worker.run();assert calls==['recover']


def test_concurrent_helper_and_recovery_share_one_attempt_budget(committed):
    config,path,plan,settings,m=committed
    with u._ack_lock(path/'plan.json') as acquired:
        assert acquired
        assert deliver(committed,lambda *a,**k:pytest.fail('concurrent acknowledgement sent'),Clock())=='pending'
    assert deliver(committed,lambda *a,**k:{'id':plan['id'],'status':'healthy'},Clock())=='acknowledged'


def test_probe_worker_never_polls_before_healthy_pointer_commit(committed,monkeypatch):
    config,path,plan,settings,m=committed
    (u.root_for(config)/'active.json').unlink()
    worker=u.UpdateWorker.__new__(u.UpdateWorker);waits=iter([False,True])
    worker.stop=SimpleNamespace(wait=lambda _:next(waits));worker.agent=SimpleNamespace(inference_enabled=True,node_id='john',heartbeat_accepted=SimpleNamespace(is_set=lambda:True))
    worker.config_path=config;worker.settings=settings;worker.ack_checked=False;worker.ack_status=None;worker.probe_plan=path/'plan.json'
    monkeypatch.setattr(u,'request',lambda *a,**k:pytest.fail('probe polled another update before commit'))
    worker.run()


def test_storage_error_after_pointer_replace_never_rolls_back(committed,monkeypatch):
    config,path,plan,settings,m=committed
    pointer=u.root_for(config)/'active.json';pointer.unlink()
    real=u.private_json
    def write(file,value):
        real(file,value)
        if file==pointer:raise OSError('fixture directory fsync interrupted')
    monkeypatch.setattr(u,'private_json',write)
    monkeypatch.setattr(u,'_pid_running',lambda _:False)
    monkeypatch.setattr(u,'ChildJob',lambda child:SimpleNamespace(kill=lambda:pytest.fail('committed child killed'),close=lambda:None))
    assert u.helper(path/'plan.json',config,settings,launch=lambda *a,**k:SimpleNamespace(poll=lambda:None))
    assert json.loads(pointer.read_text())=={'release':plan['release']}


def test_missing_health_receipt_never_sends_acknowledgement(committed,monkeypatch):
    config,path,plan,settings,m=committed;(path/'healthy.json').unlink()
    monkeypatch.setattr(u,'request',lambda *a,**k:pytest.fail('unproven health acknowledged'))
    with pytest.raises(PermissionError):u.recover_healthy_ack(config,settings,'john')


def test_tls_denial_is_permanent_and_outbox_contains_no_claim_or_secret(committed):
    import ssl
    config,path,plan,settings,m=committed;clock=Clock();calls=[]
    def denied(*a,**k):calls.append(True);raise u.URLError(ssl.SSLCertVerificationError('fixture certificate rejected'))
    assert deliver(committed,denied,clock)=='blocked'
    raw=(path/'healthy-ack.json').read_text()
    assert plan['claim'] not in raw and 'shared_secret' not in raw and calls==[True]
    assert clock.delays==[]
