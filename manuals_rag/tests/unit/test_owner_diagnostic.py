import dataclasses
import json
import socket
import threading
import time
from types import SimpleNamespace
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pytest
from manuals_rag_inference.agent_link import AgentLink
from manuals_rag_inference.gateway import InferenceGateway,create_gateway_handler,node_request_signature
from manuals_rag_inference.models import NodeSnapshot,GpuSnapshot,LoadedModel,ModelSpec
from manuals_rag_inference.owner_diagnostic import OwnerDiagnostic,APPLICATION
from test_inference_node_broker_gateway import serving

OWNER='feb94da3-80b4-4f51-b6aa-3688b9b8e41e'
MODEL='fixture:tiny'
DELL='Johns_Dell_V2'


class Stream:
    def __init__(self,events=None):
        self.events=events or [{'response':{'response':'PROVIS_DIAGNOSTIC_OK','done':True}}]
        self.closed=False;self.cancelled=threading.Event()
    def __iter__(self):yield from self.events
    def close(self):self.closed=True;self.cancel()
    def cancel(self):self.cancelled.set()


@pytest.fixture
def diag():
    records=[SimpleNamespace(node_id=n,secret='secret-'+n,enrollment_method='entra',identity_subject='owner-sub',identity_tenant='tenant',agent_url='http://'+n,allowed_applications=(APPLICATION,)) for n in ('source',DELL,'other')]
    enrollment=SimpleNamespace(nodes=lambda:records)
    g=InferenceGateway(enrollment=enrollment,node_default_applications=(APPLICATION,),
        model_catalog={MODEL:ModelSpec(1000,100,('chat',))},application_worker_allowlist={APPLICATION:(DELL,'other')})
    for n in (DELL,'other'):
        g.scheduler.upsert_node(NodeSnapshot(n,'http://'+n,'fixture',(GpuSnapshot('gpu','fixture',8192,100),),
            loaded_models=(LoadedModel(MODEL,1000),),allowed_applications=(APPLICATION,),
            observed_at_epoch=time.time(),installed_models=(MODEL,),node_app_version='2026.10.20.3',
            capabilities=('chat','diagnostic_loaded_only_v1')))
    validator=SimpleNamespace(validate=lambda t:dict(object_id=OWNER if t=='owner' else 'employee',subject='owner-sub',tenant_id='tenant',expires_at_epoch=time.time()+100))
    link=AgentLink(validator,enrollment,owner_object_id=OWNER,path=':memory:');g.agent_link=link
    token=link.session('owner',{'node_id':'source'})['session_token'];calls=[];streams=[]
    def factory(url,envelope,timeout):
        calls.append((url,envelope,timeout));s=Stream();streams.append(s);return s
    service=OwnerDiagnostic(g,stream_factory=factory);link.diagnostic=service
    g._prepare_node=lambda *a:pytest.fail('diagnostic prepared a model')
    yield SimpleNamespace(g=g,link=link,token=token,records=records,service=service,calls=calls,streams=streams)
    link.db.close()


def run(d,**extras):
    return d.link.dispatch('diagnostic-inference',d.token,{'target_node_id':DELL,'model':MODEL,**extras})


def change(d,**values):
    node=next(n for n in d.g.scheduler.snapshots() if n.node_id==DELL)
    d.g.scheduler.upsert_node(dataclasses.replace(node,**values))


def test_exact_target_signed_transport_and_server_generated_receipt(diag):
    result=run(diag);receipt=result['receipt']
    assert receipt['node_id']==DELL==receipt['target_node_id']
    assert receipt['model']==MODEL and receipt['node_version']=='2026.10.20.3'
    assert receipt['request_id'].startswith('owner-diagnostic:') and receipt['completed']
    url,envelope,timeout=diag.calls[0]
    assert url=='http://'+DELL+'/v1/node/inference' and 0<timeout<=10
    signature=envelope['signature'];unsigned={k:v for k,v in envelope.items() if k!='signature'}
    assert signature==node_request_signature(unsigned,('secret-'+DELL).encode())
    assert envelope['diagnostic_loaded_only'] is True
    assert envelope['payload']['options']=={'num_predict':8,'temperature':0}
    assert diag.streams[0].closed and diag.g.scheduler._reservations=={}


@pytest.mark.parametrize('token',['employee','operator-token','application-token','secret-source','expired'])
def test_only_current_owner_session_can_run(diag,token):
    with pytest.raises(PermissionError):diag.link.dispatch('diagnostic-inference',token,{'target_node_id':DELL,'model':MODEL})
    assert not diag.calls


def test_disabled_by_default(diag):
    diag.link.diagnostic=None
    with pytest.raises(PermissionError,match='disabled'):run(diag)


def test_other_workspace_and_enrollment_change_rejected(diag):
    diag.records[1].identity_subject='different-owner'
    with pytest.raises(PermissionError,match='workspace'):run(diag)
    diag.records[0].secret='rotated'
    with pytest.raises(PermissionError,match='enrollment'):run(diag)
    assert not diag.calls


@pytest.mark.parametrize('values',[
    {'healthy':False},{'inference_enabled':False},{'observed_at_epoch':1},
    {'observed_at_epoch':float('inf')},{'active_requests':1},{'queued_requests':1},
    {'installed_models':()},{'loaded_models':()},{'capabilities':('chat',)},
    {'node_app_version':''},{'allowed_applications':()},
])
def test_unavailable_ineligible_or_unloaded_target_never_falls_back(diag,values):
    change(diag,**values)
    with pytest.raises(PermissionError):run(diag)
    assert not diag.calls and diag.g.scheduler._reservations=={}


def test_server_worker_allowlist_is_required_and_never_widened(diag):
    diag.g.application_worker_allowlist={APPLICATION:('other',)}
    with pytest.raises(PermissionError):run(diag)
    diag.g.application_worker_allowlist={}
    with pytest.raises(PermissionError):run(diag)
    assert not diag.calls


@pytest.mark.parametrize('extra',[{'prompt':'arbitrary'},{'allow_model_pull':True},{'allow_eviction':True},{'node_id':'other'},{'request_id':'spoofed'},{'timeout':1000}])
def test_caller_cannot_override_limits_policy_or_receipt(diag,extra):
    with pytest.raises(ValueError):run(diag,**extra)
    assert not diag.calls


def test_unknown_or_wrong_catalog_model_rejected(diag):
    with pytest.raises(PermissionError):diag.link.dispatch('diagnostic-inference',diag.token,{'target_node_id':DELL,'model':'missing'})
    diag.g.model_catalog[MODEL]=ModelSpec(1000,100,('chat',),backend='onnx')
    with pytest.raises(PermissionError):run(diag)


def test_worker_attribution_is_not_taken_from_upstream_output(diag):
    diag.service.stream_factory=lambda *a,**k:Stream([{'node_id':'other','routing':{'node_id':'other'},'response':{'response':'OK','done':True,'node_id':'other'}}])
    assert run(diag)['receipt']['node_id']==DELL


@pytest.mark.parametrize('failure',[TimeoutError('fixture'),OSError('fixture'),ValueError('fixture')])
def test_transport_failure_releases_reservation_and_slot(diag,failure):
    def broken(*a,**k):raise failure
    diag.service.stream_factory=broken
    with pytest.raises((RuntimeError,TimeoutError,ValueError)):run(diag)
    assert diag.g.scheduler._reservations=={}
    assert diag.service.slot.acquire(blocking=False);diag.service.slot.release()


def test_rejects_wrong_scheduler_decision_and_releases_it(diag,monkeypatch):
    original=diag.g.scheduler.choose_and_reserve
    def wrong(request):return original(dataclasses.replace(request,allowed_node_ids=('other',)))
    monkeypatch.setattr(diag.g.scheduler,'choose_and_reserve',wrong)
    with pytest.raises(PermissionError):run(diag)
    assert not diag.calls and diag.g.scheduler._reservations=={}


def test_deadline_and_disconnect_during_stream_cleanup(diag,monkeypatch):
    clock=[0.];diag.service.clock=lambda:clock[0]
    class Slow(Stream):
        def __iter__(self):clock[0]=11;yield {'response':{'response':'late','done':True}}
    stream=Slow();diag.service.stream_factory=lambda *a,**k:stream
    with pytest.raises(TimeoutError):run(diag)
    assert stream.closed and diag.g.scheduler._reservations=={}
    clock[0]=0;cancelled=threading.Event()
    class Disconnected(Stream):
        def __iter__(self):cancelled.set();yield {'response':{'response':'late','done':True}}
    stream=Disconnected();diag.service.stream_factory=lambda *a,**k:stream
    with pytest.raises(ConnectionAbortedError):diag.link.dispatch('diagnostic-inference',diag.token,{'target_node_id':DELL,'model':MODEL},cancelled=cancelled)
    assert stream.closed and diag.g.scheduler._reservations=={}


def test_http_owner_authorization_and_receipt(diag):
    with serving(create_gateway_handler(diag.g)) as url:
        body=json.dumps({'target_node_id':DELL,'model':MODEL}).encode()
        request=Request(url+'/v1/agent-link/diagnostic-inference',data=body,headers={'Authorization':'Bearer '+diag.token})
        with urlopen(request,timeout=2) as response:r=json.load(response)
        assert r['receipt']['node_id']==DELL
        with pytest.raises(HTTPError) as error:urlopen(Request(url+'/v1/agent-link/diagnostic-inference',data=body),timeout=2)
        assert error.value.code in (401,403)
    assert diag.g.scheduler._reservations=={}


def test_http_disconnect_cancels_upstream_and_releases_reservation(diag):
    ready=threading.Event()
    class Waiting(Stream):
        def __iter__(self):
            ready.set();assert self.cancelled.wait(2),'upstream was not cancelled'
            raise ConnectionAbortedError('fixture disconnected')
            yield
    stream=Waiting();diag.service.stream_factory=lambda *a,**k:stream
    with serving(create_gateway_handler(diag.g)) as url:
        hostport=url.removeprefix('http://');host,port=hostport.split(':')
        body=json.dumps({'target_node_id':DELL,'model':MODEL}).encode()
        sock=socket.create_connection((host,int(port)),timeout=2)
        request=f'POST /v1/agent-link/diagnostic-inference HTTP/1.0\r\nHost: fixture\r\nAuthorization: Bearer {diag.token}\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body
        sock.sendall(request);assert ready.wait(2);sock.close()
        deadline=time.monotonic()+2
        while diag.g.scheduler._reservations and time.monotonic()<deadline:time.sleep(.02)
        assert diag.g.scheduler._reservations=={} and stream.closed


def test_owner_revoked_during_operation_does_not_receive_success(diag):
    def factory(*a,**k):diag.records[0].secret='rotated';return Stream()
    diag.service.stream_factory=factory
    with pytest.raises(PermissionError):run(diag)
    assert diag.g.scheduler._reservations=={}


@pytest.mark.parametrize('loaded',[True,False])
def test_worker_rechecks_residency_without_warm_or_pull(monkeypatch,loaded):
    from manuals_rag_inference import node_agent as module
    node=module.OllamaNode(node_id=DELL,endpoint='http://127.0.0.1:1',tailnet_name='fixture',secret=b'fixture')
    node.set_inference_enabled(True);calls=[]
    def inventory(url,**kwargs):
        calls.append(url);assert url.endswith('/api/ps')
        return {'models':[{'name':MODEL,'context_length':4096}]} if loaded else {'models':[]}
    monkeypatch.setattr(module,'_json_request',inventory)
    stream=Stream();stream.done=True
    def dispatch(url,body,**kwargs):
        assert loaded, 'unloaded inference started'
        assert body['options']['num_ctx']==4096  # preserve resident context
        return stream
    monkeypatch.setattr(module,'NDJSONResponse',dispatch)
    body={'operation':'generate','request_id':'owner-diagnostic:fixture','issued_at':time.time(),
          'diagnostic_loaded_only':True,'payload':{'model':MODEL,'prompt':'Reply PROVIS_DIAGNOSTIC_OK only.','stream':True,
                                                  'options':{'num_predict':8,'temperature':0}}}
    signature=node_request_signature(body,b'fixture')
    if loaded:
        with node.stream_infer(body,signature):assert node._active_by_model[MODEL]==1
        assert stream.closed
    else:
        with pytest.raises(PermissionError,match='no longer loaded'):
            with node.stream_infer(body,signature):pytest.fail('unloaded inference admitted')
    assert calls==['http://127.0.0.1:1/api/ps'] and node._active_by_model=={}


def test_worker_constraint_cannot_be_added_after_signing():
    from manuals_rag_inference.node_agent import OllamaNode
    node=OllamaNode(node_id=DELL,endpoint='http://127.0.0.1:1',tailnet_name='fixture',secret=b'fixture')
    body={'operation':'generate','request_id':'owner-diagnostic:fixture','issued_at':time.time(),'payload':{'model':MODEL}}
    signature=node_request_signature(body,b'fixture');body['diagnostic_loaded_only']=True
    with pytest.raises(PermissionError,match='signature'):node._authorize_inference(body,signature)


def test_real_ndjson_deadline_releases_reservations(diag,monkeypatch):
    from http.server import BaseHTTPRequestHandler
    from manuals_rag_inference import owner_diagnostic as module
    class SlowWorker(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson');self.end_headers()
            self.wfile.write(b'{"response":{"response":"first","done":false}}\n');self.wfile.flush()
            time.sleep(.3)
        def log_message(self,*args):pass
    monkeypatch.setattr(module,'TIMEOUT',.1)
    with serving(SlowWorker) as url:
        diag.g.node_agent_urls[DELL]=url;diag.service.stream_factory=None
        started=time.monotonic()
        with pytest.raises((TimeoutError,RuntimeError,ValueError)):run(diag)
        assert time.monotonic()-started<1.5
    assert diag.g.scheduler._reservations=={}


@pytest.mark.parametrize('context',[None,0,16384,True])
def test_worker_rejects_unknown_resident_context_instead_of_reloading(monkeypatch,context):
    from manuals_rag_inference import node_agent as module
    node=module.OllamaNode(node_id=DELL,endpoint='http://127.0.0.1:1',tailnet_name='fixture',secret=b'fixture')
    node.set_inference_enabled(True)
    monkeypatch.setattr(module,'_json_request',lambda *a,**k:{'models':[{'name':MODEL,'context_length':context}]})
    monkeypatch.setattr(module,'NDJSONResponse',lambda *a,**k:pytest.fail('would reload unknown context'))
    body={'operation':'generate','request_id':'owner-diagnostic:fixture','issued_at':time.time(),'diagnostic_loaded_only':True,
          'payload':{'model':MODEL,'prompt':'Reply PROVIS_DIAGNOSTIC_OK only.','stream':True,'options':{'num_predict':8,'temperature':0}}}
    with pytest.raises(PermissionError,match='context'):
        with node.stream_infer(body,node_request_signature(body,b'fixture')):pytest.fail('unavailable context admitted')
    assert node._active_by_model=={}
