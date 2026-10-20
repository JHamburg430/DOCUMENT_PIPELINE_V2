"""Normal production routing stays dynamic, eligible-only, and policy-owned."""
import dataclasses
import time
import pytest
from manuals_rag_inference.models import GpuSnapshot,LoadedModel,InferenceRequest,NodeSnapshot
from manuals_rag_inference.scheduler import InferenceScheduler
from test_owner_diagnostic import diag,APPLICATION,MODEL,DELL


def request(**changes):
    return dataclasses.replace(InferenceRequest('fixture',APPLICATION,'chat',MODEL,1000,4096),**changes)


def node(name='worker',**changes):
    return dataclasses.replace(NodeSnapshot(name,'http://'+name,'fixture',(GpuSnapshot('gpu','fixture',24000,0),),
        loaded_models=(LoadedModel(MODEL,1000),),installed_models=(MODEL,),allowed_applications=(APPLICATION,),
        observed_at_epoch=time.time()),**changes)


def test_normal_routing_is_dynamic_and_diagnostic_is_not_a_prerequisite(diag):
    diag.link.diagnostic=None;diag.g._prepare_node=lambda *a:None
    dell=next(n for n in diag.g.scheduler.snapshots() if n.node_id==DELL)
    diag.g.scheduler.upsert_node(dataclasses.replace(dell,gpus=(GpuSnapshot('gpu','fixture',8192,100,utilization_percent=100),)))
    calls=[];diag.g._post=lambda url,body,**kw:calls.append(url) or {'response':'fixture'}
    # A client cannot force Dell or widen the central application allowlist.
    result=diag.g.infer(request(allowed_node_ids=(DELL,)),operation='generate',payload={'model':MODEL})
    assert result['routing']['node_id']=='other' and calls==['http://other/v1/node/inference']
    assert diag.g.scheduler._reservations=={}


def test_application_allowlist_is_never_broadened_for_failover(diag):
    diag.g.application_worker_allowlist={APPLICATION:(DELL,)}
    current=next(n for n in diag.g.scheduler.snapshots() if n.node_id==DELL)
    diag.g.scheduler.upsert_node(dataclasses.replace(current,healthy=False))
    diag.g._post=lambda *a,**k:pytest.fail('outside-allowlist fallback')
    with pytest.raises(RuntimeError):diag.g.infer(request(),operation='generate',payload={})
    assert diag.g.scheduler._reservations=={}


@pytest.mark.parametrize('change',[
    {'healthy':False},{'inference_enabled':False},{'observed_at_epoch':1},
    {'observed_at_epoch':float('nan')},{'observed_at_epoch':float('inf')},
    {'observed_at_epoch':time.time()+60},{'active_requests':8},{'queued_requests':8},
    {'allowed_applications':('different-app',)},{'capabilities':('embeddings',)},
    {'gpus':(GpuSnapshot('gpu','fixture',1000,900),)},
    {'gpus':(GpuSnapshot('gpu','fixture',24000,0,healthy=False),)},
    {'gpus':(GpuSnapshot('intel','fixture',24000,0,backend='openvino'),)},
])
def test_ineligible_worker_never_beats_eligible_failover(change):
    scheduler=InferenceScheduler();scheduler.upsert_node(node('bad',**change));scheduler.upsert_node(node('good'))
    decision=scheduler.choose_and_reserve(request())
    assert decision.node_id=='good'
    assert scheduler.release('good','fixture')
    assert scheduler._reservations=={} and scheduler._reservation_placements=={}


def test_resident_known_model_memory_is_reusable_without_inventing_headroom():
    scheduler=InferenceScheduler()
    gpu=GpuSnapshot('stable','fixture',24000,22000)
    loaded=LoadedModel(MODEL,22000*1024**2,gpu_ids=('stable',),vram_by_gpu_mb={'stable':22000},placement_known=True)
    scheduler.upsert_node(node(gpus=(gpu,),loaded_models=(loaded,)))
    req=request(min_vram_mb=18000)
    first=scheduler.choose_and_reserve(req);assert first.admitted and not first.requires_model_warm
    assert not scheduler.choose(dataclasses.replace(req,request_id='second')).admitted
    scheduler.release('worker',req.request_id)
    assert scheduler.choose(req).admitted
    scheduler.upsert_node(node(gpus=(gpu,),loaded_models=(dataclasses.replace(loaded,placement_known=False),)))
    assert not scheduler.choose(req).admitted  # unknown placement gets no credit


def test_backend_and_minimum_gpu_count_are_enforced():
    scheduler=InferenceScheduler();scheduler.upsert_node(node())
    assert not scheduler.choose(request(min_gpu_count=2,placement_mode='single_gpu')).admitted
    assert not scheduler.choose(request(backend='onnx',required_capabilities=('onnx',))).admitted


def shared_node(resources):
    return node(gpus=(GpuSnapshot('intel','Intel',0,0,backend='openvino',memory_unknown=True),),
        capabilities=('onnx',),onnx_providers=('OpenVINOExecutionProvider',),host_resources=resources)


@pytest.mark.parametrize('resources',[None,{}, {'memory_available_bytes':100,'observed_at_epoch':time.time()},
    {'memory_available_bytes':32*1024**3,'observed_at_epoch':1}])
def test_unknown_shared_memory_needs_fresh_measured_headroom(resources):
    scheduler=InferenceScheduler();scheduler.upsert_node(shared_node(resources))
    assert not scheduler.choose(request(backend='onnx',required_capabilities=('onnx',))).admitted


def test_shared_memory_reservations_prevent_overcommit_and_clean_up():
    scheduler=InferenceScheduler();scheduler.upsert_node(shared_node({'memory_available_bytes':20*1024**3,'observed_at_epoch':time.time()}))
    req=request(backend='onnx',required_capabilities=('onnx',),min_vram_mb=12000)
    assert scheduler.choose_and_reserve(req).admitted
    assert not scheduler.choose(dataclasses.replace(req,request_id='another')).admitted
    scheduler.release('worker',req.request_id)
    assert scheduler._reservation_shared_bytes=={} and scheduler.choose(req).admitted


def test_fresh_load_telemetry_breaks_equal_model_placement_ties():
    scheduler=InferenceScheduler()
    scheduler.upsert_node(node('loaded-host',host_resources={'cpu_percent':95,'observed_at_epoch':time.time()}))
    scheduler.upsert_node(node('idle-host',host_resources={'cpu_percent':5,'observed_at_epoch':time.time()}))
    assert scheduler.choose(request()).node_id=='idle-host'
