import json,time
import pytest
from manuals_rag_inference.gateway import InferenceGateway, _snapshot_from_payload, node_request_signature
from manuals_rag_inference.models import GpuSnapshot,LoadedModel,NodeSnapshot,ModelSpec,InferenceRequest

@pytest.mark.parametrize("mode", ["", "multi_gpu", "single_gpu"])
def test_old_and_new_nodes_get_matching_signed_envelopes(mode):
    sent=[]
    def post(url, body, **kw):
        signature=body.pop("signature")
        expected={k:body[k] for k in ("operation","payload","request_id","issued_at")}
        if mode:
            expected["placement"]=body["placement"]
            assert expected["placement"]["gpu_ids"] == ("GPU-stable",)
        else:
            assert "placement" not in body
        assert signature == node_request_signature(expected,b"fixture-secret")
        sent.append(body)
        return {"ok": True}
    g=InferenceGateway(node_secrets={"worker":b"fixture-secret"},node_agent_urls={"worker":"http://worker:8765"},post=post,require_node_agents=True,model_catalog={"m":ModelSpec(1,1024,("chat",))},require_model_catalog=True)
    g.scheduler.upsert_node(NodeSnapshot("worker","http://worker:11434","tailnet",(GpuSnapshot("GPU-stable","gpu",24000,0,gpu_uuid="GPU-stable"),),(LoadedModel("m",1,("GPU-stable",)),),allowed_applications=("openclaw",),capabilities=("chat",),observed_at_epoch=time.time(),gpu_placement_mode=mode))
    r=g.infer(InferenceRequest("r","openclaw","chat","m",1,1024,required_capabilities=("chat",)),operation="chat",payload={"model":"m"})
    assert r["response"]["ok"] and len(sent)==1
    assert not g.scheduler._reservations

def test_new_heartbeat_preserves_uuid_and_placement_facts():
    p={"node_id":"john-System-Product-Name","endpoint":"http://127.0.0.1:11434","gpus":[{"gpu_id":"GPU-stable","name":"RTX3090","total_vram_mb":24576,"used_vram_mb":0,"gpu_uuid":"GPU-stable","gpu_index":0,"backend":"cuda","memory_unknown":False}],"loaded_models":[{"name":"m","size_bytes":1,"size_vram_bytes":1,"vram_by_gpu_mb":{"GPU-stable":1},"placement_known":True}],"gpu_placement_mode":"multi_gpu","onnx_providers":["OpenVINOExecutionProvider"]}
    # Match the existing LoadedModel wire field names.
    p["loaded_models"][0].pop("size_vram_bytes")
    n=_snapshot_from_payload(p)
    assert n.gpus[0].gpu_uuid=="GPU-stable"
    assert n.gpus[0].gpu_index==0
    assert n.gpu_placement_mode=="multi_gpu"
    assert n.loaded_models[0].placement_known
    assert n.onnx_providers==("OpenVINOExecutionProvider",)
    p.pop("gpu_placement_mode");p.pop("onnx_providers")
    assert _snapshot_from_payload(p).gpu_placement_mode==""
