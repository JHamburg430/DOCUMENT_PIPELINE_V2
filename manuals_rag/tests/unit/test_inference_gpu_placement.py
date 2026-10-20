import time

from manuals_rag_inference.models import GpuSnapshot, InferenceRequest, NodeSnapshot
from manuals_rag_inference.scheduler import InferenceScheduler


def _node(*gpus, providers=()):
    return NodeSnapshot(
        node_id="worker",
        endpoint="http://worker:11434",
        tailnet_name="tailnet",
        gpus=tuple(gpus),
        observed_at_epoch=time.time(),
        capabilities=("chat", "embeddings", "onnx") if providers else ("chat", "embeddings"),
        onnx_providers=tuple(providers),
        installed_models=("model",),
        host_resources={'observed_at_epoch':time.time(),'memory_available_bytes':32*1024**3},
    )


def _request(**overrides):
    values = dict(
        request_id="request",
        application="manuals_rag",
        workload="chat",
        model="model",
        model_size_bytes=1,
        min_vram_mb=12000,
        required_capabilities=("chat",),
    )
    values.update(overrides)
    return InferenceRequest(**values)


def test_multi_gpu_is_default_and_reserves_per_device():
    scheduler = InferenceScheduler()
    scheduler.upsert_node(
        _node(
            GpuSnapshot("uuid-a", "GPU A", 16000, 0, gpu_uuid="uuid-a", gpu_index=0),
            GpuSnapshot("uuid-b", "GPU B", 16000, 0, gpu_uuid="uuid-b", gpu_index=1),
        )
    )

    decision = scheduler.choose_and_reserve(_request(min_vram_mb=18000))

    assert decision.admitted
    assert decision.placement is not None
    assert decision.placement.mode == "multi_gpu"
    assert decision.placement.gpu_ids == ("uuid-a", "uuid-b")
    assert sum(decision.placement.vram_by_gpu_mb.values()) == 18000


def test_single_gpu_can_be_requested_explicitly():
    scheduler = InferenceScheduler()
    scheduler.upsert_node(
        _node(
            GpuSnapshot("uuid-a", "GPU A", 16000, 0),
            GpuSnapshot("uuid-b", "GPU B", 16000, 0),
        )
    )

    decision = scheduler.choose(_request(min_vram_mb=18000, placement_mode="single_gpu"))

    assert not decision.admitted
    assert "insufficient_vram_on_single_gpu" in decision.reason


def test_reservations_keep_concurrent_requests_on_separate_gpus_when_possible():
    scheduler = InferenceScheduler()
    scheduler.upsert_node(
        _node(
            GpuSnapshot("uuid-a", "GPU A", 16000, 0),
            GpuSnapshot("uuid-b", "GPU B", 16000, 0),
        )
    )

    first = scheduler.choose_and_reserve(_request(request_id="first"))
    second = scheduler.choose_and_reserve(_request(request_id="second"))

    assert first.admitted and second.admitted
    assert first.placement is not None and second.placement is not None
    assert set(first.placement.gpu_ids).isdisjoint(second.placement.gpu_ids)


def test_onnx_openvino_device_can_be_scheduled_without_fake_vram_numbers():
    scheduler = InferenceScheduler()
    scheduler.upsert_node(
        _node(
            GpuSnapshot(
                "openvino:GPU.0",
                "Intel integrated GPU",
                0,
                0,
                backend="openvino",
                memory_unknown=True,
            ),
            providers=("OpenVINOExecutionProvider",),
        )
    )

    decision = scheduler.choose(_request(backend="onnx", required_capabilities=("onnx",)))

    assert decision.admitted
    assert decision.placement is not None
    assert decision.placement.backend == "onnx"
    assert decision.placement.gpu_ids == ("openvino:GPU.0",)


def test_node_gpu_probe_preserves_uuid_fields_with_power_telemetry_schema(monkeypatch):
    import types
    from manuals_rag_inference import node_agent
    monkeypatch.setattr(node_agent.subprocess,"run",lambda *args,**kwargs: types.SimpleNamespace(stdout="0, GPU-stable-a, RTX 3090, 24576, 1024, 5, 40\n1, GPU-stable-b, RTX 3090, 24576, 2048, 6, 41\n"))
    gpus=node_agent._gpu_snapshots()
    assert len(gpus)==2
    assert [g.gpu_uuid for g in gpus]==["GPU-stable-a","GPU-stable-b"]
    assert [g.gpu_index for g in gpus]==[0,1]
    assert all(g.backend=="cuda" and g.power_w is None for g in gpus)
