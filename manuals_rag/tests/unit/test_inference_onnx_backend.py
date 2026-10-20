import sys
import types

from manuals_rag_inference.onnx_backend import OnnxBackend


def test_onnx_backend_is_disabled_without_a_manifest(monkeypatch):
    monkeypatch.delenv("INFERENCE_ONNX_MODELS_JSON", raising=False)

    backend = OnnxBackend.from_env()

    assert not backend.enabled
    assert backend.model_names() == ()


def test_openvino_provider_configuration_prefers_openvino(monkeypatch):
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setenv("INFERENCE_ONNX_DEVICE", "GPU.1")
    backend = OnnxBackend({"model": {"path": "/tmp/model.onnx"}})

    providers, options = backend._provider_configuration()

    assert providers == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert options[0] == {"device_type": "GPU.1"}
