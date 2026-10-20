import sys
import types

import pytest

from manuals_rag_inference.onnx_runtime import runtime_status, device_choices, require_bundled_runtime
from manuals_rag_inference.onnx_backend import OnnxBackend, OnnxBackendError
from manuals_rag_inference.node_agent import OllamaNode
from manuals_rag_inference import onnx_runtime


def install_fake_runtime(monkeypatch, devices=('CPU',), providers=('OpenVINOExecutionProvider', 'CPUExecutionProvider')):
    monkeypatch.setitem(sys.modules, 'numpy', types.ModuleType('numpy'))
    ort = types.SimpleNamespace(__version__='test', get_available_providers=lambda: list(providers))
    core = types.SimpleNamespace(available_devices=list(devices), get_property=lambda device, key: 'Intel '+device)
    monkeypatch.setitem(sys.modules, 'onnxruntime', ort)
    monkeypatch.setitem(sys.modules, 'openvino', types.SimpleNamespace(Core=lambda: core))
    monkeypatch.setattr(onnx_runtime, 'get_device_inventory', lambda: {'devices': onnx_runtime.detected_openvino_devices(core)})


def test_provider_installation_does_not_fabricate_an_intel_gpu(monkeypatch):
    install_fake_runtime(monkeypatch)
    status = runtime_status()
    assert status['status'] == 'cpu_ready'
    assert status['intel_gpus'] == []
    assert device_choices(status) == ('AUTO', 'CPU')
    backend = OnnxBackend()
    assert backend.providers
    assert not backend.enabled  # No assigned model yet.
    assert backend.gpu_devices() == ()


def test_real_detected_gpu_is_available_without_a_model_manifest(monkeypatch):
    install_fake_runtime(monkeypatch, ('CPU', 'GPU.0'))
    backend = OnnxBackend()
    assert backend.gpu_devices() == ('openvino:GPU.0',)
    assert runtime_status()['intel_gpus'] == [{'id': 'GPU.0', 'name': 'Intel GPU.0'}]
    assert backend.model_names() == ()


def test_device_change_clears_sessions_and_rejects_undetected_hardware(monkeypatch):
    install_fake_runtime(monkeypatch, ('CPU', 'GPU.0'))
    backend = OnnxBackend()
    backend._sessions['test'] = object()
    backend.set_device('CPU')
    assert backend.gpu_devices() == ()
    assert backend._sessions == {}
    with pytest.raises(OnnxBackendError):
        backend.set_device('GPU.9')
    assert backend.device_type == 'CPU'


def test_missing_runtime_is_reported_without_installing_anything(monkeypatch):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    status = runtime_status()
    assert not status['installed']
    assert status['status'] == 'not_installed'
    assert status['error_category'] == 'ModuleNotFoundError'


def test_release_build_rejects_missing_openvino_provider(monkeypatch):
    install_fake_runtime(monkeypatch, providers=('CPUExecutionProvider',))
    with pytest.raises(RuntimeError, match='must include'):
        require_bundled_runtime()


def test_signed_release_requires_successful_device_enumeration(monkeypatch):
    install_fake_runtime(monkeypatch)
    def fail():
        raise OSError('private native diagnostic must not escape')
    monkeypatch.setitem(sys.modules, 'openvino', types.SimpleNamespace(Core=fail))
    monkeypatch.setattr(onnx_runtime, 'get_device_inventory', fail)
    status = runtime_status()
    assert 'private' not in str(status)
    with pytest.raises(RuntimeError, match='enumerate'):
        require_bundled_runtime()


def test_non_intel_opencl_gpu_is_not_advertised_as_an_intel_accelerator(monkeypatch):
    install_fake_runtime(monkeypatch, ('CPU', 'GPU.0'))
    core = types.SimpleNamespace(available_devices=['CPU', 'GPU.0'], get_property=lambda device, key: 'CPU' if device == 'CPU' else 'NVIDIA GeForce RTX 3080 Ti')
    monkeypatch.setitem(sys.modules, 'openvino', types.SimpleNamespace(Core=lambda: core))
    monkeypatch.setattr(onnx_runtime, 'get_device_inventory', lambda: {'devices': onnx_runtime.detected_openvino_devices(core)})
    assert runtime_status()['intel_gpus'] == []
    backend = OnnxBackend()
    assert backend.gpu_devices() == ()
    assert backend._provider_configuration()[1][0]['device_type'] == 'CPU'


def test_device_setting_preserves_enrollment_and_cannot_interrupt_active_work(monkeypatch, tmp_path):
    import json
    install_fake_runtime(monkeypatch, ('CPU', 'GPU.0'))
    node = OllamaNode(node_id='fixture', endpoint='http://127.0.0.1:11434', tailnet_name='fixture', secret=b'fixture-secret')
    node.config_path = str(tmp_path/'node.json')
    original = {'node_id': 'fixture', 'shared_secret': 'fixture-private', 'allowed_applications': ['openclaw']}
    (tmp_path/'node.json').write_text(json.dumps(original))
    node._active_by_model['fixture'] = 1
    with pytest.raises(PermissionError):
        node.set_onnx_device('GPU.0')
    assert json.loads((tmp_path/'node.json').read_text()) == original
    node._active_by_model.clear()
    node.set_onnx_device('GPU.0')
    assert json.loads((tmp_path/'node.json').read_text()) == {**original, 'onnx_device': 'GPU.0'}
    monkeypatch.setenv('INFERENCE_ONNX_DEVICE', 'CPU')
    with pytest.raises(PermissionError):
        node.set_onnx_device('AUTO')


def test_discovery_uses_a_bounded_fixed_helper_without_enrollment_arguments(monkeypatch):
    monkeypatch.setattr(onnx_runtime, '_inventory_cache', None)
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    calls = []
    def run(command, **options):
        calls.append((command, options))
        return types.SimpleNamespace(returncode=0, stdout=b'{"devices":[{"id":"CPU","name":"CPU"}]}')
    monkeypatch.setattr(onnx_runtime.subprocess, 'run', run)
    assert onnx_runtime.get_device_inventory()['devices'][0]['id'] == 'CPU'
    assert onnx_runtime.get_device_inventory()['devices'][0]['id'] == 'CPU'
    assert len(calls) == 1
    assert calls[0][0] == [sys.executable, '--accelerator-probe']
    assert calls[0][1]['timeout'] == 20
    assert not calls[0][1].get('shell')


def test_discovery_does_not_import_python_openvino_into_onnx_process(monkeypatch):
    install_fake_runtime(monkeypatch)
    original = onnx_runtime.importlib.import_module
    def guarded_import(name, *args, **kwargs):
        assert name != 'openvino', 'Python SDK must stay in the helper process'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(onnx_runtime.importlib, 'import_module', guarded_import)
    assert runtime_status()['status'] == 'cpu_ready'
