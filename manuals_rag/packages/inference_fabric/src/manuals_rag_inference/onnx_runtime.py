"""Content-free diagnostics for the runtime shipped in signed node apps."""
from __future__ import annotations

import importlib
import base64
import re
import sys

INTEL_DRIVER_PAGE = 'https://www.intel.com/content/www/us/en/support/detect.html'


def detected_openvino_devices(core) -> list[dict]:
    devices = []
    for device in core.available_devices:
        device = str(device)
        if not re.fullmatch(r'(CPU|GPU|NPU)(\.\d+)?', device):
            continue
        try:
            name = str(core.get_property(device, 'FULL_DEVICE_NAME'))[:160]
        except Exception:
            name = device
        # OpenCL enumeration can include NVIDIA/AMD devices. Do not claim those
        # are supported Intel GPUs, or select them with OpenVINO AUTO.
        if device.startswith('GPU') and 'intel' not in name.lower():
            continue
        devices.append({'id': device, 'name': name})
    return devices


def runtime_status() -> dict:
    result = {'installed': False, 'status': 'not_installed', 'version': '', 'providers': [],
              'devices': [], 'intel_gpus': [], 'bundled': bool(getattr(sys, 'frozen', False))}
    try:
        ort = importlib.import_module('onnxruntime')
        importlib.import_module('numpy')
        result.update(installed=True, version=str(ort.__version__), providers=list(ort.get_available_providers()))
    except (ImportError, OSError, AttributeError) as error:
        result['error_category'] = type(error).__name__
        return result
    try:
        ov = importlib.import_module('openvino')
        core = ov.Core()
        result['devices'] = detected_openvino_devices(core)
        result['intel_gpus'] = [item for item in result['devices'] if item['id'].startswith('GPU')]
    except Exception as error:
        result['device_error_category'] = type(error).__name__
    result['status'] = ('intel_gpu_ready' if result['intel_gpus'] else 'cpu_ready') if 'OpenVINOExecutionProvider' in result['providers'] else 'provider_missing'
    return result


def device_choices(status: dict) -> tuple[str, ...]:
    return ('AUTO',) + tuple(item['id'] for item in status.get('devices', []) if item.get('id'))


def require_bundled_runtime() -> dict:
    """Fail a release build/self-test if a required runtime is missing."""
    import cryptography.hazmat.primitives.asymmetric.ed25519  # noqa: F401
    status = runtime_status()
    if not status['installed'] or 'OpenVINOExecutionProvider' not in status['providers']:
        raise RuntimeError('The node build must include ONNX Runtime, OpenVINO and NumPy')
    if 'device_error_category' in status:
        raise RuntimeError('The bundled OpenVINO runtime cannot enumerate devices')
    return status


def probe_runtime(device: str = 'CPU') -> dict:
    """Run a bundled one-element Add graph, never employee/model content."""
    status = require_bundled_runtime()
    if device not in device_choices(status):
        raise ValueError('The selected accelerator is unavailable')
    if device == 'AUTO':
        targets = [item['id'] for item in status['devices'] if item['id'].startswith(('GPU', 'NPU'))]
        device = targets[0] if targets else 'CPU'
    ort = importlib.import_module('onnxruntime')
    np = importlib.import_module('numpy')
    options = ort.SessionOptions()
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    model = base64.b64decode('CAg6SAoOCgF4CgF4EgF5IgNBZGQSFHByb3Zpcy1ydW50aW1lLXByb2JlWg8KAXgSCgoICAESBAoCCAFiDwoBeRIKCggIARIECgIIAUIECgAQDQ==')
    session = ort.InferenceSession(model, sess_options=options, providers=['OpenVINOExecutionProvider'], provider_options=[{'device_type': device}])
    session.disable_fallback()
    output = session.run(None, {'x': np.array([1.0], dtype=np.float32)})
    if float(output[0][0]) != 2.0 or session.get_providers()[0] != 'OpenVINOExecutionProvider':
        raise RuntimeError('The OpenVINO runtime probe failed')
    return {'passed': True, 'device': device, 'provider': 'OpenVINOExecutionProvider', 'runtime_version': status['version']}
