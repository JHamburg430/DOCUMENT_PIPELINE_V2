"""Content-free diagnostics for the runtime shipped in signed node apps."""
from __future__ import annotations

import importlib
import importlib.util
import base64
import re
import sys
import ctypes
import os
from pathlib import Path
import json
import subprocess
import threading
import time

INTEL_DRIVER_PAGE = 'https://www.intel.com/content/www/us/en/support/detect.html'
_inventory_cache = None
_inventory_checked = 0.0
_inventory_lock = threading.Lock()
_native_libraries = {}
_dll_directories = {}


def _preload_linux_library(package: str, directory: str):
    # ORT and the Python device-discovery SDK ship different Linux C++ ABIs
    # under the same SONAME. Select each one's own library in separate processes.
    if sys.platform != 'linux':
        return
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return
    if spec and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations))) / directory
        libraries = sorted(root.glob('libopenvino.so.*'))
        if libraries:
            key = str(libraries[0])
            if key not in _native_libraries:
                _native_libraries[key] = ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)


def load_onnx_runtime():
    if os.name == 'nt':
        for package, directory in [('openvino', 'libs'), ('onnxruntime', 'capi')]:
            try:
                spec = importlib.util.find_spec(package)
            except (ImportError, ValueError):
                continue
            if spec and spec.submodule_search_locations:
                path = str(Path(next(iter(spec.submodule_search_locations))) / directory)
                if Path(path).is_dir() and path not in _dll_directories:
                    _dll_directories[path] = os.add_dll_directory(path)
                library = str(Path(path)/'openvino.dll')
                if package == 'openvino' and Path(library).is_file() and library not in _native_libraries:
                    _native_libraries[library] = ctypes.WinDLL(library, winmode=0x1100)
    _preload_linux_library('onnxruntime', 'capi')
    return importlib.import_module('onnxruntime')


def core_device_probe() -> dict:
    """Only call in the dedicated discovery process, never the ORT process."""
    _preload_linux_library('openvino', 'libs')
    core = importlib.import_module('openvino').Core()
    return {'devices': detected_openvino_devices(core)}


def clear_device_cache():
    global _inventory_cache
    with _inventory_lock:
        _inventory_cache = None


def get_device_inventory() -> dict:
    global _inventory_cache, _inventory_checked
    with _inventory_lock:
        if _inventory_cache is not None and time.monotonic()-_inventory_checked < 300:
            return _inventory_cache
        if getattr(sys, 'frozen', False):
            command = [sys.executable, '--accelerator-probe']
        else:
            python = Path(sys.executable)
            if python.name.lower() == 'pythonw.exe':
                python = python.with_name('python.exe')
            command = [str(python), '-c', 'import sys,json;sys.path.insert(0,sys.argv[1]);from manuals_rag_inference.onnx_runtime import core_device_probe;print(json.dumps(core_device_probe()))', str(Path(__file__).resolve().parent.parent)]
        completed = subprocess.run(command, capture_output=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if completed.returncode != 0 or len(completed.stdout) > 32768:
            raise RuntimeError('Accelerator discovery failed')
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or not isinstance(result.get('devices'), list):
            raise ValueError('Invalid accelerator discovery response')
        _inventory_cache = result
        _inventory_checked = time.monotonic()
        return result


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
        ort = load_onnx_runtime()
        importlib.import_module('numpy')
        result.update(installed=True, version=str(ort.__version__), providers=list(ort.get_available_providers()))
    except (ImportError, OSError, AttributeError) as error:
        result['error_category'] = type(error).__name__
        return result
    try:
        result['devices'] = get_device_inventory()['devices']
        result['intel_gpus'] = [item for item in result['devices'] if item['id'].startswith('GPU')]
    except Exception as error:
        result['device_error_category'] = type(error).__name__
    result['status'] = ('device_unavailable' if 'device_error_category' in result or not result['devices'] else
                        'intel_gpu_ready' if result['intel_gpus'] else 'cpu_ready') if 'OpenVINOExecutionProvider' in result['providers'] else 'provider_missing'
    return result


def device_choices(status: dict) -> tuple[str, ...]:
    return ('AUTO',) + tuple(item['id'] for item in status.get('devices', []) if item.get('id'))


def require_bundled_runtime() -> dict:
    """Fail a release build/self-test if a required runtime is missing."""
    import cryptography.hazmat.primitives.asymmetric.ed25519  # noqa: F401
    status = runtime_status()
    if not status['installed'] or 'OpenVINOExecutionProvider' not in status['providers']:
        raise RuntimeError('The node build must include ONNX Runtime, OpenVINO and NumPy')
    if 'device_error_category' in status or not status['devices']:
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
