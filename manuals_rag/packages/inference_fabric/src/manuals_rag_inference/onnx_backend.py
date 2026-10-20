"""ONNX Runtime backend with OpenVINO/Intel device support.

Signed desktop apps include the runtime automatically. Administrators register
approved models through ``INFERENCE_ONNX_MODELS_JSON``. The backend accepts tensor inputs rather than
pretending that arbitrary ONNX graphs are OpenAI-compatible chat models.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class OnnxBackendError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


class OnnxBackend:
    """Lazy, allowlisted ONNX Runtime sessions.

    ``INFERENCE_ONNX_MODELS_JSON`` is a JSON object such as::

        {"intel-encoder": {"path": "/srv/models/encoder.onnx",
                           "capabilities": ["onnx", "embeddings"]}}

    The manifest is deliberately explicit so a signed node command cannot
    turn into arbitrary local file execution.
    """

    def __init__(self, manifest: dict[str, dict[str, Any]] | None = None, *, device: str = 'AUTO') -> None:
        self.manifest = manifest or self._load_manifest()
        self._sessions: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._ort: Any | None = None
        self._providers: tuple[str, ...] = ()
        self._device = device
        self._devices: tuple[str, ...] = ()
        self._load_runtime()

    @classmethod
    def from_env(cls) -> "OnnxBackend":
        return cls()

    @staticmethod
    def _load_manifest() -> dict[str, dict[str, Any]]:
        raw = os.getenv("INFERENCE_ONNX_MODELS_JSON", "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise OnnxBackendError("INFERENCE_ONNX_MODELS_JSON is invalid JSON") from error
        if not isinstance(payload, dict):
            raise OnnxBackendError("INFERENCE_ONNX_MODELS_JSON must be an object")
        result: dict[str, dict[str, Any]] = {}
        for name, item in payload.items():
            if not isinstance(name, str) or not isinstance(item, dict):
                raise OnnxBackendError("ONNX model manifest entries must be objects")
            path = Path(str(item.get("path") or "")).expanduser()
            if not path.is_absolute() or not path.name:
                raise OnnxBackendError(f"ONNX model {name!r} must use an absolute path")
            result[name] = dict(item, path=str(path))
        return result

    def _load_runtime(self) -> None:
        try:
            from .onnx_runtime import load_onnx_runtime
            ort = load_onnx_runtime()
        except (ImportError, OSError):
            return
        self._ort = ort
        self._providers = tuple(str(item) for item in ort.get_available_providers())
        try:
            from .onnx_runtime import get_device_inventory
            self._devices = tuple(item['id'] for item in get_device_inventory()['devices'])
        except Exception:
            self._devices = ()

    @property
    def enabled(self) -> bool:
        return bool(self.manifest) and self._ort is not None

    @property
    def providers(self) -> tuple[str, ...]:
        return self._providers

    @property
    def device_type(self) -> str:
        return os.getenv("INFERENCE_ONNX_DEVICE", self._device).strip().upper() or "AUTO"

    def set_device(self, device: str) -> None:
        device = str(device).upper()
        if device not in {'AUTO', 'CPU', *self._devices}:
            raise OnnxBackendError('The selected OpenVINO device is unavailable')
        with self._lock:
            self._device = device
            self._sessions.clear()

    def _provider_configuration(self) -> tuple[list[str], list[dict[str, str]]]:
        if self._ort is None:
            raise OnnxBackendError("onnxruntime is not installed")
        providers: list[str] = []
        options: list[dict[str, str]] = []
        if "OpenVINOExecutionProvider" in self._providers:
            # OpenVINO supports AUTO, GPU/GPU.N, MULTI and HETERO device
            # strings. AUTO is the default so Intel integrated GPUs are used
            # when available while retaining a CPU fallback.
            selected = self.device_type
            if selected == 'AUTO':
                targets = [item for item in self._devices if item.startswith(('GPU', 'NPU'))]
                selected = 'AUTO:'+','.join(targets+['CPU']) if targets else 'CPU'
            providers.append("OpenVINOExecutionProvider")
            options.append({"device_type": selected})
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
            if provider in self._providers and provider not in providers:
                providers.append(provider)
                options.append({})
        if not providers:
            raise OnnxBackendError("onnxruntime has no supported execution provider")
        return providers, options

    def _session(self, model: str) -> Any:
        if model not in self.manifest:
            raise OnnxBackendError("ONNX model is not registered")
        with self._lock:
            if model in self._sessions:
                return self._sessions[model]
            path = Path(str(self.manifest[model]["path"])).resolve()
            if not path.is_file():
                raise OnnxBackendError("registered ONNX model is unavailable")
            providers, options = self._provider_configuration()
            session = self._ort.InferenceSession(str(path), providers=providers, provider_options=options)
            self._sessions[model] = session
            return session

    def capabilities(self, model: str) -> tuple[str, ...]:
        item = self.manifest.get(model) or {}
        values = item.get("capabilities") or ("onnx",)
        return tuple(sorted({str(value) for value in values} | {"onnx"}))

    def model_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifest)) if self.enabled else ()

    def gpu_devices(self) -> tuple[str, ...]:
        if self._ort is None or "OpenVINOExecutionProvider" not in self._providers:
            return ()
        device = self.device_type
        if device == "CPU":
            return ()
        # Runtime/provider availability does not prove an Intel GPU exists.
        return tuple(f'openvino:{item}' for item in self._devices if item.startswith('GPU')
                     and (device == 'AUTO' or device == item or device == 'GPU'))

    def infer(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise OnnxBackendError("ONNX Runtime is not installed or no models are registered")
        inputs = payload.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            raise OnnxBackendError("ONNX inference requires a non-empty inputs object")
        try:
            import numpy as np
        except ImportError as error:
            raise OnnxBackendError("numpy is required for ONNX inference") from error
        session = self._session(model)
        feed = {str(name): np.asarray(value) for name, value in inputs.items()}
        expected = {item.name for item in session.get_inputs()}
        if set(feed) - expected:
            raise OnnxBackendError("ONNX input contains an unknown tensor name")
        outputs = session.run(None, feed)
        return {
            "model": model,
            "backend": "onnx",
            "providers": list(session.get_providers()),
            "outputs": [_jsonable(value) for value in outputs],
        }
