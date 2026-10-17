"""Serializable protocol models for the inference fabric."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GpuSnapshot:
    gpu_id: str
    name: str
    total_vram_mb: int
    used_vram_mb: int
    utilization_percent: float = 0.0
    temperature_c: float | None = None
    healthy: bool = True

    @property
    def available_vram_mb(self) -> int:
        return max(0, self.total_vram_mb - self.used_vram_mb)


@dataclass(frozen=True)
class LoadedModel:
    name: str
    size_bytes: int
    gpu_ids: tuple[str, ...] = ()
    last_used_epoch: float = 0.0
    active_requests: int = 0


@dataclass(frozen=True)
class NodeSnapshot:
    node_id: str
    endpoint: str
    tailnet_name: str
    gpus: tuple[GpuSnapshot, ...]
    loaded_models: tuple[LoadedModel, ...] = ()
    active_requests: int = 0
    queued_requests: int = 0
    allowed_applications: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ("chat", "embeddings")
    observed_at_epoch: float = 0.0
    healthy: bool = True
    request_accounting_available: bool = False
    inference_enabled: bool = True
    installed_models: tuple[str, ...] = ()
    installed_model_capabilities: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    application: str
    workload: str
    model: str
    model_size_bytes: int
    min_vram_mb: int
    context_tokens: int = 0
    context_vram_mb_per_1k: float = 0.0
    priority: int = 50
    max_latency_ms: int | None = None
    required_capabilities: tuple[str, ...] = ("chat",)
    allow_model_pull: bool = False
    allow_eviction: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Authoritative resource and capability metadata for a routable model."""

    size_bytes: int
    min_vram_mb: int
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.size_bytes <= 0 or self.min_vram_mb <= 0:
            raise ValueError("model resource requirements must be positive")
        if not self.capabilities:
            raise ValueError("model capabilities must not be empty")


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    request_id: str
    node_id: str | None = None
    endpoint: str | None = None
    reason: str = ""
    score: float | None = None
    requires_model_pull: bool = False
    eviction_candidates: tuple[str, ...] = ()
    requires_model_warm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdleEvictionCandidate:
    node_id: str
    model: str
    idle_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
