"""Application-aware local Ollama inference scheduling primitives."""

from .models import (
    AdmissionDecision,
    GpuSnapshot,
    IdleEvictionCandidate,
    InferenceRequest,
    LoadedModel,
    ModelSpec,
    NodeSnapshot,
)
from .scheduler import InferenceScheduler
from .enrollment import EnrollmentManager
from .provisioning import ProvisioningManifest, activate_staged, stage_artifact

__all__ = [
    "AdmissionDecision",
    "GpuSnapshot",
    "IdleEvictionCandidate",
    "InferenceRequest",
    "InferenceScheduler",
    "EnrollmentManager",
    "LoadedModel",
    "ModelSpec",
    "NodeSnapshot",
]
