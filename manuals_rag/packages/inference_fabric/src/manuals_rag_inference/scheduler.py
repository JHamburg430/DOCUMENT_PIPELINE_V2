"""Deterministic, application-aware admission and placement policy."""

from __future__ import annotations

import time
import math
import threading
from dataclasses import dataclass

from .models import AdmissionDecision, IdleEvictionCandidate, InferenceRequest, NodeSnapshot


@dataclass(frozen=True)
class SchedulerPolicy:
    heartbeat_ttl_seconds: float = 15.0
    reserved_vram_mb: int = 512
    max_queue_depth: int = 8
    interactive_applications: tuple[str, ...] = ("openclaw", "provis_employee_agent", "rag_console")
    require_application_allowlist: bool = False
    idle_model_ttl_seconds: float = 900.0


class InferenceScheduler:
    """In-memory scheduler; persistence/transport are deliberately separate."""

    def __init__(self, *, policy: SchedulerPolicy | None = None, clock=time.time) -> None:
        self.policy = policy or SchedulerPolicy()
        self._clock = clock
        self._lock = threading.RLock()
        self._nodes: dict[str, NodeSnapshot] = {}
        self._reservations: dict[str, set[str]] = {}
        self._reservation_models: dict[str, dict[str, str]] = {}

    def upsert_node(self, snapshot: NodeSnapshot) -> None:
        with self._lock:
            self._nodes[snapshot.node_id] = snapshot

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def snapshots(self) -> tuple[NodeSnapshot, ...]:
        with self._lock:
            return tuple(self._nodes.values())

    @staticmethod
    def _required_vram_mb(request: InferenceRequest) -> int:
        context_mb = max(0.0, request.context_tokens / 1000.0 * request.context_vram_mb_per_1k)
        return request.min_vram_mb + math.ceil(context_mb)

    def _eligible(self, request: InferenceRequest, node: NodeSnapshot) -> tuple[bool, str, bool, int]:
        if request.allowed_node_ids and node.node_id not in request.allowed_node_ids:
            return False, "application_worker_not_approved", False, 0
        if request.model_size_bytes <= 0 or request.min_vram_mb <= 0:
            return False, "invalid_model_resource_requirements", False, 0
        if not node.healthy:
            return False, "node_unhealthy", False, 0
        if not node.inference_enabled:
            return False, "inference_disabled", False, 0
        if self._clock() - node.observed_at_epoch > self.policy.heartbeat_ttl_seconds:
            return False, "node_heartbeat_stale", False, 0
        if node.allowed_applications and request.application not in node.allowed_applications:
            return False, "application_not_allowed_on_node", False, 0
        if self.policy.require_application_allowlist and not node.allowed_applications:
            return False, "application_allowlist_missing", False, 0
        missing = set(request.required_capabilities) - set(node.capabilities)
        if missing:
            return False, f"missing_capabilities:{','.join(sorted(missing))}", False, 0
        model_capabilities = node.installed_model_capabilities.get(request.model)
        if model_capabilities and set(request.required_capabilities) - set(model_capabilities):
            return False, "missing_model_capabilities", False, 0
        reserved = len(self._reservations.get(node.node_id, set()))
        if node.queued_requests + reserved >= self.policy.max_queue_depth:
            return False, "queue_full", False, 0
        loaded = next((item for item in node.loaded_models if item.name == request.model), None)
        required_vram_mb = self._required_vram_mb(request)
        loaded_vram = sum(g.available_vram_mb for g in node.gpus)
        requires_pull = loaded is None and request.model not in node.installed_models
        if loaded is not None and loaded.active_requests > 0:
            loaded_vram = max(0, loaded_vram)
        if loaded_vram - self.policy.reserved_vram_mb < required_vram_mb:
            if not request.allow_eviction:
                return False, "insufficient_vram_without_eviction", requires_pull, loaded_vram
            if not node.request_accounting_available:
                return False, "eviction_request_accounting_unavailable", requires_pull, loaded_vram
            idle_candidates = tuple(
                item.name
                for item in sorted(node.loaded_models, key=lambda item: item.last_used_epoch)
                if item.active_requests == 0 and item.name != request.model
            )
            if not idle_candidates:
                return False, "insufficient_vram_without_idle_eviction_candidate", requires_pull, loaded_vram
        if not node.gpus:
            return False, "no_gpu", requires_pull, loaded_vram
        return True, "eligible", requires_pull, loaded_vram

    def _choose_unlocked(self, request: InferenceRequest) -> AdmissionDecision:
        candidates: list[tuple[float, NodeSnapshot, bool, tuple[str, ...]]] = []
        rejected: list[str] = []
        for node in self._nodes.values():
            eligible, reason, requires_pull, available_vram = self._eligible(request, node)
            if not eligible:
                rejected.append(f"{node.node_id}:{reason}")
                continue
            if requires_pull and not request.allow_model_pull:
                rejected.append(f"{node.node_id}:model_not_installed_and_pull_not_allowed")
                continue
            loaded = next((item for item in node.loaded_models if item.name == request.model), None)
            model_loaded_bonus = 1000.0 if loaded else 0.0
            model_installed_bonus = 300.0 if request.model in node.installed_models else 0.0
            app_bonus = 200.0 if request.application in self.policy.interactive_applications else 0.0
            reserved = len(self._reservations.get(node.node_id, set()))
            queue_penalty = float((node.queued_requests + reserved) * 50 + (node.active_requests + reserved) * 25)
            required_vram_mb = self._required_vram_mb(request)
            headroom_bonus = min(200.0, max(0.0, (available_vram - required_vram_mb) / 100.0))
            score = model_loaded_bonus + model_installed_bonus + app_bonus + headroom_bonus - queue_penalty + request.priority
            evictions = ()
            if request.allow_eviction and available_vram - self.policy.reserved_vram_mb < required_vram_mb:
                evictions = tuple(
                    item.name
                    for item in sorted(node.loaded_models, key=lambda item: item.last_used_epoch)
                    if item.active_requests == 0 and item.name != request.model
                )
            candidates.append((score, node, requires_pull, evictions))
        if not candidates:
            return AdmissionDecision(False, request.request_id, reason="; ".join(rejected) or "no_eligible_nodes")
        score, node, requires_pull, evictions = max(candidates, key=lambda item: item[0])
        return AdmissionDecision(True, request.request_id, node.node_id, node.endpoint, "eligible", score, requires_pull, evictions, requires_model_warm=not any(model.name == request.model for model in node.loaded_models))

    def choose(self, request: InferenceRequest) -> AdmissionDecision:
        with self._lock:
            return self._choose_unlocked(request)

    def choose_and_reserve(self, request: InferenceRequest) -> AdmissionDecision:
        with self._lock:
            if any(request.request_id in request_ids for request_ids in self._reservations.values()):
                return AdmissionDecision(False, request.request_id, reason="duplicate_request_id")
            decision = self._choose_unlocked(request)
            if decision.admitted and decision.node_id:
                self._reservations.setdefault(decision.node_id, set()).add(request.request_id)
                self._reservation_models.setdefault(decision.node_id, {})[request.request_id] = request.model
            return decision

    def release(self, node_id: str | None, request_id: str) -> bool:
        if not node_id:
            return False
        with self._lock:
            request_ids = self._reservations.get(node_id)
            if not request_ids or request_id not in request_ids:
                return False
            request_ids.remove(request_id)
            models = self._reservation_models.get(node_id)
            if models is not None:
                models.pop(request_id, None)
                if not models:
                    self._reservation_models.pop(node_id, None)
            if not request_ids:
                self._reservations.pop(node_id, None)
            return True

    def plan_idle_evictions(
        self,
        *,
        max_items: int = 10,
        now: float | None = None,
    ) -> tuple[IdleEvictionCandidate, ...]:
        """Return safe, explicitly reviewable idle-model eviction candidates."""
        now = self._clock() if now is None else now
        limit = max(1, min(int(max_items), 100))
        candidates: list[IdleEvictionCandidate] = []
        with self._lock:
            for node in self._nodes.values():
                if not node.healthy or self._clock() - node.observed_at_epoch > self.policy.heartbeat_ttl_seconds:
                    continue
                if not node.request_accounting_available:
                    continue
                reserved_models = set((self._reservation_models.get(node.node_id) or {}).values())
                for loaded in node.loaded_models:
                    if loaded.active_requests > 0 or loaded.name in reserved_models or loaded.last_used_epoch <= 0:
                        continue
                    idle_seconds = max(0.0, now - loaded.last_used_epoch)
                    if idle_seconds < self.policy.idle_model_ttl_seconds:
                        continue
                    candidates.append(IdleEvictionCandidate(node.node_id, loaded.name, idle_seconds))
        candidates.sort(key=lambda item: item.idle_seconds, reverse=True)
        return tuple(candidates[:limit])
