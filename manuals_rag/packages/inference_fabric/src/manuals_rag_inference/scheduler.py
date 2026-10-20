"""Deterministic, application-aware admission and placement policy."""

from __future__ import annotations

import time
import math
import threading
from dataclasses import dataclass

from .models import AdmissionDecision, GpuPlacement, IdleEvictionCandidate, InferenceRequest, NodeSnapshot


@dataclass(frozen=True)
class SchedulerPolicy:
    heartbeat_ttl_seconds: float = 15.0
    reserved_vram_mb: int = 512
    max_queue_depth: int = 8
    interactive_applications: tuple[str, ...] = ("openclaw", "provis_employee_agent", "rag_console")
    require_application_allowlist: bool = False
    idle_model_ttl_seconds: float = 900.0
    # Multi-GPU placement is the default. A request can still opt into
    # single-GPU placement when the backend or model requires it.
    gpu_placement_mode: str = "multi_gpu"


class InferenceScheduler:
    """In-memory scheduler; persistence/transport are deliberately separate."""

    def __init__(self, *, policy: SchedulerPolicy | None = None, clock=time.time) -> None:
        self.policy = policy or SchedulerPolicy()
        self._clock = clock
        self._lock = threading.RLock()
        self._nodes: dict[str, NodeSnapshot] = {}
        self._reservations: dict[str, set[str]] = {}
        self._reservation_models: dict[str, dict[str, str]] = {}
        self._reservation_placements: dict[str, dict[str, dict[str, int]]] = {}
        self._reservation_shared_bytes: dict[str, dict[str, int]] = {}

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

    def _reserved_vram_by_gpu(self, node_id: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for placement in (self._reservation_placements.get(node_id) or {}).values():
            for gpu_id, amount in placement.items():
                result[gpu_id] = result.get(gpu_id, 0) + max(0, int(amount))
        return result

    def _placement_for(self, request: InferenceRequest, node: NodeSnapshot) -> tuple[GpuPlacement | None, str, int]:
        if request.backend == "onnx" and not node.onnx_providers:
            return None, "onnx_runtime_unavailable", 0
        backends = {'cuda','rocm','metal','vulkan'} if request.backend == 'ollama' else set()
        if request.backend == 'onnx':
            if 'OpenVINOExecutionProvider' in node.onnx_providers: backends.add('openvino')
            if 'CUDAExecutionProvider' in node.onnx_providers: backends.add('cuda')
            if 'ROCMExecutionProvider' in node.onnx_providers: backends.add('rocm')
        gpus = [gpu for gpu in node.gpus if gpu.healthy and gpu.backend in backends]
        if not gpus:
            return None, "no_gpu", 0

        reserved = self._reserved_vram_by_gpu(node.node_id)
        devices = []
        loaded = next((m for m in node.loaded_models if m.name == request.model and m.placement_known), None)
        for gpu in gpus:
            # Resident model memory is reusable, not a second model allocation.
            # Credit only explicitly measured per-device placement, capped by
            # observed usage. Unknown placement stays conservatively uncredited.
            credit = loaded.vram_by_gpu_mb.get(gpu.gpu_id,0) if loaded else 0
            credit = min(credit,gpu.used_vram_mb,gpu.total_vram_mb) if type(credit) is int and credit>0 else 0
            free = gpu.available_vram_mb + credit - reserved.get(gpu.gpu_id, 0)
            if not gpu.memory_unknown:
                free -= self.policy.reserved_vram_mb
            devices.append((gpu, max(0, free)))

        required = self._required_vram_mb(request)
        mode = request.placement_mode or self.policy.gpu_placement_mode
        if node.gpu_placement_mode == "single_gpu":
            mode = "single_gpu"
        if mode not in {"single_gpu", "multi_gpu"}:
            return None, "invalid_gpu_placement_mode", 0
        if not request.allow_multi_gpu:
            mode = "single_gpu"
        if mode == 'single_gpu' and request.min_gpu_count > 1:
            return None, 'minimum_gpu_count_unavailable', 0

        # OpenVINO manages shared Intel memory and can select GPU.0/GPU.1 or
        # AUTO/MULTI itself. Preserve the device inventory for observability,
        # but do not invent VRAM numbers when the driver cannot report them.
        unknown = [(gpu, free) for gpu, free in devices if gpu.memory_unknown]
        if request.backend == "onnx" and unknown:
            resources=node.host_resources or {}
            available=resources.get('memory_available_bytes')
            stamp=resources.get('observed_at_epoch')
            fresh=(type(stamp) in (int,float) and math.isfinite(stamp) and
                   0 <= self._clock()-stamp <= self.policy.heartbeat_ttl_seconds)
            needed=max(request.model_size_bytes,required*1024*1024)+1024**3
            reserved_shared=sum(self._reservation_shared_bytes.get(node.node_id,{}).values())
            if not fresh or type(available) is not int or available-reserved_shared < needed:
                return None, 'shared_memory_headroom_unavailable', 0
            if len(unknown) < request.min_gpu_count:
                return None, 'minimum_gpu_count_unavailable', 0
            if mode == "single_gpu":
                gpu = unknown[0][0]
                return GpuPlacement("single_gpu", (gpu.gpu_id,), {}, "onnx"), "eligible", 0
            gpu_ids = tuple(gpu.gpu_id for gpu, _ in unknown)
            return GpuPlacement("multi_gpu", gpu_ids, {}, "onnx"), "eligible", 0

        devices.sort(key=lambda item: item[1], reverse=True)
        if mode == "single_gpu":
            for gpu, free in devices:
                if free >= required:
                    return GpuPlacement("single_gpu", (gpu.gpu_id,), {gpu.gpu_id: required}, request.backend), "eligible", free
            return None, "insufficient_vram_on_single_gpu", sum(free for _, free in devices)

        selected = []
        remaining = required
        for gpu, free in devices:
            if free <= 0:
                continue
            selected.append((gpu, free))
            remaining -= free
            if remaining <= 0 and len(selected) >= max(1, request.min_gpu_count):
                break
        if remaining > 0 or len(selected) < max(1, request.min_gpu_count):
            return None, "insufficient_vram_for_multi_gpu_placement", sum(free for _, free in devices)
        allocation: dict[str, int] = {}
        remaining = required
        for gpu, free in selected:
            amount = min(free, remaining)
            allocation[gpu.gpu_id] = amount
            remaining -= amount
        placement = GpuPlacement("multi_gpu", tuple(gpu.gpu_id for gpu, _ in selected), allocation, request.backend)
        return placement, "eligible", sum(free for _, free in devices)

    def _eligible(self, request: InferenceRequest, node: NodeSnapshot) -> tuple[bool, str, bool, int, GpuPlacement | None]:
        if request.allowed_node_ids and node.node_id not in request.allowed_node_ids:
            return False, "application_worker_not_approved", False, 0, None
        if request.model_size_bytes <= 0 or request.min_vram_mb <= 0:
            return False, "invalid_model_resource_requirements", False, 0, None
        if not node.healthy:
            return False, "node_unhealthy", False, 0, None
        if not node.inference_enabled:
            return False, "inference_disabled", False, 0, None
        age=self._clock()-node.observed_at_epoch
        if not math.isfinite(age) or not 0 <= age <= self.policy.heartbeat_ttl_seconds:
            return False, "node_heartbeat_stale", False, 0, None
        if node.allowed_applications and request.application not in node.allowed_applications:
            return False, "application_not_allowed_on_node", False, 0, None
        if self.policy.require_application_allowlist and not node.allowed_applications:
            return False, "application_allowlist_missing", False, 0, None
        missing = set(request.required_capabilities) - set(node.capabilities)
        if missing:
            return False, f"missing_capabilities:{','.join(sorted(missing))}", False, 0, None
        model_capabilities = node.installed_model_capabilities.get(request.model)
        if model_capabilities and set(request.required_capabilities) - set(model_capabilities):
            return False, "missing_model_capabilities", False, 0, None
        reserved = len(self._reservations.get(node.node_id, set()))
        if node.active_requests + node.queued_requests + reserved >= self.policy.max_queue_depth:
            return False, "queue_full", False, 0, None
        loaded = next((item for item in node.loaded_models if item.name == request.model), None)
        requires_pull = loaded is None and request.model not in node.installed_models
        placement, placement_reason, loaded_vram = self._placement_for(request, node)
        if placement is None:
            if not request.allow_eviction:
                return False, placement_reason, requires_pull, loaded_vram, None
            if not node.request_accounting_available:
                return False, "eviction_request_accounting_unavailable", requires_pull, loaded_vram, None
            idle_candidates = tuple(
                item.name
                for item in sorted(node.loaded_models, key=lambda item: item.last_used_epoch)
                if item.active_requests == 0 and item.name != request.model
            )
            if not idle_candidates:
                return False, "insufficient_vram_without_idle_eviction_candidate", requires_pull, loaded_vram, None
            # Eviction changes capacity only after the command executes. Do
            # not claim a placement before that point.
            return False, placement_reason, requires_pull, loaded_vram, None
        return True, "eligible", requires_pull, loaded_vram, placement

    def _choose_unlocked(self, request: InferenceRequest) -> AdmissionDecision:
        candidates: list[tuple[float, NodeSnapshot, bool, tuple[str, ...], GpuPlacement]] = []
        rejected: list[str] = []
        for node in self._nodes.values():
            eligible, reason, requires_pull, available_vram, placement = self._eligible(request, node)
            if not eligible:
                rejected.append(f"{node.node_id}:{reason}")
                continue
            if placement is None:
                rejected.append(f"{node.node_id}:placement_unavailable")
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
            selected=set(placement.gpu_ids)
            loads=[gpu.utilization_percent for gpu in node.gpus if gpu.gpu_id in selected
                   and type(gpu.utilization_percent) in (int,float) and math.isfinite(gpu.utilization_percent)
                   and 0 <= gpu.utilization_percent <= 100]
            if loads: queue_penalty += sum(loads)/len(loads)*.5
            resources=node.host_resources or {}
            cpu=resources.get('cpu_percent');stamp=resources.get('observed_at_epoch')
            if (type(cpu) in (int,float) and math.isfinite(cpu) and 0 <= cpu <= 100 and
                    type(stamp) in (int,float) and math.isfinite(stamp) and
                    0 <= self._clock()-stamp <= self.policy.heartbeat_ttl_seconds):
                queue_penalty += cpu*.25
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
            candidates.append((score, node, requires_pull, evictions, placement))
        if not candidates:
            return AdmissionDecision(False, request.request_id, reason="; ".join(rejected) or "no_eligible_nodes")
        score, node, requires_pull, evictions, placement = max(candidates, key=lambda item: item[0])
        return AdmissionDecision(True, request.request_id, node.node_id, node.endpoint, "eligible", score, requires_pull, evictions, requires_model_warm=not any(model.name == request.model for model in node.loaded_models), placement=placement)

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
                if decision.placement:
                    self._reservation_placements.setdefault(decision.node_id, {})[request.request_id] = dict(decision.placement.vram_by_gpu_mb)
                    if request.backend=='onnx' and not decision.placement.vram_by_gpu_mb:
                        self._reservation_shared_bytes.setdefault(decision.node_id,{})[request.request_id]=max(request.model_size_bytes,self._required_vram_mb(request)*1024*1024)
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
            placements = self._reservation_placements.get(node_id)
            if placements is not None:
                placements.pop(request_id, None)
                if not placements:
                    self._reservation_placements.pop(node_id, None)
            shared=self._reservation_shared_bytes.get(node_id)
            if shared is not None:
                shared.pop(request_id,None)
                if not shared:self._reservation_shared_bytes.pop(node_id,None)
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
