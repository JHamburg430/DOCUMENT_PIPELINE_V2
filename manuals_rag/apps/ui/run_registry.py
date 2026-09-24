"""Thread-safe runtime registry for transport visibility, not run authority."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Mapping


TERMINAL_RUN_STATUSES = frozenset({"completed", "succeeded", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: str
    workflow: str
    status: str
    last_sequence: int
    subscribers: int
    source: str
    metadata: dict[str, Any]

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunRegistry:
    """Caches observable run state while leaving persisted artifacts authoritative."""

    def __init__(self) -> None:
        self._runs: dict[str, RunSnapshot] = {}
        self._lock = RLock()

    def observe(
        self,
        run_id: str,
        *,
        workflow: str = "unknown",
        status: str = "unknown",
        last_sequence: int = 0,
        source: str = "app_run_events",
        metadata: Mapping[str, Any] | None = None,
    ) -> RunSnapshot:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if isinstance(last_sequence, bool) or not isinstance(last_sequence, int) or last_sequence < 0:
            raise ValueError("last_sequence must be a non-negative integer")
        with self._lock:
            previous = self._runs.get(run_id)
            snapshot = RunSnapshot(
                run_id=run_id,
                workflow=workflow or (previous.workflow if previous else "unknown"),
                status=status or (previous.status if previous else "unknown"),
                last_sequence=max(last_sequence, previous.last_sequence if previous else 0),
                subscribers=previous.subscribers if previous else 0,
                source=source,
                metadata=deepcopy(dict(metadata or (previous.metadata if previous else {}))),
            )
            self._runs[run_id] = snapshot
            return snapshot

    def subscriber_opened(self, run_id: str) -> RunSnapshot:
        with self._lock:
            current = self._runs.get(run_id) or self.observe(run_id)
            updated = RunSnapshot(**{**current.to_dict(), "subscribers": current.subscribers + 1})
            self._runs[run_id] = updated
            return updated

    def subscriber_closed(self, run_id: str) -> RunSnapshot | None:
        with self._lock:
            current = self._runs.get(run_id)
            if current is None:
                return None
            updated = RunSnapshot(**{**current.to_dict(), "subscribers": max(0, current.subscribers - 1)})
            self._runs[run_id] = updated
            return updated

    def get(self, run_id: str) -> RunSnapshot | None:
        with self._lock:
            return self._runs.get(run_id)
