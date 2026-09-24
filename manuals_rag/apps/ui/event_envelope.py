"""Validated event contract shared by snapshot, replay, and SSE transports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID, uuid4


REQUIRED_FIELDS = frozenset(
    {
        "event_id", "sequence", "timestamp", "run_id", "parent_run_id",
        "workflow", "phase", "entity_type", "entity_id", "status",
        "completed", "total", "provisional", "payload", "artifact_ref",
    }
)
TERMINAL_STATUSES = frozenset({"completed", "succeeded", "failed", "cancelled"})


class EventValidationError(ValueError):
    """Raised when untrusted event data does not satisfy the envelope contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _required_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _validated_timestamp(value: Any) -> str:
    text = _required_text("timestamp", value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise EventValidationError("timestamp must be RFC3339/ISO-8601") from error
    if parsed.tzinfo is None:
        raise EventValidationError("timestamp must include a timezone")
    return text


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_id: str
    sequence: int
    timestamp: str
    run_id: str
    parent_run_id: str | None
    workflow: str
    phase: str
    entity_type: str
    entity_id: str
    status: str
    completed: bool
    total: int | None
    provisional: bool
    payload: dict[str, Any] = field(default_factory=dict)
    artifact_ref: str | None = None

    def __post_init__(self) -> None:
        try:
            UUID(str(self.event_id))
        except (ValueError, TypeError, AttributeError) as error:
            raise EventValidationError("event_id must be a UUID") from error
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise EventValidationError("sequence must be a positive integer")
        object.__setattr__(self, "timestamp", _validated_timestamp(self.timestamp))
        for name in ("run_id", "workflow", "phase", "entity_type", "entity_id", "status"):
            object.__setattr__(self, name, _required_text(name, getattr(self, name)))
        if self.parent_run_id is not None:
            object.__setattr__(self, "parent_run_id", _required_text("parent_run_id", self.parent_run_id))
        if not isinstance(self.completed, bool):
            raise EventValidationError("completed must be a boolean")
        if not isinstance(self.provisional, bool):
            raise EventValidationError("provisional must be a boolean")
        if self.total is not None and (
            isinstance(self.total, bool) or not isinstance(self.total, int) or self.total < 0
        ):
            raise EventValidationError("total must be null or a non-negative integer")
        if not isinstance(self.payload, dict):
            raise EventValidationError("payload must be an object")
        object.__setattr__(self, "payload", deepcopy(self.payload))
        if self.artifact_ref is not None:
            object.__setattr__(self, "artifact_ref", _required_text("artifact_ref", self.artifact_ref))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_sequence(self, sequence: int) -> "EventEnvelope":
        return replace(self, sequence=sequence)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EventEnvelope":
        if not isinstance(value, Mapping):
            raise EventValidationError("event must be an object")
        missing = sorted(REQUIRED_FIELDS.difference(value))
        unknown = sorted(set(value).difference(REQUIRED_FIELDS))
        if missing:
            raise EventValidationError(f"event is missing required fields: {', '.join(missing)}")
        if unknown:
            raise EventValidationError(f"event has unknown fields: {', '.join(unknown)}")
        return cls(**dict(value))


def make_event(
    *, sequence: int, run_id: str, workflow: str, phase: str,
    entity_type: str, entity_id: str, status: str,
    completed: bool = False, total: int | None = None,
    provisional: bool = False, payload: dict[str, Any] | None = None,
    artifact_ref: str | None = None, parent_run_id: str | None = None,
    event_id: str | None = None, timestamp: str | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id or str(uuid4()), sequence=sequence,
        timestamp=timestamp or utc_now(), run_id=run_id,
        parent_run_id=parent_run_id, workflow=workflow, phase=phase,
        entity_type=entity_type, entity_id=entity_id, status=status,
        completed=completed, total=total, provisional=provisional,
        payload=payload or {}, artifact_ref=artifact_ref,
    )
