from __future__ import annotations

import hashlib
import uuid


NAMESPACE = uuid.UUID("6f067160-c979-4ef5-80df-570f684d20b8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_uuid(*parts: object) -> str:
    raw = "::".join(str(part) for part in parts)
    return str(uuid.uuid5(NAMESPACE, raw))
