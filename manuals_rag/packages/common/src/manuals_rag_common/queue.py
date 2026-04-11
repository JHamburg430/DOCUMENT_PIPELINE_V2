from __future__ import annotations

import json
from typing import Any

from redis import Redis

from manuals_rag_common.config import settings


def redis_client() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)


def enqueue(queue_name: str, payload: dict[str, Any]) -> None:
    redis_client().rpush(queue_name, json.dumps(payload, default=str))


def dequeue(queue_name: str, timeout: int = 5) -> dict[str, Any] | None:
    item = redis_client().blpop(queue_name, timeout=timeout)
    if not item:
        return None
    _, raw = item
    return json.loads(raw)
