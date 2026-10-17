"""Dependency-free helpers for displaying the node's local inference URL."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def local_broker_endpoints(port: object) -> tuple[str, str]:
    """Return local OpenAI-compatible URLs for host and Docker clients."""
    try:
        broker_port = int(str(port).strip())
    except (TypeError, ValueError):
        broker_port = 8767
    if not 1 <= broker_port <= 65535:
        broker_port = 8767
    return (
        f"http://127.0.0.1:{broker_port}/v1",
        f"http://host.docker.internal:{broker_port}/v1",
    )


def copy_endpoint_to_clipboard(
    endpoint: str,
    allowed_endpoints: Iterable[str],
    *,
    clipboard_clear: Callable[[], object],
    clipboard_append: Callable[[str], object],
    update: Callable[[], object],
) -> bool:
    """Copy only one of the currently displayed non-secret endpoint URLs."""
    if endpoint not in set(allowed_endpoints):
        return False
    clipboard_clear()
    clipboard_append(endpoint)
    update()
    return True
