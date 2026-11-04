"""Public, non-secret contract for the owner-scoped PROVIS Node Channel."""

PROTOCOL_ID = "provis-node-channel/1"
PROTOCOL_HEADER = "X-ProVIS-Node-Channel-Protocol"
CLIENT_RELEASE_HEADER = "X-ProVIS-Node-Release"
MINIMUM_NODE_RELEASE = "2026.11.4.1"
LEASE_SECONDS = 30
HEARTBEAT_SECONDS = 6
CLAIM_TIMEOUT_SECONDS = 600


class ProtocolVersionError(ValueError):
    """The caller and gateway do not implement the same Node Channel contract."""


def require_protocol(value):
    if value != PROTOCOL_ID:
        supplied = str(value or "missing")[:80]
        raise ProtocolVersionError(
            "PROVIS_NODE_CHANNEL_VERSION_MISMATCH: gateway requires "
            f"{PROTOCOL_ID}; client supplied {supplied}. Update the PROVIS "
            f"Inference Node to {MINIMUM_NODE_RELEASE} or newer."
        )


def availability(enabled, *, component="gateway", release="unknown"):
    return {
        "enabled": bool(enabled),
        "node_channel_protocol": PROTOCOL_ID,
        "supported_protocols": [PROTOCOL_ID],
        "minimum_node_release": MINIMUM_NODE_RELEASE,
        "component": str(component)[:40],
        "component_release": str(release)[:40],
        "lease_seconds": LEASE_SECONDS,
        "heartbeat_seconds": HEARTBEAT_SECONDS,
    }


def validate_availability(document):
    if not isinstance(document, dict) or document.get("node_channel_protocol") != PROTOCOL_ID:
        actual = document.get("node_channel_protocol", "missing") if isinstance(document, dict) else "invalid"
        raise ProtocolVersionError(
            "PROVIS_NODE_CHANNEL_VERSION_MISMATCH: this node requires "
            f"{PROTOCOL_ID}, but the gateway reported {actual}. The gateway "
            "must be upgraded before Node Channel discovery can be used."
        )
    return document
