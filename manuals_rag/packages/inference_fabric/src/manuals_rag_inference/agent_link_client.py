"""HTTPS owner-client transport. Session capabilities never enter activity logs."""
import json
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from urllib.error import HTTPError

from .node_channel_protocol import (
    CLIENT_RELEASE_HEADER,
    PROTOCOL_HEADER,
    PROTOCOL_ID,
    ProtocolVersionError,
    validate_availability,
)
from .node_release import DEFAULT_VERSION


def call(gateway, token, action, payload=None):
    parsed = urlsplit(gateway)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Managed HTTPS gateway required')
    if action not in {'availability', 'session', 'register', 'heartbeat', 'unregister', 'workspace', 'history', 'audit', 'post', 'executing', 'publish', 'targets', 'send', 'poll', 'complete', 'messages', 'reply-poll', 'reply-decide', 'logout', 'update-nodes', 'update-releases', 'request-update', 'update-status', 'diagnostic-inference'}:
        raise ValueError('Unsupported owner action')
    request = Request(
        gateway.rstrip('/') + '/v1/agent-link/' + action,
        data=None if action == 'availability' else json.dumps(payload or {}).encode(),
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token,
            PROTOCOL_HEADER: PROTOCOL_ID,
            CLIENT_RELEASE_HEADER: DEFAULT_VERSION,
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read(1_000_001)
    except HTTPError as error:
        if action == 'availability' and error.code == 404:
            raise ProtocolVersionError(
                'PROVIS_NODE_CHANNEL_VERSION_MISMATCH: the gateway does not expose '
                f'{PROTOCOL_ID}. Upgrade the gateway before using Node Channel.'
            ) from error
        if error.code == 409:
            try:
                detail = json.loads(error.read(65537)).get('detail')
            except Exception:
                detail = None
            raise ProtocolVersionError(
                detail or f'PROVIS_NODE_CHANNEL_VERSION_MISMATCH: {PROTOCOL_ID} required.'
            ) from error
        raise
    if len(raw) > 1_000_000: raise ValueError('Response too large')
    result = json.loads(raw)
    if type(result) is not dict: raise ValueError('Response object required')
    if action == 'availability':
        validate_availability(result)
    elif action == 'session':
        actual = result.get('node_channel_protocol')
        if actual != PROTOCOL_ID:
            raise ProtocolVersionError(
                'PROVIS_NODE_CHANNEL_VERSION_MISMATCH: owner verification reached '
                f'a stale gateway ({actual or "protocol missing"}); {PROTOCOL_ID} is required.'
            )
    return result

