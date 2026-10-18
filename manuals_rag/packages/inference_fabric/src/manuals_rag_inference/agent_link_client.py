"""HTTPS owner-client transport. Session capabilities never enter activity logs."""
import json
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from urllib.error import HTTPError


def call(gateway, token, action, payload=None):
    parsed = urlsplit(gateway)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Managed HTTPS gateway required')
    if action not in {'availability', 'session', 'register', 'workspace', 'history', 'post', 'executing', 'publish', 'targets', 'send', 'poll', 'complete', 'messages', 'logout', 'update-nodes', 'update-releases', 'request-update', 'update-status'}:
        raise ValueError('Unsupported owner action')
    request = Request(gateway.rstrip('/') + '/v1/agent-link/' + action,
                      data=None if action == 'availability' else json.dumps(payload or {}).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read(1_000_001)
    except HTTPError as error:
        if action == 'availability' and error.code == 404: return {'enabled': False}
        raise
    if len(raw) > 1_000_000: raise ValueError('Response too large')
    result = json.loads(raw)
    if type(result) is not dict: raise ValueError('Response object required')
    return result
