"""Single-use, process-local registration challenges; no node credentials persist."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass

PURPOSE = 'provis-node-registration-verification-v1'
CHALLENGE_TTL_SECONDS = 60


def registration_proof_bytes(node_id: str, challenge: str, access_token: str) -> bytes:
    return json.dumps({
        'purpose': PURPOSE,
        'node_id': node_id,
        'challenge': challenge,
        'access_token_sha256': hashlib.sha256(access_token.encode()).hexdigest(),
    }, sort_keys=True, separators=(',', ':')).encode()


@dataclass(frozen=True)
class _Pending:
    binding: tuple[str, str, str, str]
    expires: float


class RegistrationChallenges:
    """Bounded outstanding nonces, consumed atomically only after valid proof.

    A restart discards every outstanding nonce, so proofs from the previous
    process fail closed. Expired entries are never recycled or accepted.
    Neither bearer tokens nor node secrets are retained in this cache.
    """
    def __init__(self, *, clock=time.monotonic, max_pending=4096, max_per_node=4):
        self._clock = clock
        self._max_pending = max_pending
        self._max_per_node = max_per_node
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    @staticmethod
    def _binding(node_id, subject, tenant, token):
        return node_id, subject, tenant, hashlib.sha256(token.encode()).hexdigest()

    def _purge(self, now):
        self._pending = {nonce: record for nonce, record in self._pending.items() if record.expires > now}

    def issue(self, *, node_id, subject, tenant, access_token):
        with self._lock:
            now = self._clock()
            self._purge(now)
            if len(self._pending) >= self._max_pending:
                return None
            if sum(record.binding[0] == node_id for record in self._pending.values()) >= self._max_per_node:
                return None
            for _ in range(3):
                nonce = secrets.token_urlsafe(32)
                if nonce not in self._pending:
                    self._pending[nonce] = _Pending(
                        self._binding(node_id, subject, tenant, access_token), now + CHALLENGE_TTL_SECONDS)
                    return nonce
            return None

    def consume(self, *, node_id, subject, tenant, access_token, challenge, signature, secret):
        if not isinstance(challenge, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', challenge):
            return False
        if not isinstance(signature, str) or not re.fullmatch(r'[0-9a-f]{64}', signature):
            return False
        with self._lock:
            self._purge(self._clock())
            pending = self._pending.get(challenge)
            if pending is None or pending.binding != self._binding(node_id, subject, tenant, access_token):
                return False
            expected = hmac.new(secret, registration_proof_bytes(node_id, challenge, access_token), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                return False
            del self._pending[challenge]
            return True
