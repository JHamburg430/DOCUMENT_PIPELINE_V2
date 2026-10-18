from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from provis_inference_node.enrollment import EnrollmentManager
from provis_inference_node.gateway import InferenceGateway
from provis_inference_node.entra_client import reuse_or_enroll_node_with_entra


class FakeValidator:
    def __init__(self, identity=None, error=None):
        self.identity = identity or {"subject": "subject-1", "tenant_id": "tenant-1"}
        self.error = error

    def validate(self, token):
        if self.error:
            raise self.error
        if token != "valid-token":
            raise PermissionError("invalid token")
        return self.identity


class RegistrationReuseTests(unittest.TestCase):
    def setUp(self):
        self.manager = EnrollmentManager()
        self.gateway = InferenceGateway(
            enrollment=self.manager,
            entra_validator=FakeValidator(),
        )
        self.registration = self.gateway.enroll_node_with_entra(
            access_token="valid-token",
            node_id="NODE-1",
            endpoint="http://127.0.0.1:11434",
            agent_url="http://127.0.0.1:8765",
            tailnet_name="private",
        )

    def challenge(self, *, node_id="NODE-1", secret=None, challenge=None):
        challenge = challenge or secrets.token_urlsafe(32)
        secret = secret or self.registration["shared_secret"]
        payload = json.dumps({"node_id": node_id, "challenge": challenge}, sort_keys=True, separators=(",", ":")).encode()
        return challenge, hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    def verify(self, *, token="valid-token", node_id="NODE-1", challenge=None, signature=None):
        if challenge is None or signature is None:
            challenge, signature = self.challenge(node_id=node_id)
        return self.gateway.verify_node_registration_with_entra(
            access_token=token,
            node_id=node_id,
            challenge=challenge,
            signature=signature,
        )

    def test_returning_registration_is_accepted_and_secret_is_not_returned(self):
        self.assertTrue(self.verify())
        self.assertEqual(self.manager.nodes()[0].secret, self.registration["shared_secret"])

    def test_repeated_sign_in_uses_fresh_challenge_without_reenrollment(self):
        self.assertTrue(self.verify(challenge="first"))
        self.assertTrue(self.verify(challenge="second"))
        self.assertEqual(len(self.manager.nodes()), 1)

    def test_replayed_challenge_is_denied_but_fresh_recovery_challenge_works(self):
        challenge, signature = self.challenge(challenge="replay-me")
        self.assertTrue(self.verify(challenge=challenge, signature=signature))
        self.assertFalse(self.verify(challenge=challenge, signature=signature))
        self.assertTrue(self.verify(challenge="recovery"))

    def test_invalid_authentication_and_owner_mismatch_fail_closed(self):
        self.assertFalse(self.verify(token="expired-token", challenge="expired"))
        self.gateway.entra_validator = FakeValidator({"subject": "other-user", "tenant_id": "tenant-1"})
        self.assertFalse(self.verify(challenge="wrong-owner"))
        self.gateway.entra_validator = FakeValidator({"subject": "subject-1", "tenant_id": "other-tenant"})
        self.assertFalse(self.verify(challenge="wrong-tenant"))

    def test_missing_revoked_invalid_registration_and_bad_proof_fail_closed(self):
        self.assertFalse(self.verify(node_id="MISSING", challenge="missing"))
        self.assertFalse(self.verify(challenge="bad-proof", signature="0" * 64))
        self.manager.revoke("NODE-1")
        self.assertFalse(self.verify(challenge="revoked"))

    def test_first_enrollment_still_uses_enrollment_route_and_duplicate_fails(self):
        self.assertEqual(len(self.manager.nodes()), 1)
        with self.assertRaises(ValueError):
            self.gateway.enroll_node_with_entra(
                access_token="valid-token",
                node_id="NODE-1",
                endpoint="http://127.0.0.1:11434",
                agent_url="http://127.0.0.1:8765",
                tailnet_name="private",
            )
        self.assertEqual(len(self.manager.nodes()), 1)

    def test_client_first_enrollment_then_returning_and_repeated_sign_in(self):
        fresh = {}
        created = {"node_id": "NODE-1", "shared_secret": self.registration["shared_secret"]}
        with patch("provis_inference_node.entra_client.enroll_node_with_entra", return_value=created) as enroll:
            fresh = reuse_or_enroll_node_with_entra(
                config=fresh, gateway_url="https://gateway", access_token="token", node_id="NODE-1",
                endpoint="http://127.0.0.1:11434", agent_url="http://127.0.0.1:8765", tailnet_name="private",
            )
            self.assertEqual(enroll.call_count, 1)
        with patch("provis_inference_node.entra_client.verify_node_registration_with_entra", return_value=True) as verify, \
             patch("provis_inference_node.entra_client.enroll_node_with_entra") as enroll:
            first = reuse_or_enroll_node_with_entra(
                config=dict(fresh), gateway_url="https://gateway", access_token="token", node_id="NODE-1",
                endpoint="http://127.0.0.1:11434", agent_url="http://127.0.0.1:8765", tailnet_name="private",
            )
            second = reuse_or_enroll_node_with_entra(
                config=dict(fresh), gateway_url="https://gateway", access_token="token", node_id="NODE-1",
                endpoint="http://127.0.0.1:11434", agent_url="http://127.0.0.1:8765", tailnet_name="private",
            )
            self.assertEqual(first["shared_secret"], fresh["shared_secret"])
            self.assertEqual(second["shared_secret"], fresh["shared_secret"])
            self.assertEqual(verify.call_count, 2)
            enroll.assert_not_called()

    def test_client_never_auto_enrolls_after_saved_registration_denial(self):
        saved = {"node_id": "NODE-1", "shared_secret": self.registration["shared_secret"]}
        with patch("provis_inference_node.entra_client.verify_node_registration_with_entra", return_value=False), \
             patch("provis_inference_node.entra_client.enroll_node_with_entra") as enroll:
            with self.assertRaises(PermissionError):
                reuse_or_enroll_node_with_entra(
                    config=saved, gateway_url="https://gateway", access_token="token", node_id="NODE-1",
                    endpoint="http://127.0.0.1:11434", agent_url="http://127.0.0.1:8765", tailnet_name="private",
                )
            enroll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
