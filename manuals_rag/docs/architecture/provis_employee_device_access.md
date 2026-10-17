# PROVIS Employee and Device Access Architecture

Status: design proposal for review. Authentication enforcement is not enabled by
this document or by the listener-refresh changes. Do not roll it out without
action-time approval and a tested recovery procedure.

## Existing identity boundaries

- The inference node desktop uses Microsoft's public-client sign-in for node
  enrollment and owner actions. The gateway validates Entra tokens for those
  operations.
- Each enrolled node receives a separate random shared secret. Signed node
  heartbeats and node-to-gateway inference identify that enrolled node.
- Owner update controls use an Entra identity check. The release pipeline signs
  fixed platform-specific artifacts; nodes verify a pinned public key before
  applying an update.
- Application inference currently uses a server-held application token in
  addition to gateway routing policy. That token authenticates the application
  service, not the employee at the keyboard.
- Therefore having the PROVIS Inference Node App installed is not an access
  control for OpenClaw. Direct access to any reachable OpenClaw/gateway path
  must be rejected by the server unless it validates a user and device session.

## Required login flow

1. The PROVIS app signs the employee in through Entra's public-client flow. The
   gateway validates the tenant, issuer, audience, expiry, object ID and
   approved group/role. Email address alone is never an identity key.
2. The PC creates a device key in its operating-system key store. On Windows,
   use a non-exportable TPM/CNG key. On managed Linux PCs, use a TPM2-backed
   key or a reviewed hardware-backed secret-service implementation. Do not put
   a device private key in the app config, logs, or repository.
3. An admin-approved device enrollment binds the public key to the Entra tenant,
   employee object ID and stable managed-device identity. Entra/Intune
   compliance or an equivalent device-attestation source must establish that
   identity; a caller-supplied PC name or a copied node secret is insufficient.
4. At login, the server sends a one-use, short-lived challenge. The app proves
   possession by signing the challenge and the target gateway/audience with
   the registered device key. The server consumes the challenge, verifies the
   employee and device binding, and issues a short-lived session token bound to
   that public key (DPoP-style proof of possession).
5. Every OpenClaw-facing request, including direct requests to the DigitalOcean
   gateway, must pass through the same server-side user/device authorization
   boundary. Require the user token and a fresh request proof signed by the
   bound key. Keep the application token only for service-to-service identity;
   it cannot substitute for employee authentication.
6. Authorization is the intersection of active employee status, approved
   application role/group, device enrollment/revocation state and the
   application's server-owned policy. Scope records by tenant and user. Log
   identity, device ID, action, result and correlation ID, without prompts,
   responses, bearer tokens or signing material.
7. Provide an admin recovery route that revokes a lost device, requires a new
   admin-approved enrollment and records the event. Short session lifetimes and
   a server-side revocation check limit the impact of a lost device.

## Enforcement and rollout conditions

- Deny requests by default at the OpenClaw/gateway ingress. The desktop UI is
  only a login client; hiding UI controls or checking for an installed app does
  not enforce access.
- The gateway must validate the Entra token and device proof on every protected
  request. A reverse proxy may enforce the same policy, but must not trust
  caller-provided identity headers from an untrusted network path.
- Preserve the existing enrolled-node secret, signed updater keys, application
  service token, and pilot allowlists. Employee/device login is a separate
  authorization layer; it does not widen inference worker allowlists.
- Before enabling enforcement, verify a supported credential store on each PC,
  managed-device identity claims for Windows and Linux, token audience/scopes,
  key rotation and recovery, and cross-tenant isolation.
- Required negative acceptance: no app installed; invalid, expired or wrong-
  tenant token; valid user with a foreign device key; revoked device; stale or
  replayed proof; direct gateway bypass; disabled user; unapproved role; and
  other-employee isolation. All must fail closed. Positive tests must cover
  sign-in, device challenge, authorized OpenClaw request, token renewal and
  admin recovery.
- The current pilot's documented application-worker allowlists are limited to
  ProVisWork. Expanding those lists or enabling employee authentication changes
  security posture and employee access; obtain action-time approval first.

## Deployment plan to prepare after code review

1. Stage the reviewed gateway and node-app release without enabling the new
   flags. Keep existing node secrets, signing public keys, Entra app identity,
   application worker allowlists and firewall rules unchanged.
2. Install on one approved test PC and use a test employee/device. Verify direct
   gateway rejection and all positive/negative identity cases before touching
   production employees.
3. Activate only after explicit approval, first for the named pilot PC and
   employee. Monitor authentication-denial and gateway-health metrics. Do not
   enable or broaden any worker allowlist as part of login activation.
4. Rollback by disabling the new server-side gate and returning to the prior
   gateway release/config. Revoke pilot device sessions and keys. Do not remove
   the existing node enrollment or signing credentials as a rollback shortcut.

No device keys, user records, gateway settings, or employee access were changed
while preparing this design.
