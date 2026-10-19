# Returning-node registration verification

This protocol verifies an existing registration. It never returns a node
credential, rotates a secret, changes ownership, or falls back to enrollment
after a saved registration is denied. First-time empty configurations retain
the separate, existing enrollment flow.

1. The node presents its current Entra bearer token and node ID to
   `POST /v1/nodes/registration-challenge/entra`.
2. The gateway validates the token using its configured issuer, audience,
   tenant, public client, delegated scope and role/group rules. Its subject and
   tenant must match the persisted Entra registration, whose secret must still
   match the active heartbeat credential.
3. The gateway issues a random 256-bit nonce, held in memory for 60 seconds.
   The record binds the nonce to that node, subject, tenant and the SHA-256
   digest of the exact bearer token. The bearer token and node secret are not
   stored in this cache. At most four nonces per node and 4096 globally may be
   outstanding; exhausted capacity fails closed without evicting live entries.
4. The node HMACs canonical JSON containing the fixed protocol purpose,
   node ID, nonce and bearer-token digest using its already-saved node secret.
   It submits the nonce and signature to
   `POST /v1/nodes/verify-registration/entra` with the same bearer token.
5. The gateway repeats token, ownership and secret checks, verifies the nonce
   binding and HMAC, then consumes the nonce atomically. Only one concurrent
   request can succeed. The response remains only accepted or denied.

Expired, unknown, consumed, token-mismatched, legacy client-generated or
previous-process nonces are denied. A restart clears all outstanding nonces,
so old proofs cannot become valid again. Failed authentication, wrong proofs,
revocation, secret mismatch and unavailable protocol routes do not trigger
credential creation or recovery enrollment.

Both packaged Python implementations use the same protocol. A future rollout
must deploy the reviewed gateway routes and a separately reviewed/signed node
release together. An older gateway returns a denial/unavailable result to the
new client; it is not bypassed. This source change is not authorization to
deploy, restart, sign, publish or modify any live enrollment.

Security CI runs the real HTTP revocation regression and adversarial checks
for replay, expiry, restart, concurrent consumption, exact-token binding,
ownership, secret mismatch, bounded cache behavior and both HTTP clients.
