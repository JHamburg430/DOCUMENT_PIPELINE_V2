# PROVIS Node Channel protocol v1

Protocol identifier: `provis-node-channel/1`

This contract joins the verified node UI, loopback MCP adapter, and gateway
owner API. It is owner-scoped, addressed-only, and fail-closed. It never grants
remote approvals, exposes enrollment credentials, transports hidden prompts, or
allows one employee to discover another employee's workspace.

## Negotiation

`GET /v1/agent-link/availability` returns the protocol identifier, supported
protocols, minimum node release, component release, 30-second lease, and
6-second heartbeat interval. Every owner API `POST` includes
`X-ProVIS-Node-Channel-Protocol: provis-node-channel/1`. A missing or different
value receives HTTP 409 with `node_channel_version_mismatch`; it must never be
reported as an empty workspace.

## Owner verification and lifecycle

1. `session` validates the Entra object ID against the configured owner and
   binds the capability to one Entra-enrolled node, tenant, subject, current
   enrollment secret hash, and a maximum one-hour expiry.
2. The node connects to its connector-owned local Codex server and creates or
   recovers the dedicated Node Channel thread.
3. `workspace` discovers only the same owner workspace. `register` publishes
   that local thread and stable agent ID for 30 seconds. Registration cannot
   happen before steps 1 and 2 succeed.
4. `heartbeat` renews only the exact node, agent ID, owner capability, and
   workspace registration. The UI refreshes it every 6 seconds.
5. `unregister` makes the session non-current immediately. Process loss is
   fail-safe because the lease expires after 30 seconds. `logout` unregisters
   before invalidating the owner capability.

`workspace` returns enrolled participants plus `online`, `current`,
`last_seen`, and `lease_expires_at`. MCP exposes only current registrations as
public sessions.

## Addressed delivery

`post` accepts one UUID, `kind`, exact `target_node_id`, bounded message,
`origin` (`owner_ui` or `owner_mcp`), and `hop_count: 0`. The gateway rejects a
self target, offline target, embedded leading address, nonzero hop, replay with
changed content, cross-workspace peer, or a previously delivered PROVIS
envelope. It stores one immutable `@<exact-pc-name>:` envelope and one recipient
snapshot.

`poll` returns work only to that exact registered node and connector-owned
thread. Before any model call, the node independently verifies the target,
single-recipient snapshot, and visible exact address. Address mismatches are
completed as `skipped` without invoking Codex. Busy registered sessions remain
queued until idle; the receiver runs the response in its existing registered
thread. `executing` and `complete` require the one-time claim and stable agent
ID. A successful reply remains on the original job and is never posted as
another channel message.

The gateway records the sender's registered local channel thread (or the
sender's registered direct thread for a DM) as the reply destination. The source
node app polls `reply-poll` and shows each completed response in a pending
Notifications list. The user must choose **Accept & add to session** or
**Reject** for each response. Accept appends the response as an assistant
history item to the registered session using Codex `thread/inject_items`, then
records `reply-decide: accepted`; reject records `reply-decide: rejected`
without adding content to the session. Neither action starts or steers a turn.
The injected item carries the original request ID as an idempotency marker.
Within one running connector process, retry after an interrupted acknowledgement
does not append the item twice. The current Codex server on supported installs
does not expose the history lookup needed to prove that an earlier injection
survived a connector restart. Therefore a pending notification can reappear
after restart; inspect the destination session before accepting that same
request again. The gateway verifies the source capability, job ownership,
completion status, decision, and registered destination. Older gateways that
do not implement these optional reply actions remain usable, but cannot show
response notifications until upgraded.

## Replay, expiry, and audit

Message UUIDs are idempotent only for identical conversation, source, and body
hashes. Delivery claims are random and expire to `uncertain` after ten minutes;
uncertain work is never automatically replayed. Owner capabilities, enrollment
binding, leases, and workspace membership are checked on every operation.

The gateway appends sanitized receipts for registration, heartbeat, expiry,
unregister, post, claim, execution, and completion. Receipts contain identifiers,
status, timestamps, and hashes only—never credentials, raw messages, prompts, or
model output. `audit` is restricted to the verified owner workspace.

