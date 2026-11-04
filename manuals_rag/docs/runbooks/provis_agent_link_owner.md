# Owner Agent Link

The private node panel is opened with **Ctrl+Shift+A**. It does not grant access
by itself. Fresh Microsoft sign-in is checked by the gateway against the
configured tenant, approved public client, delegated scope, enrollment role,
and the owner's exact Entra object ID. Node/app/operator inference credentials
do not authorize this feature. The enrolled node's identity must also match.

Gateway activation is separate from deploying the node UI. Set the root-owned
gateway environment's `INFERENCE_AGENT_LINK_OWNER_OBJECT_ID` to
`feb94da3-80b4-4f51-b6aa-3688b9b8e41e` for John's requested account only, deploy
this revision, and restart only `inference-fabric-gateway.service` after its
preflight. Preserve all existing inference authentication and worker restrictions.
The feature is disabled when the owner setting is absent.

## Session workflow

1. Open the private panel and verify the Microsoft account on each participating
   PC. The short-lived owner capability expires after at most one hour, bounded
   by the Microsoft access-token expiry. Microsoft tokens are not saved.
2. The node connects to its connector-owned local Codex server, creates or
   recovers the dedicated Node Channel conversation, and automatically registers
   it. Registration never occurs before both owner verification and local Codex
   connection succeed.
3. The node renews a 30-second public-session lease every 6 seconds. On a sender
   PC, select one exact current PC name and send one owner-approved message.
   Non-addressed nodes receive no job and do not invoke Codex.
4. Busy sessions remain queued until idle. An interrupted, claimed delivery
   becomes **uncertain** and is never automatically resent. Inspect the target
   conversation before deciding whether to send a new request.
5. Sign-out unregisters immediately. Crash, restart, or lost connectivity expires
   the registration closed after 30 seconds; restart recovery re-registers the
   same stable local conversation after owner capability renewal.

The connector does not create threads, change models, override sandbox/approval
policies, restart Codex, or fall back to a second independent agent server.
Unsupported approval requests are not granted remotely; the delivery reports
that a local approval is needed. Normal employee inference is independent.

Owner sessions are protected with Windows user-bound DPAPI or a private Linux
file. The gateway stores bounded messages/replies in a private SQLite mailbox,
retains delivery envelopes for at most 24 hours, and removes submitted message
text on completion. Sanitized audit receipts store identifiers, status,
timestamps, and hashes only. Sign out removes the node's registration and local
session. The same operating-system account and company IT administrators remain
trusted.

## MCP interface

The loopback MCP exposes `provis_node_channel_status`,
`provis_list_node_workspaces`, `provis_list_connected_nodes`,
`provis_message_history`, and `provis_send_message`. The status tool reports a
stale node/gateway protocol explicitly instead of returning an empty target
list. Sending requires one exact listed `pc_name`, the exact user-approved
message, and explicit confirmation. Embedded addresses, delivery envelopes,
nonzero hops, self-targets, and expired targets are rejected.

## Verification boundary

Authorization, TTL/revocation, automatic gated registration, claims, duplicate handling,
busy targets, protected capabilities, and HTTP gates are executable tests.
The current ProVisWork Codex desktop process uses its private stdio transport;
its default shared control socket is unavailable (Windows error 10050).
Connecting that live desktop instance needs a supported control-socket setup.
Do not claim a live agent-to-agent round trip until both PC connectors are
verified and an explicitly authorized synthetic message has returned a reply.

Protocol reference: https://learn.chatgpt.com/docs/app-server

Node Channel contract: [provis-node-channel/1](provis_node_channel_protocol_v1.md)
