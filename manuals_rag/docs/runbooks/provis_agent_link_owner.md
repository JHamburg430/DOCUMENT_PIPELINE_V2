# Owner Agent Communication

Owner verification opens with **Ctrl+Shift+A**. The Microsoft owner sign-in is
checked by the gateway against the configured tenant, approved public client,
delegated scope, enrollment role, exact owner Entra object ID, and the node's
existing enrollment identity. Inference credentials do not authorize this
feature. The owner capability expires at the earlier of the Microsoft token
expiry or one hour. Microsoft tokens are not saved.

Gateway activation remains a separate deployment step. The root-owned gateway
environment must set `INFERENCE_AGENT_LINK_OWNER_OBJECT_ID` to
`feb94da3-80b4-4f51-b6aa-3688b9b8e41e` for John's requested account, and the
gateway revision must be deployed after preflight. Preserve current inference
authentication and worker restrictions. The feature remains unavailable when
the owner setting or the required conversation API revision is absent.

## Chat workflow

After owner sign-in, the app connects automatically to a connector-owned Codex
app-server over private stdio. It does not proxy or list the user's unrelated
desktop threads. Each node keeps one locally owned Codex thread for the Node
Channel and one for each direct-message peer. The private
`ProVIS/codex-owner-sessions.json` and `codex-owner-conversations.json` files
retain the mapping so reconnects reuse those threads. The normal Codex model,
sandbox, and approval settings remain in effect; remote messages cannot grant
local approvals.

The chat tabs include one **Node Channel** and one-to-one conversations with
nodes enrolled to the same Entra subject and tenant. The roster reports online
when a node has a fresh owner-session registration; otherwise it is shown
offline. Posts record a fixed recipient snapshot. Nodes that join after a post
can read channel history but are not added to that post's recipient list.
Direct messages can remain queued for an offline enrolled peer and are bound to
that peer's local conversation when it registers again.

Recipient states progress through `queued`, `admitted`, `executing`, and
`replied`, with `failed`, `blocked`, `skipped`, or `uncertain` where relevant.
The execution record names the actual node. A channel agent is instructed to
ignore unrelated posts, respond once when addressed, and never rebroadcast an
agent reply. The server uses a unique message/recipient record and a stable
delivery ID so reconnects cannot add duplicate recipients or accepted replies.
Uncertain claims are not automatically retried; inspect the conversation
before manually sending a new message.

Conversation history and pending recipient delivery are stored in the private
gateway SQLite mailbox. Owner sign-out expires the local capability and removes
the node's online registration; it does not erase the user's Codex conversation
history or its stable local thread mapping. Ordinary node inference and signed
app updates remain separate from owner chat.

## Signed app updates

The private owner area retains **App updates**. Its panel lists enrolled nodes
and signed releases, then uses **Update selected PC** after an explicit
confirmation. Source logic verifies the signed manifest and downloaded artifact
hash, waits for the node to become idle, preserves the existing configuration,
launches the candidate, and rolls back to the previous app if the candidate
fails its health probe. An enabled heartbeat indicates updater readiness; it
does not prove that an update has been requested or applied.

## Verification boundary

The chat state machine, local session reuse, UI tabs, duplicate handling,
recipient snapshotting, and the Dell/sender identity regression are covered by
local tests. The cross-PC chat experience requires the matching gateway API and
owner setting to be deployed. A source-level test does not establish that the
current gateway or every employee node has that revision. Do not infer network
failure from a Linux node's OS or routing output. Never change enrollment,
credentials, VPN, or access policy as a chat workaround.

The previous manual **New owner session**, **Choose local Codex sessions**, and
**Refresh remote sessions** flow has been replaced in source by automatic chat
registration. The release and deployment remain separate actions.
