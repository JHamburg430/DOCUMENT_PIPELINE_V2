# Owner Agent Link

Owner verification is opened with **Ctrl+Shift+A**. After successful Microsoft owner verification, a small arrow appears at the bottom of the main node panel. It expands or collapses the owner controls inside the app; ordinary employees do not see the arrow. Signing out or expiry removes the controls and arrow. It does not grant access
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
2. Choose **local Codex sessions** and publish only the desired existing threads.
   Existing-server mode uses `codex app-server proxy`; an operator can provide
   an existing local socket through `PROVIS_CODEX_CONTROL_SOCKET`.
   John's approved dedicated mode instead starts a connector-owned Codex
   app-server over private stdio. It opens no TCP listener and keeps the user's
   normal model, approval and sandbox defaults. Set `server_mode` to `dedicated`
   in the private per-user `ProVIS/codex-owner-settings.json` file (under
   LOCALAPPDATA on Windows or ~/.config on Linux).
   Choose **New owner session** to create an empty session, then publish it
   explicitly. Dedicated mode lists only those locally registered sessions;
   it never imports existing desktop conversations. A per-user lock prevents
   another dedicated connector from operating those sessions concurrently.
3. On a sender PC, refresh published sessions, select a destination, and send a
   message. Replies appear only in the private panel, never in node activity.
4. Busy sessions remain queued until idle. An interrupted, claimed delivery
   becomes **uncertain** and is never automatically resent. Inspect the target
   conversation before deciding whether to send a new request.

The connector creates a thread only through **New owner session** in explicitly
selected dedicated mode. It does not change models, override sandbox/approval
policies, restart the desktop, or silently fall back between connection modes.
Unsupported approval requests are not granted remotely; the delivery reports
that a local approval is needed. Normal employee inference is independent.

Owner sessions are protected with Windows user-bound DPAPI or a private Linux
file. The gateway stores bounded messages/replies in a private SQLite mailbox,
retains them for at most 24 hours, and removes submitted message text on
completion. HTTP access logging remains disabled. Published targets expire
unless renewed. Sign out removes the node's published targets and local session.
The same operating-system account and company IT administrators remain trusted.

## Assistant command interface

The bundled `provis_agent_link.py` can be used by a user-authorized assistant:

```text
python provis_agent_link.py targets
python provis_agent_link.py send --input message.json
python provis_agent_link.py messages
```

`message.json` contains `id` (a fresh UUID), `node_id`, `thread_id`, and `message`.
Keep the same UUID when checking an uncertain submission; never retry blindly.
Use `--input -` for JSON on stdin. This interface requires the owner's active
panel sign-in and does not grant an assistant standing permission to send
arbitrary messages. Human authorization must exist for the requested message.
For the Linux zipapp, use `python3 inference-node.pyz --agent-link targets`.

## Verification boundary

Authorization, TTL/revocation, explicit publication, claims, duplicate handling,
busy targets, protected capabilities, and HTTP gates are executable tests.
The current ProVisWork Codex desktop process uses its private stdio transport;
its default shared control socket is unavailable (Windows error 10050).
The approved dedicated stdio connection initialized successfully on ProVisWork;
this does not grant control of the live desktop's existing conversations.
The approved single-PC gateway mailbox round trip on ProVisWork completed and
returned exactly PROVIS_LINK_OK. Its test publication was revoked afterward.
Cross-PC delivery has not been verified; require explicit authorization for a
specific target/message before testing it. The final .6 owner drawer and
scrollable layout were checked in the signed-in employee session.
See [the setup handoff](provis_node_agent_setup_handoff_20261017.md).

Protocol reference: https://learn.chatgpt.com/docs/app-server
