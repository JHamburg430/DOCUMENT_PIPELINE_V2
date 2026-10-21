# Owner-only Codex communication setup

The node's verified owner drawer contains **Set up Codex communication** on
Windows and Linux. No control is constructed for an unverified employee.

1. Verify the owner account through the node's existing Microsoft flow.
2. Expand the private drawer and select **Set up Codex communication**.
3. **Check setup** reads the local Codex entry. When configured, it also tests
   MCP initialize/tools-list locally without loading credentials or contacting
   the gateway. This does not establish remote delivery health.
4. Review the exact command. **Add to Codex** or **Replace in Codex** requires
   a fresh confirmation before running `codex mcp add provis-agent-link`.
   Registration is bound to the checked configuration revision and an owner
   session that remains valid immediately before the write. A disabled entry
   stays disabled and must be reviewed directly in Codex.
5. Readback must confirm the command, arguments, enabled state and explicit
   `PROVIS_AGENT_LINK_SESSION_PATH`. Copying a command does not execute it.

The frozen command uses the installed signed application in its dedicated
`--agent-link-mcp` mode. It opens no node/tray window, listener, console,
updater, enrollment or model runtime. Windows windowed builds reconnect only
to pipes already inherited from Codex; they do not attach a console or change
network/security settings. Source checkouts use Python and
`manuals_rag/scripts/run_agent_link_mcp.py` instead.

The explicit environment value is a **file path**, never a token. It ensures
Codex finds the node's existing owner capability even when Windows launch
contexts use different LocalAppData roots. The protected capability remains
short-lived. Every MCP tool call reloads it; sign-out and renewed sign-in are
observed by a running Codex adapter. No session is published or message sent
by setup; sending still requires exact recipient/message approval.

## Connection diagnosis

- A Tailscale-online machine may still have its node app stopped. Verify the
  enrolled node listener and identity, not just peer reachability.
- Verify the gateway's HTTPS health using its configured name and normal TLS
  checks. An executor's DNS/network failure does not prove host-wide failure.
- Verify a fresh accepted node heartbeat and version in Compute Control.
- Treat local MCP configuration, local protocol readiness, owner authentication
  and gateway delivery as separate checks. Do not label setup connected merely
  because the Codex entry exists.
- For Dell's existing pilot enrollment, the recorded listener is
  `http://100.106.245.96:8768`. Its historical enrollment is in the Codex
  LocalCache root. Preserve that file; do not re-enroll to work around a
  missing listener. The existing Dell installation task owns recovery.

## Release gate

This source change does not retrofit the already-published .20.6 app. A new
reviewed Windows/Linux binary needs the normal frozen MCP initialize/tools
test, signed release verification, explicit deployment approval and staged
live checks. Existing replay protection, signed updates and enrollments are
unchanged. No new credentials or persistent production registrations are
created by source/unit tests; the real-CLI integration uses an isolated
`CODEX_HOME` with an absent synthetic owner-session path.

Command syntax: https://learn.chatgpt.com/docs/extend/mcp?surface=cli
Windowed runtime behavior: https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html
