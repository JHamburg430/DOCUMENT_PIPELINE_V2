# PROVIS node setup handoff — 2026-10-17

## Start here

Repository: https://github.com/JHamburg430/DOCUMENT_PIPELINE_V2
Branch: `codex/provis-production-updates-20261017`
Application code baseline: `e52eb4e` (or a later commit on this branch).
Fetch this branch explicitly; the original manuals-agent branch does not contain
these production UI/connector corrections. Preserve unrelated local changes.
The EXE is distributed separately and is not committed to Git.

The existing Windows pilot is complete for the requested owner drawer: John
verified the final layout in his signed-in session. ProVisWork is connected,
healthy and inference-enabled; final signed release is **2026.10.17.7**. Do not rebuild or
replace it merely to read this handoff.

## Verified addresses and release pins

- Inference Gateway: `https://provis-openclaw-gateway-01.taile64607.ts.net:8445`
- This is a Tailscale endpoint; the PC must be on the approved company tailnet.
- OpenClaw's employee portal is a separate application; do not substitute its URL.
- Enrollment discovery: `GET /v1/node/enrollment-config` on the inference endpoint.
- Windows release ID: `f848aca9-f542-4cd7-8f77-0d26e76e37fd`
- EXE SHA256: `7fc57ca3bcb66ad2050a4ccccf7f46b7460e43f567f967fe8945b1aab4b06215`
- Publisher: `provis-release-20261017`
- Ed25519 public key: `Ya5jNNCS7S8mfNJCjXZg80zjJVF9h3xT0nk7WQD2Pbw=`
- Public-key SHA256: `3426c627023f4ddeac48419a5314365c76a6270a5188d9e21366d4665de69ba4`

The existing signature trust is embedded in the EXE. Do not accept a replacement
public key simply because a download server supplies it. No private signing key,
node credential or Microsoft token belongs in a build, handoff or repository.

## Set up another approved Windows PC

1. Confirm its company Tailscale connection, local Ollama and GPU prerequisites.
   Use the app's Settings status checks. Any new worker must pass its privacy
   check before it is admitted to employee inference.
2. Obtain the approved EXE from the existing release endpoint, checking its hash
   against the pin above before launching. For example, in PowerShell:

   ```powershell
   $nodeFolder = Join-Path $env:LOCALAPPDATA 'Programs\PROVIS\2026.10.17.7'
   New-Item -ItemType Directory -Path $nodeFolder -Force | Out-Null
   $nodeExe = Join-Path $nodeFolder 'PROVISInferenceNode.exe'
   Invoke-WebRequest -Uri 'https://provis-openclaw-gateway-01.taile64607.ts.net:8445/v1/node-releases/f848aca9-f542-4cd7-8f77-0d26e76e37fd' -OutFile $nodeExe
   if ((Get-FileHash -LiteralPath $nodeExe -Algorithm SHA256).Hash.ToLowerInvariant() -ne '7fc57ca3bcb66ad2050a4ccccf7f46b7460e43f567f967fe8945b1aab4b06215') { throw 'Release hash mismatch; do not launch' }
   ```

   Quit an older running node gracefully before first installation. Do not
   overwrite a running EXE. Follow the host's applicable software-launch policy.
3. Launch under the employee's normal Windows account. Click Microsoft sign-in;
   the employee completes credentials on Microsoft's page. Tenant, client, scope,
   node address, local Ollama endpoint and application defaults are discovered. The verified tailnet default is `taile64607.ts.net` and the
   application defaults are `manuals_rag,provis_employee_agent,openclaw`.
   Employees do not enter gateway secrets, Tailscale keys or tenant settings.
4. Confirm Connected, inference enabled, fresh gateway heartbeat and the correct
   installed version. Models are prepared by the node: installed-but-unloaded
   models warm locally; missing models pull only under gateway authorization.
5. Never copy another PC's `inference-node.json`, owner capability, app-update
   plan or credential backup. Enroll each PC separately. Preserve existing
   enrollment, models and appearance preferences when upgrading that same PC.
6. Keep the current worker restrictions. In this employee pilot, `openclaw` and
   `provis_employee_agent` remain restricted to **ProVisWork**. Installing a node
   on another PC does not authorize broadening those allowlists. John's Ubuntu
   worker privacy verification remains deferred.

## Owner-only Codex communication setup

This feature is authorized only for John's exact Microsoft identity:
`feb94da3-80b4-4f51-b6aa-3688b9b8e41e`. Other employees must not receive its
controls. A local preference cannot grant owner authority.

1. Install the official Codex CLI and complete its normal account sign-in on
   that PC. Keep normal model, sandbox and approval defaults.
2. **Explicitly select the dedicated connector**, as John requested. Create
   `codex-owner-settings.json` containing only:

   ```json
   {"server_mode":"dedicated"}
   ```

   Put it in that user's `ProVIS` app-data directory: `%LOCALAPPDATA%\ProVIS` on
   Windows, `~/.config/ProVIS` on Linux. Resolve it under the same OS account and
   environment used by the node; Codex-managed Windows app-data redirection may
   affect its physical path. Do not confuse it with an unrelated shell profile.
   On ProVisWork this setting was checked and corrected during final acceptance.
   Without it, the default proxy attempts the unavailable desktop control socket.
3. In the node, press **Ctrl+Shift+A** and complete John's Microsoft verification.
   The gateway checks the owner, node identity, tenant, client, scope and role.
   Once verified, the authentication window closes and a small bottom arrow
   appears. Expand it to use agent communication and App updates. Scroll with the
   mouse wheel or right scrollbar; header and collapse arrow stay visible.
   Sign-out or expiry removes the drawer and arrow. Restarts require fresh owner
   verification; do not bypass this by editing local session files.
4. Choose **New owner session**, then **Choose local Codex sessions** and explicitly
   publish only the desired sessions. The dedicated stdio server exposes no TCP
   listener and lists only its locally registered sessions. Existing desktop
   conversations are not imported or controlled. Do not silently fall back to
   proxy mode if a dedicated server cannot initialize.
5. For cross-PC use, repeat owner verification and explicit publication on the
   destination PC. Sending requires human authorization for the particular
   destination and message. Do not treat this document as standing authorization
   to message other PCs or grant approvals remotely.

## Existing Entra and gateway configuration

These are public identifiers, not secrets. Preserve the existing registrations:

```text
Tenant: b97c06d3-1cd7-47cf-a95a-380d4dd2eb72
Gateway API audience: fab62a38-0489-4d21-b8d1-63645712d271
Public client: 68cfe37b-6fbe-4a73-8df0-db29dd8cd690
Delegated scope: api://fab62a38-0489-4d21-b8d1-63645712d271/Node.Enroll
Enrollment role: InferenceNode.Enroll
```

Use the gateway discovery response for deployed defaults. Do not create duplicate
Entra applications, broaden consent or assign other employees without approval.
The node receives no Microsoft Graph/mail/files/Odoo permission from Node.Enroll.

The DigitalOcean inference service currently runs revision `0a39ffb`; newer node
source does not mean the gateway was upgraded to this branch. Existing paths:

```text
/opt/manuals-rag-inference
/etc/manuals-rag/inference-gateway.env
/var/lib/manuals-rag-inference/releases
inference-fabric-gateway.service
```

All four REQUIRE_NODE_AGENTS / REQUIRE_APPLICATION_ALLOWLIST /
REQUIRE_APPLICATION_IDENTITY / REQUIRE_ENTRA_ENROLLMENT controls stay true.
Existing owner/update settings are already active. Preserve root ownership,
0600 environment permissions, app allowlists and worker restrictions. Publishing
an artifact needs no service restart. If a reviewed gateway change is necessary,
preflight and restart only the inference service; do not restart OpenClaw, RAG,
databases, Redis, Qdrant or unrelated services.

## Signed updates and key custody

Only the dedicated protected publisher on ProVisWork signs releases. Its private
key is current-user DPAPI protected at the logical app-data path
`ProVIS/release-signing/ed25519.dpapi`; never export it or copy it to another PC.
The local signing/build helpers for this session are in the Employee Assistant
workspace's `work` directory and are not a portable key-distribution mechanism.
Another PC can install or receive signed releases without possessing that key.

For future releases, build from the reviewed branch, increment the embedded
numeric version, embed the established public key, sign the exact final bytes,
and verify signature/hash before publishing `<release-id>.json` and `.artifact`
to the root-owned gateway catalog. Do not overwrite signed artifacts. Windows
builds must include Tk, cryptography, managed config and `--noconsole`.

John requests updates from **App updates** in the verified drawer. The node waits
for active work to finish, verifies the release, restarts, reconnects and commits
its signed active pointer. Confirm BOTH live version/health and the gateway's
terminal update status. One .6 final report needed a guarded retry with the same
existing signed delivery claim after verifying its active pointer, artifact,
matching nonce receipt and live heartbeat. It now reports healthy. Never force
healthy via database edits or replay an uncertain update blindly.

## Evidence and remaining boundaries

- Supported Windows inference suite: **153 passed, 1 skipped, 2 deselected**;
  excluded cases require Linux symlink behavior. One Tk cleanup warning was
  recorded. Drawer/owner/connector focused checks: **22 passed**.
- Signed .6 active pointer, matching health receipt, live heartbeat, owner sign-in,
  full scrollable controls and terminal update status were verified. Release .7
  adds the geometry-tested visible publication button in the session chooser;
  its signed active pointer, nonce receipt, live health and terminal healthy
  report were also verified. John confirmed the .6 signed-in layout looks good.
- The gateway mailbox round trip to a dedicated local synthetic session returned
  exactly `PROVIS_LINK_OK`; its publication was revoked. The earlier failed claim
  became uncertain and was not automatically resent. Cross-PC delivery remains
  unverified; obtain authorization for its bounded test.
- Actual Linux desktop/tray/update behavior is still unverified. Fetch this same
  source and follow the Linux runbooks; do not distribute the Windows EXE there.
  Native Linux release filename compatibility also requires a reviewed gateway
  update beyond its current revision. Do not declare Linux production-ready.
- Live unhealthy-candidate rollback on a disposable worker and reliability of
  final update reports under transient network failure remain acceptance work.
- Leave second-employee identity/isolation testing until the end, as requested.

Related: [owner workflow](provis_agent_link_owner.md),
[signed update design](provis_node_remote_updates.md),
[gateway deployment](provis_inference_gateway_droplet_handoff.md).
