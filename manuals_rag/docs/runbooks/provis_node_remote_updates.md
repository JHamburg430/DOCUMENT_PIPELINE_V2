# PROVIS node app remote updates

Implementation status: staged and tested locally. The live droplet has not been
changed for this feature. No production publisher key or release was created.
The running ProVisWork node has not been replaced by an updater-enabled build.

## Employee and owner experience

Normal employees continue to see only events/utilization and Microsoft sign-in.
Windows and Linux use the same updater. John opens Ctrl+Shift+A, verifies his
Microsoft account, and chooses **App updates**. Select a PC and compatible signed
release, then **Update selected PC**. Progress refreshes every ten seconds.
Server-side authorization uses John's exact Entra object ID, not the visibility
of the menu, email spelling, or a local configuration switch. Agent sessions do
not have to be published or connected for updates to work.

A node authenticates its own poll/report using its current enrollment secret,
HMAC, timestamp and replay nonce. This credential cannot request updates for
another node or publish software. A node without the trusted bootstrap reports
that an initial installation is needed.

## Release trust and first installation

The administrator must select an offline Ed25519 release-signing private key.
Keep it outside this repository, node packages, gateway environment and activity
logs. Use a protected signing machine or existing organization key service.
Only its base64 raw 32-byte PUBLIC key goes into managed node configuration and
the gateway. The updater does not establish trust by downloading a key from the
gateway. Test keys are generated only inside pytest temporary directories.

Create nonsecret updater JSON, for example:

```json
{
  "enabled": true,
  "version": "2026.10.17.1",
  "public_keys": {"provis-release-1": "<base64 Ed25519 PUBLIC key>"}
}
```

Build the managed bundle and Linux zipapp using the existing verified HTTPS URL:

```sh
export PYTHONPATH=manuals_rag/packages/inference_fabric/src:manuals_rag/scripts
python manuals_rag/scripts/build_inference_node_bundle.py \
  --gateway-url https://provis-openclaw-gateway-01.taile64607.ts.net:8445 \
  --update-config /secure/public-updater-config.json --output /tmp/provis-node.zip
python manuals_rag/scripts/build_inference_node_app.py \
  --gateway-url https://provis-openclaw-gateway-01.taile64607.ts.net:8445 \
  --update-config /secure/public-updater-config.json --output /tmp/inference-node.pyz
```

On Windows use the bundle's PyInstaller build helper (`--onefile --noconsole`,
embedded managed JSON). Its build environment needs `cryptography` so the public
signature verifier is included. Windows employees need neither Python nor a
signing key. Linux needs system Python, Tk and `python3-cryptography`; optional
GTK/AppIndicator enables the tray. Linux process/desktop behavior still needs a
live Linux acceptance check; this Windows PC has no Linux runtime available.
Source launchers and headless node agents deliberately do not advertise desktop
self-update support. Use the packaged EXE or PYZ for remote replacement.

Install this trusted bootstrap once on each approved PC under the employee's
normal account. Existing credentials, UI preferences and Ollama models remain
in their existing locations. It does not create an elevated updater service.
Subsequent signed releases can be delivered remotely.

## Publishing later app releases

Increment the numeric app version in the embedded config for every release.
Build and test the EXE/PYZ, then sign the actual artifact:

```sh
python manuals_rag/scripts/publish_inference_node_release.py \
  --artifact /release/PROVISInferenceNode.exe --output /release/catalog \
  --private-key /offline/provis-release-signing.pem --publisher provis-release-1 \
  --version 2026.10.17.2 --platform windows-x86_64
```

Linux platforms are `linux-x86_64` and `linux-aarch64`. Each release has a UUID,
platform, fixed filename, numeric version, size, SHA256 and Ed25519 signature.
There is no command, installer argument, destination path or external URL in the
signed manifest. Copy `<UUID>.json` and `<UUID>.artifact` into the operator-owned
release catalog. Artifacts contain software only and are downloadable on the
existing tailnet HTTPS endpoint; they must contain no credentials or records.

Configure these optional root-owned gateway environment entries:

```
INFERENCE_AGENT_LINK_OWNER_OBJECT_ID=feb94da3-80b4-4f51-b6aa-3688b9b8e41e
INFERENCE_NODE_UPDATE_RELEASE_DIR=/var/lib/manuals-rag-inference/releases
INFERENCE_NODE_UPDATE_PUBLIC_KEYS_JSON={"provis-release-1":"<PUBLIC key>"}
```

Keep the existing enrollment/identity/application/node-agent restrictions.
Preflight and restart only `inference-fabric-gateway.service` when deploying the
reviewed code/config. OpenClaw and other services do not need a restart for this
feature. This document is preparation, not a claim that activation occurred.
For key rotation, distribute a bootstrap/release trusting both old and new public
keys before publishing a release signed only by the new key.

## Application and recovery behavior

The node verifies signature, platform, strictly newer version, byte count and
SHA256. Downloads are bounded, same-gateway HTTPS only and reject redirects.
Maximum artifact size is 256 MiB. It keeps at most five release directories;
an administrator must inspect and remove old inactive directories when full.

After download, the node pauses admission, publishes a heartbeat, and waits up to
120 seconds for inference, model preparation and owner-message delivery to finish.
A busy node rejects the update without terminating work and restores its prior
sharing state. New work cannot race the drain. No model installation is removed.

The old signed app runs the helper, which waits for the old process to exit,
then starts the verified artifact. Windows assigns the new suspended process to
a dedicated job before resuming it, so recovery can terminate only that new
process tree. The new version must open its node server and receive a fresh
accepted gateway heartbeat before the active-release pointer is committed.
The embedded version must match the signed version. Existing shortcuts follow
that verified active pointer on later launches.

If startup/health fails, restore the backed-up enrollment config and relaunch the
previous local app with its prior paused/enabled state. Report `rolled_back` only
after that app reconnects. If recovery cannot finish, report `interrupted` for
inspection; do not blindly replay. Update claims are single-use. An ambiguous
in-flight update becomes `interrupted` after 20 minutes and requires inspection.

Private per-user app-updates files include a temporary delivery claim and config
backup and must remain under the same employee account. No prompt, response,
secret, token, raw upstream error or authentication response enters app activity
or update progress. The feature updates only the PROVIS node app; it has no
remote shell, arbitrary file-write, OS installer or remote approval capability.

## Activation acceptance still required

Use an actual approved publisher key, deploy the reviewed gateway, install the
trusted bootstrap, and verify owner sign-in plus one signed EXE update/readback.
Repeat on a real Linux desktop. Test a deliberately unhealthy signed candidate
on a disposable node to verify live recovery. Keep other employee identity tests
at the end of the wider Employee Assistant acceptance work.
