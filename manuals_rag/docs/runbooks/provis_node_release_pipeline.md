# PROVIS Inference Node release pipeline

## Automated checks

The `PROVIS Inference Node CI` workflow runs on relevant pull requests and
pushes. It checks the signed updater contract, gateway update routes, listener
fallback and authenticated endpoint refresh, Linux platform handling, and
owner update-panel discovery/selection behavior. It then builds an
**unsigned** Linux x86_64 smoke artifact and records its SHA256 as a short-lived
Actions artifact.

CI has `contents: read` permission. It does not publish a release, change a
gateway catalog, create credentials, or hold the publisher private key. The
unsigned CI artifact is only evidence that the source builds; it must never be
installed or added to the release catalog. It embeds the reserved `.invalid`
fixture gateway URL, so it cannot connect to the production service.

## Protected signing and publication

Use the existing protected operator signing process after a maintainer reviews
the CI result and selects the intended release candidate. Build the employee
artifact for the target platform with the managed public update configuration,
then sign the exact artifact with the existing protected Ed25519 publisher
using `scripts/publish_inference_node_release.py`. Keep the private key outside
the checkout and CI. Do not upload it, its passphrase, or a signed catalog to
Actions.

Example operator commands (run from the repository root with an approved
public-only updater configuration and the protected signing key path supplied
by the signing operator):

```sh
export PYTHONPATH=manuals_rag/packages/inference_fabric/src:manuals_rag/scripts
python manuals_rag/scripts/build_inference_node_linux.py \
  --gateway-url https://<approved-gateway-host>:8445 \
  --update-config /secure/public-updater-config.json \
  --output /release/PROVISInferenceNode
python manuals_rag/scripts/publish_inference_node_release.py \
  --artifact /release/PROVISInferenceNode \
  --output /release/catalog \
  --private-key /protected/provis-release-signing.pem \
  --publisher provis-release-1 \
  --version <reviewed-version> \
  --platform linux-x86_64
```

The signing script verifies the exact artifact and writes its signed manifest
and bytes into the operator-selected catalog. Only after separate deployment
approval may the signed pair be copied to the gateway's configured release
catalog. Windows builds use the existing bundle's Windows PyInstaller helper;
the same signing script validates the Windows artifact and platform. Keep all
private signing material on the approved protected signing system.

## Update button and acceptance limits

The owner UI polls the signed gateway update-node, release, and status routes.
It submits an update only after one PC and one release are selected, the PC has
a trusted updater and matching platform, and the owner confirms the action.
Isolated UI tests cover discovery requests, compatible confirmed submission,
cancellation, and incompatible-platform rejection. These checks do not prove
that a newly signed release was deployed or installed on employee PCs.

Before a live release is accepted, verify signature/hash rejection, failed
startup rollback, transient network retry, same-enrolled-host endpoint update,
and a healthy post-restart heartbeat on each supported OS. Record each
installed/running version and hash. Linux desktop runtime and live signed
release acceptance remain separate operational checks. The detailed existing
operator flow is in [the remote update runbook](provis_node_remote_updates.md).

## Separate Windows installer releases

Standalone installers use `installer_release.py`, never
`publish_inference_node_release.py`. Their exact signed manifest has the
`artifact_type` discriminator `provis-node-installer`, fixed filename
`PROVISInferenceNodeInstaller.exe`, Windows platform, version, release UUID,
size and SHA-256, plus the exact node application release UUID and SHA-256 the
installer will deploy. The signature is over the canonical full manifest. The
protected publisher must verify the bound app release with the node-app schema
and existing trusted key before signing the installer manifest; the caller
supplies a signature from the existing protected publisher to
`write_signed_installer_release`. That API accepts no private-key path and
verifies the returned signature with the public key before writing.

Installers are stored in an isolated directory configured through
`INFERENCE_INSTALLER_RELEASE_DIR`, distinct from
`INFERENCE_NODE_UPDATE_RELEASE_DIR`. The gateway exposes only per-ID retrieval
at `/v1/installer-releases/{id}/manifest.json` and
`/v1/installer-releases/{id}/artifact`; it provides no installer listing or
update action. Do not copy installer files into the node release directory or
add them to node inventory. Node `ReleaseCatalog` rejects this schema.

The custom Ed25519 manifest authenticates the installer artifact to a verifier;
it is not a Windows Authenticode signature and does not change SmartScreen or
Defender reputation. Before distributing to a normal Windows user, the operator
must verify the installer through an already trusted pre-launch verifier and
confirm the existing Windows trust route does not require bypassing a warning.
If no such approved route exists, do not launch the artifact or create new
signing credentials as a workaround.
