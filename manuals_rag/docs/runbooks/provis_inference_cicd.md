# PROVIS inference CI/CD — supported path

Read [AGENTS.md](../../../AGENTS.md), [CONTRIBUTING.md](../../../CONTRIBUTING.md)
and the immutable `.AGENT.md`. Use `manuals_rag/scripts/provis_release_pipeline.py`
for checks, verification and explicitly approved dual publication. Existing
builders/signing helpers implement this flow; they are not alternate authority.

## Bootstrap is pending

Both workflows are **hard-disabled (`if: false`)**. No CI budget, runner, token,
environment, security setting or product release is provisioned. Skipped checks
are not passes. This branch targets `codex/provis-production-updates-20261017`;
`main` does not contain the inference feature yet. Instructions do not reach the
default branch/every PC until reviewed integration. JSON-formatted YAML lets the
stdlib validate workflow structures; official actions use full resolved SHAs.

Before a separate activation PR, obtain explicit approval for:

- Hosted Windows/Linux CI minutes and evidence storage/retention costs.
- An existing trusted operator or dedicated `provis-trusted-publisher` Linux
  runner with only the approved app catalog, separate installer catalog and
  private ledger writable. Never run PR code there or provision a runner here.
- A repository-scoped GitHub release credential (necessary Contents write only).
  Workflow `GITHUB_TOKEN` remains read-only. No credential is provided by this PR.
- Human reviewers and trusted-main restrictions for environment
  `provis-release-publication`; naming it in YAML does not configure protection.
- Required CI/reviews, protected publication code, immutable tags and GitHub
  immutable Releases where available. No settings are changed. Prose/scripts
  cannot stop an administrator or another credential modifying a sink.
- Existing protected publisher approval/support for the provenance envelope
  below. Never put its private key into CI, the repo or publisher runner. Public
  pins must be an external trusted file, never supplied by a candidate/PR author.

Manual publication must execute trusted `main` only. Never add privileged
`pull_request_target`/`workflow_run` handoffs, arbitrary-ref checkout, PR artifact
auto-promotion into a secret job or publishing on ordinary pushes/tags.

## Checks and packaging

```sh
python manuals_rag/scripts/provis_release_pipeline.py ci --evidence-dir /tmp/provis-ci
# Matching approved OS/build environment only:
python manuals_rag/scripts/provis_release_pipeline.py ci --packaging --evidence-dir /tmp/provis-package
```

Use pinned `requirements-node-ci.txt` and `requirements-inference-onnx.txt` in an
approved isolated environment. Checks cover gateway/updater, replay/auth,
endpoint refresh, model preparation, GPU placement and publication contracts.
JUnit and append-only CI receipts record tested SHA, dirty-tree status, steps
and packaging hashes. Domain/retrieval evaluations in `.AGENT.md` still apply.

The unprivileged Ubuntu22.04/Windows2022 matrix checks native Linux packaging
under Tk/Xvfb; Windows checks full-installer units, isolated-profile native-app
startup and the frozen full-installer self-test. Frozen app tests cover existing
Ed25519/ONNX/OpenVINO CPU packaging. Intel drivers/GPU behavior remains a live
check. Fixture keys, temporary profiles and `.invalid` gateway smoke artifacts
are never employee releases. PR jobs have no signing/catalog credentials.

This contribution uses `[skip ci]` to preserve the cost hold; GitHub documents
[skip behavior](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs).
Skipped required checks may remain pending; this is not a security barrier or a
way to satisfy a release check. Local/static tests are not hosted CI acceptance.

## Immutable release and signed provenance

Choose a new canonical four-part version, e.g. `2026.10.22.1`, and tag
`provis-node/<version>` bound to one full lowercase40-character source SHA.
Create/approve that tag through the trusted flow; publication never creates or
moves tags. A version cannot be reused for changed bytes.

1. Verify clean reviewed source and CI/build receipts. Build native Windows and
   Linux x86_64 apps once with existing builders, approved HTTPS gateway URL and
   public-only updater configuration carrying the exact version.
2. Obtain explicit protected-operator approval/signatures for exact app bytes.
   `publish_inference_node_release.py` remains a low-level signing helper.
3. Build/sign the Windows full installer once. Its separate existing schema must
   bind the exact Windows app UUID/hash; all three manifests use the same version.
   Linux has a native app channel, not a separate full-installer schema.
4. Stage distinct manifest/artifact basenames and have the existing protected
   operator sign the index below using `node_release.canonical(index)` and the
   existing Ed25519 publisher. Preserve source/build/config hashes, frozen tests
   and human approval. This is operator-attested provenance, not reproducible
   PyInstaller or hardware-backed build attestation.

The full-installer release build must pass the approved Windows app document,
external public pins and managed URL explicitly:

```sh
python manuals_rag/scripts/build_provis_node_windows_installer.py \
  --output /approved/candidate/PROVISInferenceNodeInstaller.exe \
  --app-release /approved/candidate/windows-app.json \
  --public-keys /trusted/provis-public-keys.json --gateway-url https://APPROVED_GATEWAY
```

This stages public constants in a temporary installer-source copy; checked-out
runtime defaults remain unchanged. It embeds `provis-installer-binding.json`
and verifies that resource after freezing. Default historical bindings remain
available only for smoke checks; they are not a future release build recipe.
Publication reads bounded data members from exact snapshotted archives without
executing them: app config version/trusted pins/managed URL must match, both apps
must use the same non-`.invalid` HTTPS gateway, and the installer's embedded app
document and gateway must match the approved Windows app. Outer-manifest binding
alone is insufficient. Existing pinned PyInstaller is a reader dependency on the
trusted verifier as well as the build hosts; no runtime dependency is installed
by publication.

`approved-release.json` has exactly `manifest`, `publisher`, `signature` (base64).
The manifest has exactly `schema` (integer1), `repository`
(`JHamburg430/DOCUMENT_PIPELINE_V2`), `version`, `tag`, `source_commit`, and
`artifacts`. The artifacts list has exactly one descriptor for each slot:

| Slot | Signed manifest | Artifact example |
| --- | --- | --- |
| windows-app | windows-app.json | PROVISInferenceNode.exe |
| linux-app | linux-app.json | PROVISInferenceNode-linux |
| windows-installer | windows-installer.json | PROVISInferenceNodeInstaller.exe |

Each descriptor has exactly `slot`, `manifest`, `artifact`, `manifest_sha256`,
`artifact_sha256`; paths are plain basenames and hashes are of exact raw files.
Runtime app/installer schemas are unchanged. The separate signed envelope binds
source without adding unsupported runtime fields. `approved-release.json` and
generated `SHA256SUMS` are reserved asset names. Do not rebuild after approval.

## Verify and publish the same approved bytes

```sh
python manuals_rag/scripts/provis_release_pipeline.py verify \
  --bundle /approved/releases/2026.10.22.1 --public-keys /trusted/provis-public-keys.json

# Only after exact source/tag/artifact/destination approval and bootstrap:
python manuals_rag/scripts/provis_release_pipeline.py publish --execute \
  --bundle /approved/releases/2026.10.22.1 --public-keys /trusted/provis-public-keys.json \
  --approval-sha256 APPROVED_ENVELOPE_SHA256 \
  --app-catalog /approved/app-catalog --installer-catalog /approved/installer-catalog \
  --ledger /approved/publication-ledger
```

Paths are placeholders, not permission to provision mounts/access. Supply the
already-approved `PROVIS_GITHUB_PUBLISH_TOKEN` through the operator environment,
never command text/receipts. This CLI has no SSH or gateway-deployment command.
The manual workflow accepts only version/digest and uses a pre-staged signed
bundle, not a PR artifact. Required existing absolute-path variables are
`PROVIS_APPROVED_BUNDLES`, `PROVIS_RELEASE_PUBLIC_KEYS`, `PROVIS_APP_CATALOG`,
`PROVIS_INSTALLER_CATALOG`, `PROVIS_RELEASE_LEDGER`, `PROVIS_PUBLISH_PYTHON`, plus
environment-secret `PROVIS_RELEASE_GITHUB_TOKEN`. These are declared, not created.

The CLI revalidates/snapshots signed bytes before writing; budget memory for up
to three256MiB artifacts. It checks capacity/duplicate versions and locks the
private ledger. Catalogs/ledger must be separate and not concurrently writable
by operators bypassing that lock. It resolves exact `refs/tags/` including
bounded annotated-tag peeling, and checks source before/after publication.

Bootstrap must confirm these are the directories actually served by the gateway,
and that its existing reader identity can read publisher-owned0600 outputs.
Otherwise stop; do not chmod/chown or grant catalog access as an implicit fix.
Verification uses private temporary archive snapshots, never writes a bundle or
catalog, and never runs frozen code. File visibility uses an exclusive atomic
link on POSIX or non-replacing Windows rename. Incomplete hidden staging files
remain preserved after failure; only the successful operation's own staging
alias is removed. Neither this nor temporary verification cleanup touches logs.

One GitHub draft is tied to the approved-envelope digest. Only missing assets
are added; every downloaded asset and generated checksum file is verified.
One HTTPS GitHubusercontent download redirect is allowed without forwarding the
publisher token. The same artifact/manifest bytes are copied into separate PROVIS
catalogs, artifact before metadata visibility. GitHub is finalized only after
catalog readback. Never overwrite/delete signed assets, versions or IDs.

## Partial publication and rollback

There is no atomic transaction across GitHub and two catalogs. Immutable ledger
claims, stage receipts and final `complete-<version>.json` record progress.
Receipts do not prove another administrator has not subsequently changed a sink.

On timeout/exit preserve all drafts/files/receipts, run offline verification,
inspect source/tag/catalogs/ledger, then rerun the identical approved command.
Exact existing bytes reconcile and missing items are added. Different claims,
signatures, hashes/tags, unexpected assets, corrupt partial files or capacity
limits stop the operation. Never delete an asset/reset a claim to make it pass.

Pause a bad release. Preserve prior versions and use existing per-node startup
rollback where applicable. Corrected public artifacts require a new version and
approval. Withdrawal/quarantine needs a separate reviewed operator plan; this
tool has no destructive rollback/history-cleanup executor. Publication does not
install, enqueue updates, prepare models, restart services, change auth/replay,
delete logs or release review holds. Installer Ed25519 signatures are not
Authenticode and never authorize bypassing SmartScreen/Defender. See
[remote updates](provis_node_remote_updates.md).

Troubleshoot disabled CI as a bootstrap hold;401/403 as an approved access issue,
not permission to broaden scopes; hash/binding failure using original bytes and
trusted pins, not new keys; and conflicts with evidence-preserving reconciliation
or a new approved version. Report missing platform/live checks honestly.

Official references: [workflow permissions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax),
[protected environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments),
[draft/immutable Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
