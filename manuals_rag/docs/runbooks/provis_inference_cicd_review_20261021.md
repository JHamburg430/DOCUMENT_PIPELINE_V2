# PR #4 release readiness review — 2026-10-21

Reviewed PR #4 at `dd50c62509cfd37c966b3f0483c020e94c35401f`, against
integration base `b70537f96c928d32dd72dd28df5882a1f710c1e3`.
The PR was open/draft with no discussion/review comments at inspection.
Conclusion: retain both disabled workflow gates; this is not release acceptance.
No workflow was dispatched and no signing, publication, gateway deployment,
employee installation, credential or protection setting was changed.

## Reproduced deficiencies and corrections

1. **Clean CI dependencies fail Entra preflight (high).** The workflow installs
   `requirements-node-ci.txt` and ONNX dependencies, neither of which supplies
   PyJWT. In an isolated environment using the declared focused dependencies,
   `test_preflight_requires_restricted_entra_configuration` failed because
   `PyJWT[crypto]` was missing. Add pinned `PyJWT[crypto]==2.10.1` to the focused
   requirements. The existing six Entra tests then pass. This changes build/test
   dependencies, not employee authentication policy.
2. **Consolidation drops the Windows installer archive gate (medium).** The old
   workflow explicitly called `verify_provis_node_windows_installer_archive.py`;
   the replacement `ci --packaging` omitted it. The frozen GUI smoke does not
   independently prove all required modules/resources are present. Restore the
   verifier before the full-installer self-test. A command-orchestration test
   confirms this ordering; it does not claim native Windows execution.

Six additional fault-injection tests cover interruption after app/installer
artifact and metadata writes, a catalog stage receipt, and the final completion
receipt. Identical approved bytes reconcile while all existing files survive.
These use real temporary filesystem writes and fixture signatures; GitHub is a
test double. They do not establish live service or power-loss behavior.

## Safety and provenance assessment

- Both jobs remain hard-disabled. PR CI is unprivileged, uses hosted OS runners,
  read-only contents permission, pinned action SHAs and no stored checkout
  credentials. Publication is manual, guarded to main, uses a named protected
  environment and a separate trusted runner. The environment name alone does
  not establish protection; settings were not inspected/provisioned here.
- The signed envelope binds repository, canonical four-part version, tag, source
  SHA and exact manifest/artifact hashes. All three signed manifests must share
  version and publisher; external public pins establish trust. This is signer
  attestation of source provenance, not a build attestation proving that the
  executable was compiled from that SHA or passed its recorded checks.
- Windows installer manifest UUID/hash and embedded binding document must match
  the Windows app document; Windows/Linux app config must agree on HTTPS gateway,
  updater version and trusted pins. Staging rewrites installer constants in a
  temporary copy and preserves source defaults.
- **Binding assurance limit:** publication examines a JSON resource, not the
  executable installer module's constants/control flow. The installer runtime
  does not consume `provis-installer-binding.json`. A correct marker is not
  independent proof of behavior. The trusted builder/source review and native
  tests must establish that the actual installer installs the approved app.
  Do not describe offline verification as executing or proving that behavior.
- Bundle/artifact reads are bounded; file symlinks/hardlinks are rejected;
  verification reads private archive snapshots without running candidate code.
  GitHub asset readback strips publisher credentials at the allowed HTTPS
  GitHubusercontent redirect. Trust still depends on protected build hosts,
  approved source, publisher access and immutable destination controls.

## Publication and recovery assessment

The private ledger lock serializes cooperating publishers. Preflight rejects
catalog capacity conflicts, reused version/platform identities, changed bytes
and overlapping roots. Publication claims the version, identifies a digest-bound
GitHub draft, checks draft assets, exposes app artifacts before their metadata,
then installer artifacts/metadata, reads catalogs back and finalizes GitHub.
Exact-byte reconciliation is supported; conflicting bytes are never overwritten.
Tag/source is checked again around GitHub finalization.

There is no release-wide visibility barrier: an app catalog entry can be read
before the installer catalog or GitHub finalization completes. Readers do not
consult the completion receipt. Operators must accept this ordering or design
and validate a coordinated promotion mechanism before requiring atomic visibility.
Do not infer that publication itself authorizes node updates.

The new interruption tests cover durable-write boundaries. Existing tests cover
GitHub asset/finalization failures and corrupt destinations. A crash between
POSIX linking and removal of the temporary alias can leave a two-link file;
the regular-file guard then rejects it. Disk failures/corrupt partial files,
foreign writers, changed tags/assets, and capacity exhaustion are operator
reconciliation cases, not guaranteed automatic resume. Preserve all evidence;
do not delete claims/assets or rerun with rebuilt bytes to make publication pass.
There is no withdrawal/quarantine executor or automatic distributed rollback.
Signed corrected software requires a new version and approval.

## Required before enabling CI

1. Review/integrate the dependency and restored archive checks. Preserve the
   historical 228-pass/one-skip report as prior evidence, not a hosted run.
2. Approve hosted Windows/Linux minutes and evidence retention, then use a
   separate reviewed activation change for CI only. Publisher activation remains
   separately blocked. Confirm exact pinned action revisions and supported
   Python/Tk/build dependencies in those actual images.
3. Run both matching-OS `ci --packaging` jobs and retain tested SHA, dependency
   inventory, commands, JUnit and frozen-artifact hashes. The installer-catalog
   route test passed here on Linux (7 installer-release tests passed); its
   Windows-only skip is explicitly delegated to Linux coverage. Confirm that
   complementary OS coverage in hosted CI rather than counting skips as passes.

## Required before adopting publication

| Requirement | Kind / acceptance evidence |
| --- | --- |
| Reviewed integration into main, required checks/reviews and protected publication code | Maintainer decision/setup; trusted-main release code must actually exist |
| Windows clean install, repair/update, shortcut/start-menu path, preserved enrollment/config and real post-install startup | Technical; native tests on exact proposed source and final signed candidate, including the previously failing startup scenario |
| Linux native package, desktop/tray, listener/reconnect and signed update | Technical; live Linux acceptance, not Windows or archive fixtures |
| Unhealthy signed update rollback and transient final-status reporting | Technical; disposable worker evidence, no production rollout implied |
| Correct gateway revision/catalog reader supporting native Linux filename | Technical compatibility validation and separately approved gateway rollout if needed; preserve deployed newer code/permissions |
| Existing DPAPI-protected signing operator supports all manifests and envelope | Operator approval and technical signing/verification drill; no private key migration |
| Trusted publisher host/runner and pinned verifier environment, capacity and mount behavior | Infrastructure approval plus isolated validation; never run PR code there |
| External trusted public pins, approved HTTPS endpoint, immutable candidate staging | Operator decision plus exact-byte/config validation |
| Repository-scoped release token, protected environment reviewers/main restriction, protected tags/immutable Releases where available | Security/infrastructure approval and verified settings; read-only workflow token stays read-only |
| App and installer catalogs actually served by the gateway; private ledger; gateway identity can read new 0600 files | Infrastructure approval and real identity/readback check; no implicit chmod/chown |
| Partial visibility, recovery ownership, alerting and withdrawal/quarantine procedure | Operating decision and nonproduction fault/recovery drill |
| One source/tag/version/digest, destination and release approval | Explicit per-release human approval after technical evidence; never rebuild approved bytes |

Only after these gates should a separate reviewed publication activation remove
its hold. Publication acceptance does not authorize PC installs, model preparation,
gateway restarts, auth changes or log cleanup. Repository architecture/evaluation
requirements remain applicable to their domains.

## Validation in this review

Environment: isolated Python 3.12 Linux venv, focused pinned requirements. This
differs from the hosted Python 3.11 matrix. Original publication contract suite:
46 passed. Corrected publication contract suite: 54 passed. Entra suite after
the dependency correction: 6 passed. Installer-release suite: 7 passed, including
real loopback HTTP manifest/artifact reads and isolated route checks.

The supported baseline command was attempted: 221 passed, one skipped, one failed
(missing PyJWT), six GUI setup errors (no DISPLAY). This host has no Xvfb, so
GUI/native packaging acceptance is unavailable. The prior 228/one-skip result is
not independently established by this environment. Final headless selection:
230 passed, one Windows process-containment test skipped, six GUI cases
deselected. The six exclusions are the two light/dark owner-panel cases, the
refresh case, and three compatible/confirmed selection cases. JUnit was retained
at `/tmp/provis-review-headless.xml` during execution. No excluded GUI test is
treated as passing. Native Windows, native Linux packaging, final signed
candidate startup, live signing and publication remain unrun.

Sources: [PR #4](https://github.com/JHamburg430/DOCUMENT_PIPELINE_V2/pull/4),
`provis_inference_cicd.md`, `provis_node_remote_updates.md`,
`provis_node_agent_setup_handoff_20261017.md`, the two workflows, pipeline,
installer builder/runtime/archive verifier and signed catalog implementations.
