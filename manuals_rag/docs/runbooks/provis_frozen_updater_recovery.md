# Frozen node updater runtime lifetime

## Failure and correction

Dell's .4 updater failed before launching .5 with
`ModuleNotFoundError: cryptography.hazmat.bindings._rust`. Both archives contain
the Rust extension. The old launcher spawned the same one-file executable and
then exited. PyInstaller treated the helper as a worker sharing the parent's
extracted runtime; cleanup removed libraries before the helper imported them.

All independent app launches now set `PYINSTALLER_RESET_ENVIRONMENT=1` through
`independent_app_environment()`: update helper, candidate, rollback, and active
release redirect. The Windows candidate is still created suspended and assigned
to its containment job before it is resumed. Signature checks, claim handling,
receipt validation and HMAC replay protection remain required.

Reference: https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html

## Offline verification

Freeze `manuals_rag/scripts/check_frozen_updater_restart.py` using PyInstaller
with `manuals_rag/packages/inference_fabric/src` on `--paths` and
`--collect-all cryptography`. Run its source with
`--check <absolute path to frozen probe>` using the build Python. Run outside
the Windows app-container sandbox so its temporary-directory permissions do
not interfere with the parent-exit test.

The probe starts a frozen parent, lets it exit, then exercises Ed25519 in its
child after five seconds. It compares inherited and independent runtimes using
the production launcher environment function. It uses an ephemeral test key,
opens no listeners and reads no enrollment or publisher credentials.

The Windows reproduction on 2026-10-20 returned the exact missing `_rust`
module for the legacy launch and a successful signature round trip for the
corrected launch. The app's packaged `--self-test` also now requires a real
Ed25519 signature round trip, alongside the existing ONNX/OpenVINO checks.

## Recover an interrupted update

Use a verified administrative connection to the affected node. Do not enqueue
another update while the existing claim is `restarting`. Preserve its private
plan, signed release, config backup and logs; do not copy credentials into chat.

An old running launcher cannot be corrected by publishing a new payload alone:
its helper fails before it can launch that payload. Recovery therefore needs a
fresh independent launch from the affected PC, using a verified signed release
and the existing enrollment. Inspect the current processes and signed plan
before choosing rollback or resumption. Do not fabricate a healthy receipt,
change the active pointer without verification, or weaken signature checks.

Before resuming normal updates, install the corrected launcher through the
approved recovery path. Confirm the real Tailscale listener, heartbeat, signed
installed bytes, and bounded inference; then verify the queued update has a
valid terminal status. Preserve existing models and enrollment throughout.
