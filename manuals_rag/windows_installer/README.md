# PROVIS Inference Node Windows installer

This is a separate, per-user bootstrapper pinned to the already published
`2026.10.17.10` Windows release. It downloads the artifact from the existing
gateway catalog, verifies the embedded Ed25519 release document against the
existing public key, checks the artifact size and SHA-256, then installs it
under the Windows user's `Programs\PROVIS\2026.10.17.10` directory.

The installer uses `SHGetKnownFolderPath` for Local AppData, Desktop, and the
user Start Menu. It never derives install paths from `LOCALAPPDATA`, `APPDATA`,
`USERPROFILE`, or the working directory. It rejects known Codex sandbox folder
roots so it cannot persist a sandbox-specific path. The normal shortcuts target
the verified versioned executable and are replaced atomically; the previous
default shortcut is preserved as a rollback link, and other version-named
shortcuts are left untouched. Destination paths are checked for symlinks and
Windows reparse points before writes. The executable and
per-user enrollment/UI configuration are kept side by side in their established
locations. Installer uninstall removes only a signed version with an installer
receipt and shortcuts that target that exact release; it preserves node
configuration and all unreceipted versions.

The bootstrapper is a local install/repair tool, not the node's managed update
client. It does not call `/v1/nodes/updates/*`, submit a plan, change gateway
authorization, or start/restart/terminate the node. Subsequent managed updates
remain subject to the existing owner/server authorization and idle-drain flow.
The artifact endpoint must serve the pinned release; a missing or denied
artifact causes installation to stop without fallback.

## Local build

On a Windows build host with the repository's `cryptography` and PyInstaller
dependencies installed:

```powershell
py -3 manuals_rag/scripts/build_provis_node_windows_installer.py `
  --output C:\path\to\local\PROVISInferenceNodeInstaller.exe
```

The output is unsigned and local. This build step neither signs nor publishes
an installer. Any distribution requires a separate review of installer
publisher trust, signing, and rollout approval. Do not run an unsigned
candidate through a Windows warning or policy prompt by bypassing it.

The source-level installer tests use a local fake catalog response and temporary
folders. The repository CI workflow runs these tests and an unsigned Windows
PyInstaller smoke build on a Windows runner. They verify signature/hash rejection
before install, reparse-point rejection, rollback-folder preservation, config
preservation, shortcut ownership, transactional shortcut repair, and uninstall
behavior. Only a normal interactive Windows session can verify the installed
shortcuts and app UI end to end.
