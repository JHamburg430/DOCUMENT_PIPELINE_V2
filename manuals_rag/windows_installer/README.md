# PROVIS Inference Node Windows installer

This is a standalone, per-user Windows installer with a graphical progress
window. It includes its Python runtime and does not require Codex, Python, a
source checkout, developer tools, or environment variables. After the installer
starts, it downloads the pinned `2026.10.17.10` Windows app from the existing
node catalog, verifies the embedded Ed25519 release document and artifact
SHA-256, then installs it under the current Windows user's
`Programs\PROVIS\2026.10.17.10` directory.

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

The installer itself is distributed through a separate signed manifest type
(`provis-node-installer`), a separate gateway catalog directory, and the
`/v1/installer-releases/{id}/manifest.json` and `/artifact` endpoints. Installer
entries are never returned from node update inventory. The separate Ed25519
manifest authenticates installer bytes to an independent verifier; it does not
create a Windows Authenticode signature. Do not bypass Windows or browser
security warnings. A normal interactive Windows session is still required to
verify desktop launch, OS trust prompts, and the app UI end to end.

The repository CI runs cross-type signature/catalog tests, installer behavior
tests, a standalone Windows PyInstaller smoke build, and a non-mutating frozen
executable self-test. Tests do not verify Windows SmartScreen, Authenticode trust,
real Desktop/Start Menu COM shortcuts, or installation under an actual standard
user profile.
