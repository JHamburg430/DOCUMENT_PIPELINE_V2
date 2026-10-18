"""Separate, signed catalog for standalone node installers.

Installer artifacts deliberately have a different manifest discriminator,
directory, and HTTP namespace from the node update release catalog.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from pathlib import Path

from .node_release import MAX_ARTIFACT, canonical, verify_release, version

ARTIFACT_TYPE = "provis-node-installer"
PLATFORM = "windows-x86_64"
FILENAME = "PROVISInferenceNodeInstaller.exe"
MANIFEST_FIELDS = {
    "format", "artifact_type", "release_id", "version", "platform",
    "filename", "size", "sha256", "app_release_id", "app_sha256",
}


def verify_installer_release(document, keys, *, platform=PLATFORM):
    """Verify the exact installer schema and signature with a trusted publisher key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if type(document) is not dict or set(document) != {"manifest", "publisher", "signature"}:
        raise ValueError("Exact signed installer release required")
    manifest = document["manifest"]
    if type(manifest) is not dict or set(manifest) != MANIFEST_FIELDS:
        raise ValueError("Unsupported installer manifest")
    if type(manifest["format"]) is not int or manifest["format"] != 1:
        raise ValueError("Unsupported installer manifest version")
    if manifest["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("Installer artifact type required")
    if str(uuid.UUID(manifest["release_id"])) != manifest["release_id"]:
        raise ValueError("Canonical installer release ID required")
    if manifest["platform"] != PLATFORM or manifest["filename"] != FILENAME:
        raise ValueError("Fixed Windows installer artifact required")
    if platform is not None and manifest["platform"] != platform:
        raise PermissionError("Installer platform mismatch")
    version(manifest["version"])
    if type(manifest["size"]) is not int or not 1 <= manifest["size"] <= MAX_ARTIFACT:
        raise ValueError("Bounded installer artifact required")
    for field in ("sha256", "app_sha256"):
        if not isinstance(manifest[field], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[field]):
            raise ValueError("SHA256 required")
    if str(uuid.UUID(manifest["app_release_id"])) != manifest["app_release_id"]:
        raise ValueError("Canonical application release ID required")
    publisher = document["publisher"]
    if not isinstance(publisher, str) or publisher not in keys:
        raise PermissionError("Installer publisher is not trusted")
    try:
        public = base64.b64decode(keys[publisher], validate=True)
        signature = base64.b64decode(document["signature"], validate=True)
        Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical(manifest))
    except Exception:
        raise PermissionError("Installer release signature invalid") from None
    return dict(manifest)


def verify_installer_artifact(path, manifest):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != manifest["size"]:
        raise PermissionError("Installer artifact size or type mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise PermissionError("Installer artifact hash mismatch")


class InstallerReleaseCatalog:
    """Read-only catalog rooted separately from node update inventory."""

    def __init__(self, root, keys):
        self.root = Path(root).resolve(strict=True)
        self.keys = dict(keys)

    def get(self, identifier):
        if str(uuid.UUID(identifier)) != identifier:
            raise ValueError("Canonical installer release ID required")
        metadata = self.root / (identifier + ".json")
        if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 16384:
            raise PermissionError("Registered installer metadata required")
        document = json.loads(metadata.read_text(encoding="utf-8"))
        manifest = verify_installer_release(document, self.keys)
        if manifest["release_id"] != identifier:
            raise PermissionError("Installer release identity mismatch")
        artifact = self.root / (identifier + ".artifact")
        verify_installer_artifact(artifact, manifest)
        return document, artifact

    def validate_all(self):
        """Validate the isolated catalog at startup; intentionally no public listing."""
        entries = list(self.root.glob("*.json"))
        if len(entries) > 64:
            raise ValueError("Installer release catalog is too large")
        for entry in entries:
            self.get(entry.stem)
        return len(entries)


def installer_manifest(*, release_id, version_value, size, sha256, app_release_id, app_sha256):
    """Build canonical manifest data; a protected caller supplies the signature."""
    manifest = {
        "format": 1,
        "artifact_type": ARTIFACT_TYPE,
        "release_id": release_id,
        "version": version_value,
        "platform": PLATFORM,
        "filename": FILENAME,
        "size": size,
        "sha256": sha256,
        "app_release_id": app_release_id,
        "app_sha256": app_sha256,
    }
    # Reuse the verifier's shape and bounds checks; the dummy signature is not accepted.
    if str(uuid.UUID(release_id)) != release_id or str(uuid.UUID(app_release_id)) != app_release_id:
        raise ValueError("Canonical release IDs required")
    version(version_value)
    if type(size) is not int or not 1 <= size <= MAX_ARTIFACT:
        raise ValueError("Bounded installer artifact required")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256) or not re.fullmatch(r"[0-9a-f]{64}", app_sha256):
        raise ValueError("SHA256 required")
    return manifest


def write_signed_installer_release(artifact, output, *, publisher, signature, manifest,
                                  app_release_document, public_keys):
    """Write a signed pair after a protected signer supplies signature bytes.

    This module does not read or load private keys. The caller must be the
    approved signer integration, which signs ``canonical(manifest)``.
    """
    artifact = Path(artifact)
    if artifact.name != FILENAME or not artifact.is_file() or artifact.is_symlink():
        raise ValueError("Standalone installer executable required")
    actual_size = artifact.stat().st_size
    actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if actual_size != manifest.get("size") or actual_hash != manifest.get("sha256"):
        raise PermissionError("Installer bytes differ from the signed manifest")
    document = {
        "manifest": dict(manifest),
        "publisher": publisher,
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    verify_installer_release(document, public_keys)
    app_manifest = verify_release(app_release_document, public_keys, platform=PLATFORM)
    if (app_release_document["publisher"] != publisher
            or app_manifest["release_id"] != manifest["app_release_id"]
            or app_manifest["sha256"] != manifest["app_sha256"]):
        raise PermissionError("Installer must bind to the exact trusted signed node application release")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    release_id = manifest["release_id"]
    artifact_path = output / (release_id + ".artifact")
    metadata_path = output / (release_id + ".json")
    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError("Installer release ID already exists")
    artifact_path.write_bytes(artifact.read_bytes())
    try:
        metadata_path.write_text(json.dumps(document, sort_keys=True, indent=2), encoding="utf-8")
    except Exception:
        artifact_path.unlink(missing_ok=True)
        raise
    return document
