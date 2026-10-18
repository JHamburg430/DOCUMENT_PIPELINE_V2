import base64
import hashlib
import os
import threading
import uuid
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from manuals_rag_inference.installer_release import (
    ARTIFACT_TYPE,
    FILENAME,
    InstallerReleaseCatalog,
    installer_manifest,
    verify_installer_artifact,
    verify_installer_release,
    write_signed_installer_release,
)
from manuals_rag_inference.gateway import InferenceGateway, create_gateway_handler
from manuals_rag_inference.node_release import ReleaseCatalog, canonical, verify_release


@pytest.fixture
def keys():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private, {"fixture": base64.b64encode(public).decode("ascii")}


@pytest.fixture
def releases(tmp_path, keys):
    private, public_keys = keys
    content = b"standalone-installer-bytes"
    artifact = tmp_path / FILENAME
    artifact.write_bytes(content)
    manifest = installer_manifest(
        release_id=str(uuid.uuid4()), version_value="2026.10.18.1",
        size=len(content), sha256=hashlib.sha256(content).hexdigest(),
        app_release_id="39fd973c-c97a-4b54-97de-938a2f88bd6c",
        app_sha256="e635207ca0b1d13869523f7b679b0100ba05234222af487c9fadebaa94b914f2",
    )
    signature = private.sign(canonical(manifest))
    document = {
        "manifest": manifest,
        "publisher": "fixture",
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    app_manifest = {
        "format": 1, "release_id": manifest["app_release_id"], "version": "2026.10.17.10",
        "platform": "windows-x86_64", "filename": "PROVISInferenceNode.exe", "size": 1,
        "sha256": manifest["app_sha256"],
    }
    app_document = {
        "manifest": app_manifest, "publisher": "fixture",
        "signature": base64.b64encode(private.sign(canonical(app_manifest))).decode("ascii"),
    }
    output = tmp_path / "installer-catalog"
    written = write_signed_installer_release(
        artifact, output, publisher="fixture", signature=signature, manifest=manifest,
        app_release_document=app_document, public_keys=public_keys,
    )
    return document, manifest, artifact, output, public_keys, written, app_document


def test_installer_document_is_distinct_and_node_verifier_rejects_it(releases):
    document, manifest, artifact, output, public_keys, _, _ = releases
    assert manifest["artifact_type"] == ARTIFACT_TYPE
    assert manifest["filename"] == FILENAME
    assert verify_installer_release(document, public_keys) == manifest
    with pytest.raises(ValueError, match="Unsupported release manifest"):
        verify_release(document, public_keys)

    node_doc = {
        "manifest": {
            "format": 1,
            "release_id": str(uuid.uuid4()),
            "version": "2026.10.18.1",
            "platform": "windows-x86_64",
            "filename": "PROVISInferenceNode.exe",
            "size": 1,
            "sha256": "0" * 64,
        },
        "publisher": "fixture",
        "signature": "AA==",
    }
    with pytest.raises(ValueError, match="Unsupported installer manifest"):
        verify_installer_release(node_doc, public_keys)


def test_installer_artifact_hash_is_verified_and_tamper_rejected(releases):
    document, manifest, artifact, output, public_keys, _, _ = releases
    published = output / (manifest["release_id"] + ".artifact")
    verify_installer_artifact(published, manifest)
    published.write_bytes(b"tampered")
    with pytest.raises(PermissionError, match="size or type mismatch|hash mismatch"):
        verify_installer_artifact(published, manifest)


def test_installer_catalog_is_separate_and_not_node_inventory(releases, tmp_path):
    document, manifest, artifact, installer_root, public_keys, _, _ = releases
    installer_catalog = InstallerReleaseCatalog(installer_root, public_keys)
    fetched, installer_artifact = installer_catalog.get(manifest["release_id"])
    assert fetched == document
    assert installer_artifact.name == manifest["release_id"] + ".artifact"
    assert installer_catalog.validate_all() == 1

    # Even if an operator mistakenly points the node catalog at this root,
    # its app-only verifier fails closed instead of listing the installer.
    with pytest.raises(ValueError, match="Unsupported release manifest"):
        ReleaseCatalog(installer_root, public_keys).list()

    # The node catalog is pointed only at its own directory; it sees no installer IDs.
    node_root = tmp_path / "node-releases"
    node_root.mkdir()
    assert ReleaseCatalog(node_root, public_keys).list() == []


def test_manifest_binds_installer_to_exact_application_payload():
    manifest = installer_manifest(
        release_id=str(uuid.uuid4()), version_value="2026.10.18.1", size=5,
        sha256=hashlib.sha256(b"hello").hexdigest(),
        app_release_id="39fd973c-c97a-4b54-97de-938a2f88bd6c",
        app_sha256="e635207ca0b1d13869523f7b679b0100ba05234222af487c9fadebaa94b914f2",
    )
    assert manifest["app_release_id"] == "39fd973c-c97a-4b54-97de-938a2f88bd6c"
    assert manifest["app_sha256"] == "e635207ca0b1d13869523f7b679b0100ba05234222af487c9fadebaa94b914f2"


@pytest.mark.skipif(os.name == "nt", reason="This Windows sandbox denies loopback sockets; Linux CI covers routes")
def test_gateway_serves_installer_manifest_and_artifact_from_isolated_catalog(releases):
    document, manifest, _, output, public_keys, _, _ = releases
    gateway = InferenceGateway()
    gateway.installer_catalog = InstallerReleaseCatalog(output, public_keys)
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_gateway_handler(gateway))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1/installer-releases/{manifest['release_id']}"
    try:
        with urlopen(base + "/manifest.json") as response:
            assert response.headers.get_content_type() == "application/json"
            assert response.read() == __import__("json").dumps(
                document, sort_keys=True, separators=(",", ":")
            ).encode()
        with urlopen(base + "/artifact") as response:
            assert response.read() == (output / (manifest["release_id"] + ".artifact")).read_bytes()
        with pytest.raises(HTTPError) as missing:
            urlopen(base + "/metadata")
        assert missing.value.code == 404
        # The installer path is not wired through the owner-only node catalog.
        with pytest.raises(HTTPError) as node_missing:
            urlopen(f"http://127.0.0.1:{server.server_port}/v1/node-releases/{manifest['release_id']}")
        assert node_missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_writer_refuses_non_installer_filename_and_duplicate_id(releases):
    document, manifest, artifact, output, public_keys, _, app_document = releases
    renamed = artifact.with_name("PROVISInferenceNode.exe")
    renamed.write_bytes(artifact.read_bytes())
    with pytest.raises(ValueError, match="Standalone installer executable"):
        write_signed_installer_release(
            renamed, output, publisher="fixture",
            signature=base64.b64decode(document["signature"]), manifest=manifest,
            app_release_document=app_document, public_keys=public_keys,
        )
    with pytest.raises(FileExistsError, match="already exists"):
        write_signed_installer_release(
            artifact, output, publisher="fixture",
            signature=base64.b64decode(document["signature"]), manifest=manifest,
            app_release_document=app_document, public_keys=public_keys,
        )


def test_writer_rejects_signed_but_unrelated_app_binding(releases, keys, tmp_path):
    document, manifest, artifact, _, public_keys, _, app_document = releases
    private, _ = keys
    wrong_manifest = dict(manifest, app_sha256="0" * 64)
    wrong_signature = private.sign(canonical(wrong_manifest))
    with pytest.raises(PermissionError, match="exact trusted signed node application"):
        write_signed_installer_release(
            artifact, tmp_path / "installer-catalog", publisher="fixture",
            signature=wrong_signature, manifest=wrong_manifest,
            app_release_document=app_document, public_keys=public_keys,
        )
