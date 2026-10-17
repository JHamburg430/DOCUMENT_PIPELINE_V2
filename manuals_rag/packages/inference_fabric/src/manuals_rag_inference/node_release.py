"""Signed, platform-specific node releases. No executable command in a manifest."""
import base64
import hashlib
import json
import re
import uuid
from pathlib import Path

DEFAULT_VERSION = '2026.10.17.7'
MAX_ARTIFACT = 256 * 1024 * 1024
FILENAMES = {'windows-x86_64': 'PROVISInferenceNode.exe', 'linux-x86_64': 'PROVISInferenceNode', 'linux-aarch64': 'PROVISInferenceNode'}
LEGACY_LINUX_FILENAME = 'inference-node.pyz'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r'(0|[1-9][0-9]{0,7})(\.(0|[1-9][0-9]{0,7})){2,3}', value):
        raise ValueError('Numeric release version required')
    parts = tuple(int(part) for part in value.split('.'))
    return parts + (0,) * (4-len(parts))


def verify_release(document, keys, *, platform=None, current_version=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if type(document) is not dict or set(document) != {'manifest', 'publisher', 'signature'}:
        raise ValueError('Exact signed release required')
    manifest = document['manifest']
    if type(manifest) is not dict or set(manifest) != {'format', 'release_id', 'version', 'platform', 'filename', 'size', 'sha256'}:
        raise ValueError('Unsupported release manifest')
    if type(manifest['format']) is not int or manifest['format'] != 1:
        raise ValueError('Unsupported release format')
    if str(uuid.UUID(manifest['release_id'])) != manifest['release_id']:
        raise ValueError('Canonical release ID required')
    allowed = {FILENAMES.get(manifest['platform'])}
    if manifest['platform'] in ('linux-x86_64', 'linux-aarch64'):
        allowed.add(LEGACY_LINUX_FILENAME)
    if manifest['platform'] not in FILENAMES or manifest['filename'] not in allowed:
        raise ValueError('Fixed node app artifact required')
    version(manifest['version'])
    if type(manifest['size']) is not int or not 1 <= manifest['size'] <= MAX_ARTIFACT:
        raise ValueError('Bounded release artifact required')
    if not isinstance(manifest['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', manifest['sha256']):
        raise ValueError('SHA256 required')
    publisher = document['publisher']
    if not isinstance(publisher, str) or publisher not in keys:
        raise PermissionError('Release publisher is not trusted')
    try:
        public = base64.b64decode(keys[publisher], validate=True)
        signature = base64.b64decode(document['signature'], validate=True)
        Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical(manifest))
    except Exception:
        raise PermissionError('Release signature invalid') from None
    if platform is not None and manifest['platform'] != platform:
        raise PermissionError('Release platform mismatch')
    if current_version is not None and version(manifest['version']) <= version(current_version):
        raise PermissionError('Release must be newer than the installed app')
    return dict(manifest)


def verify_artifact(path, manifest):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != manifest['size']:
        raise PermissionError('Release artifact size or type mismatch')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''): digest.update(chunk)
    if digest.hexdigest() != manifest['sha256']:
        raise PermissionError('Release artifact hash mismatch')


class ReleaseCatalog:
    def __init__(self, root, keys):
        self.root = Path(root).resolve(strict=True)
        self.keys = dict(keys)

    def get(self, identifier):
        if str(uuid.UUID(identifier)) != identifier: raise ValueError('Canonical release ID required')
        metadata = self.root / (identifier + '.json')
        if metadata.is_symlink() or metadata.stat().st_size > 16384:
            raise PermissionError('Registered release metadata required')
        document = json.loads(metadata.read_text(encoding='utf-8'))
        manifest = verify_release(document, self.keys)
        if manifest['release_id'] != identifier: raise PermissionError('Release identity mismatch')
        artifact = self.root / (identifier + '.artifact')
        verify_artifact(artifact, manifest)
        return document, artifact

    def list(self):
        identifiers = sorted(path.stem for path in self.root.glob('*.json'))
        if len(identifiers) > 64: raise ValueError('Release catalog is too large')
        return [self.get(identifier)[0] for identifier in identifiers]
