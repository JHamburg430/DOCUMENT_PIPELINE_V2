"""Explicitly owned dedicated-server sessions. No desktop thread discovery/import."""
import json
import os
from pathlib import Path
import uuid

def root():
    directory = Path(os.environ['LOCALAPPDATA']) / 'ProVIS' if os.name == 'nt' else Path.home() / '.config/ProVIS'
    if directory.is_symlink():
        raise PermissionError('Private owner settings directory required')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory

def server_mode():
    path = root() / 'codex-owner-settings.json'
    if not path.exists():
        return 'proxy'
    if path.is_symlink() or path.stat().st_size > 4096:
        raise PermissionError('Bounded private Codex settings required')
    value = json.loads(path.read_text()).get('server_mode')
    if value not in ('proxy', 'dedicated'):
        raise ValueError('Explicit Codex server mode required')
    return value

def owned_sessions():
    path = root() / 'codex-owner-sessions.json'
    if not path.exists():
        return {}
    if path.is_symlink() or path.stat().st_size > 16384:
        raise PermissionError('Bounded private owner session registry required')
    value = json.loads(path.read_text())
    if type(value) is not dict or len(value) > 60:
        raise ValueError('Bounded owner session registry required')
    for identifier, name in value.items():
        if str(uuid.UUID(identifier)) != identifier or not isinstance(name, str) or not 1 <= len(name) <= 160:
            raise ValueError('Invalid owner session metadata')
    return value

def add_session(identifier, name):
    from .node_updater import private_json
    sessions = owned_sessions()
    if str(uuid.UUID(identifier)) != identifier or len(sessions) >= 60:
        raise ValueError('Owner session capacity or identity invalid')
    sessions[identifier] = name
    private_json(root() / 'codex-owner-sessions.json', sessions)

def conversation_sessions():
    """Return stable local Codex thread IDs keyed by PROVIS conversation."""
    path = root() / 'codex-owner-conversations.json'
    if not path.exists():
        return {}
    if path.is_symlink() or path.stat().st_size > 16384:
        raise PermissionError('Bounded private conversation registry required')
    value = json.loads(path.read_text())
    if type(value) is not dict or len(value) > 60:
        raise ValueError('Bounded conversation registry required')
    sessions = owned_sessions()
    for key, identifier in value.items():
        if (type(key) is not str or not 1 <= len(key) <= 140
                or not (key == 'channel' or key.startswith('dm:'))
                or str(uuid.UUID(identifier)) != identifier or identifier not in sessions):
            raise ValueError('Invalid conversation session mapping')
    return value

def bind_conversation(key, identifier):
    from .node_updater import private_json
    if type(key) is not str or not 1 <= len(key) <= 140 or not (key == 'channel' or key.startswith('dm:')):
        raise ValueError('Invalid conversation key')
    identifier = str(uuid.UUID(identifier))
    sessions = owned_sessions()
    if identifier not in sessions:
        raise PermissionError('Only a connector-owned Codex conversation can be bound')
    mappings = conversation_sessions()
    if len(mappings) >= 60 and key not in mappings:
        raise ValueError('Conversation capacity reached')
    mappings[key] = identifier
    private_json(root() / 'codex-owner-conversations.json', mappings)

def agent_identity():
    """Return the stable non-secret identity for this node's local Codex agent."""
    path = root() / 'codex-owner-agent.json'
    if path.exists():
        if path.is_symlink() or path.stat().st_size > 256:
            raise PermissionError('Private local agent identity required')
        value = json.loads(path.read_text())
        identifier = value.get('agent_id') if type(value) is dict else None
        if str(uuid.UUID(identifier)) != identifier:
            raise ValueError('Invalid local agent identity')
        return identifier
    from .node_updater import private_json
    identifier = str(uuid.uuid4())
    private_json(path, {'agent_id':identifier})
    return identifier

class ServerLock:
    """Only one dedicated server may operate this node's published sessions."""
    def __init__(self):
        path = root() / 'codex-owner-server.lock'
        if path.is_symlink():
            raise PermissionError('Private server lock required')
        self.file = path.open('a+b')
        try:
            if path.stat().st_size == 0:
                self.file.write(b'1'); self.file.flush()
            self.file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.file.close()
            raise PermissionError('The dedicated owner server is already connected') from None
    def close(self):
        self.file.close()
