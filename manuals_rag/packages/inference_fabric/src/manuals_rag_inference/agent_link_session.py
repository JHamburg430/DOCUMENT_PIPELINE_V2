"""Short-lived owner capability, DPAPI protected on Windows; private file on Linux."""
import base64
import json
import os
from pathlib import Path
import time


def _dpapi(raw, decrypt=False):
    import ctypes as c
    from ctypes import wintypes as w
    class Blob(c.Structure):
        _fields_ = [('length', w.DWORD), ('data', c.c_void_p)]
    buffer = c.create_string_buffer(raw)
    source, destination = Blob(len(raw), c.cast(buffer, c.c_void_p)), Blob()
    dll = c.WinDLL('crypt32', use_last_error=True)
    if decrypt:
        function = dll.CryptUnprotectData
        function.argtypes = [c.POINTER(Blob), c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p, w.DWORD, c.POINTER(Blob)]
        success = function(c.byref(source), None, None, None, None, 1, c.byref(destination))
    else:
        function = dll.CryptProtectData
        function.argtypes = [c.POINTER(Blob), w.LPCWSTR, c.c_void_p, c.c_void_p, c.c_void_p, w.DWORD, c.POINTER(Blob)]
        success = function(c.byref(source), 'PROVIS owner agent link', None, None, None, 1, c.byref(destination))
    if not success: raise PermissionError('Owner session protection unavailable')
    try: return c.string_at(destination.data, destination.length)
    finally:
        free = c.WinDLL('kernel32').LocalFree
        free.argtypes = [c.c_void_p];free.restype = c.c_void_p
        free(destination.data)


def default_path():
    root = Path(os.environ['LOCALAPPDATA']) / 'ProVIS' if os.name == 'nt' else Path.home() / '.config/ProVIS'
    return root / 'agent-link-session.json'


def save_session(value, path=None):
    target = Path(path) if path else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    outer = {'dpapi': base64.b64encode(_dpapi(raw)).decode()} if os.name == 'nt' else value
    temporary = target.with_name(target.name + '.pending')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream: json.dump(outer, stream)
    os.replace(temporary, target)


def load_session(path=None):
    target = Path(path) if path else default_path()
    if os.name != 'nt':
        info = target.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError('Private owner session file required')
    value = json.loads(target.read_text(encoding='utf-8'))
    if os.name == 'nt': value = json.loads(_dpapi(base64.b64decode(value['dpapi']), True))
    if value['expires_at_epoch'] <= time.time(): raise PermissionError('Owner sign-in expired')
    return value


def clear_session():
    default_path().unlink(missing_ok=True)
