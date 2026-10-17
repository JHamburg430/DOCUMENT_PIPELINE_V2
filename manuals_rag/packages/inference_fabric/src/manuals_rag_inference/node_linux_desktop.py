"""Linux desktop integration. No enrollment, inference content or credentials."""
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading


def private_runtime():
    root = Path(os.environ.get('XDG_RUNTIME_DIR') or tempfile.gettempdir())
    directory = root / ('provis-inference-' + str(os.getuid()))
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError('Private desktop runtime required')
    return directory


class LinuxInstance:
    def __init__(self):
        import fcntl
        self.directory = private_runtime()
        self.path = self.directory / 'panel.sock'
        self.lock = os.open(self.directory / 'instance.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.server = None
        self.existing = False
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.existing = True
            # The first process can still be binding its socket after acquiring the lock.
            import time
            for attempt in range(20):
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.settimeout(1)
                        client.connect(str(self.path));client.sendall(b'SHOW\n')
                    break
                except (ConnectionError, FileNotFoundError):
                    if attempt == 19: raise RuntimeError('Existing node is still starting') from None
                    time.sleep(0.1)
            return
        if self.path.exists():
            self.path.unlink()  # Owned private directory, under our exclusive instance lock.
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path));os.chmod(self.path, 0o600)
        self.server.listen(2)

    def watch(self, emit):
        if self.server is None:
            return
        def receive():
            import struct
            while self.server is not None:
                server = self.server
                try:
                    connection, _ = server.accept()
                    with connection:
                        connection.settimeout(1)
                        _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if uid == os.getuid() and connection.recv(16) == b'SHOW\n':
                            emit('show_existing', None)
                except OSError:
                    if self.server is None: return
        threading.Thread(target=receive, daemon=True, name='provis-show-panel').start()

    def close(self):
        if self.server is not None:
            server, self.server = self.server, None
            server.close()
            self.path.unlink(missing_ok=True)
        if self.lock is not None:
            os.close(self.lock);self.lock = None


def run_indicator(emit, ready, stopped):
    """GTK lives on this thread; all Tk operations are queued to its main thread."""
    import gi
    gi.require_version('Gtk', '3.0')
    try:
        gi.require_version('AyatanaAppIndicator3', '0.1')
        from gi.repository import AyatanaAppIndicator3 as Indicator
    except (ValueError, ImportError):
        gi.require_version('AppIndicator3', '0.1')
        from gi.repository import AppIndicator3 as Indicator
    from gi.repository import Gtk, GLib, Gio
    initialized, _ = Gtk.init_check([])
    if not initialized:
        raise RuntimeError('Desktop session unavailable')
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    owner = bus.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'NameHasOwner',
                          GLib.Variant('(s)', ('org.kde.StatusNotifierWatcher',)), GLib.VariantType.new('(b)'), Gio.DBusCallFlags.NONE, 2000, None)
    if not owner.unpack()[0]:
        raise RuntimeError('Desktop tray unavailable')
    icon = private_runtime() / 'provis-compute.svg'
    icon.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" rx="14" fill="#18468b"/><path d="M20 50V14h18a12 12 0 0 1 0 24H28v12zM28 22v8h10a4 4 0 0 0 0-8z" fill="white"/></svg>', encoding='utf-8')
    indicator = Indicator.Indicator.new('provis-private-compute', str(icon), Indicator.IndicatorCategory.APPLICATION_STATUS)
    indicator.set_title('PROVIS Private Compute')
    menu = Gtk.Menu()
    for label, action in [('Open PROVIS', 'show_existing'), ('Pause sharing', 'tray_pause'), ('Resume sharing', 'tray_resume'), ('Disconnect safely', 'tray_disconnect'), ('Quit', 'tray_quit')]:
        item = Gtk.MenuItem.new_with_label(label)
        item.connect('activate', lambda _item, event=action: emit(event, None))
        menu.append(item)
    menu.show_all();indicator.set_menu(menu)
    indicator.set_status(Indicator.IndicatorStatus.ACTIVE)
    ready.set()
    def poll_stop():
        if stopped.is_set():
            indicator.set_status(Indicator.IndicatorStatus.PASSIVE)
            Gtk.main_quit()
            return False
        return True
    GLib.timeout_add(250, poll_stop)
    Gtk.main()
