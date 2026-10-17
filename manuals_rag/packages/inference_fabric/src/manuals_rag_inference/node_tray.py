"""Windows notification-area icon. Callbacks are queued onto the Tk thread."""
from __future__ import annotations

import ctypes
from ctypes import wintypes as w
import os
import threading


class SingleInstance:
    """A second launch opens the existing tray panel instead of another node."""
    def __init__(self):
        self.handle = None
        self.existing = False
        if os.name != 'nt':
            return
        import hashlib
        identity = hashlib.sha256((os.environ.get('USERDOMAIN', '') + '/' + os.environ.get('USERNAME', '')).encode()).hexdigest()[:24]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateEventW.argtypes = [w.LPVOID, w.BOOL, w.BOOL, w.LPCWSTR]
        self.kernel.CreateEventW.restype = w.HANDLE
        self.kernel.SetEvent.argtypes = [w.HANDLE]
        self.kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateEventW(None, False, False, 'Local\\ProvisComputeShow-' + identity)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.existing = ctypes.get_last_error() == 183
        if self.existing:
            self.kernel.SetEvent(self.handle)

    def watch(self, emit):
        if self.handle and not self.existing:
            def wait():
                while self.kernel.WaitForSingleObject(self.handle, 0xffffffff) == 0:
                    emit('show_existing', None)
            threading.Thread(target=wait, daemon=True, name='provis-show-panel').start()

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class NodeTray:
    def __init__(self, emit):
        self.emit = emit
        self.ready = threading.Event()
        self.hwnd = None
        self.error = None
        if os.name == "nt":
            threading.Thread(target=self._run, name="provis-tray", daemon=True).start()
        else:
            self.error = "unsupported"
            self.ready.set()

    def close(self):
        if self.hwnd:
            self._post(self.hwnd, 0x0010, 0, 0)

    def _run(self):
        try:
            self._windows_loop()
        except Exception:
            self.error = "tray_unavailable"
            self.emit("tray_error", None)
        finally:
            self.ready.set()

    def _windows_loop(self):
        user = ctypes.WinDLL("user32", use_last_error=True)
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        pointer = ctypes.c_ssize_t
        callback_type = ctypes.WINFUNCTYPE(pointer, w.HWND, w.UINT, w.WPARAM, w.LPARAM)

        class WindowClass(ctypes.Structure):
            _fields_ = [("style", w.UINT), ("proc", callback_type), ("cls_extra", ctypes.c_int),
                        ("win_extra", ctypes.c_int), ("instance", w.HINSTANCE), ("icon", w.HICON),
                        ("cursor", w.HANDLE), ("background", w.HBRUSH), ("menu", w.LPCWSTR), ("name", w.LPCWSTR)]

        class NotifyData(ctypes.Structure):
            _fields_ = [("size", w.DWORD), ("hwnd", w.HWND), ("id", w.UINT), ("flags", w.UINT),
                        ("message", w.UINT), ("icon", w.HICON), ("tip", w.WCHAR * 128), ("state", w.DWORD),
                        ("state_mask", w.DWORD), ("info", w.WCHAR * 256), ("version", w.UINT),
                        ("title", w.WCHAR * 64), ("info_flags", w.DWORD), ("guid", ctypes.c_byte * 16),
                        ("balloon_icon", w.HICON)]

        kernel.GetModuleHandleW.restype = w.HMODULE
        user.DefWindowProcW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
        user.DefWindowProcW.restype = pointer
        user.CreateWindowExW.argtypes = [w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HWND, w.HMENU, w.HINSTANCE, w.LPVOID]
        user.CreateWindowExW.restype = w.HWND
        user.LoadIconW.argtypes = [w.HINSTANCE, w.LPCWSTR]
        user.LoadIconW.restype = w.HICON
        user.CreatePopupMenu.restype = w.HMENU
        user.AppendMenuW.argtypes = [w.HMENU, w.UINT, ctypes.c_size_t, w.LPCWSTR]
        user.TrackPopupMenu.argtypes = [w.HMENU, w.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HWND, w.LPVOID]
        user.DestroyMenu.argtypes = [w.HMENU]
        user.DestroyWindow.argtypes = [w.HWND]
        user.SetForegroundWindow.argtypes = [w.HWND]
        user.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
        self._post = user.PostMessageW
        user.GetMessageW.argtypes = [ctypes.POINTER(w.MSG), w.HWND, w.UINT, w.UINT]
        user.DispatchMessageW.argtypes = [ctypes.POINTER(w.MSG)]
        user.DispatchMessageW.restype = pointer
        user.UnregisterClassW.argtypes = [w.LPCWSTR, w.HINSTANCE]
        user.CreateIcon.argtypes = [w.HINSTANCE, ctypes.c_int, ctypes.c_int, w.BYTE, w.BYTE, ctypes.c_void_p, ctypes.c_void_p]
        user.CreateIcon.restype = w.HICON
        user.DestroyIcon.argtypes = [w.HICON]
        shell.Shell_NotifyIconW.argtypes = [w.DWORD, ctypes.POINTER(NotifyData)]
        taskbar_restart = user.RegisterWindowMessageW("TaskbarCreated")
        notify = NotifyData()
        self._notify = notify

        @callback_type
        def procedure(hwnd, message, wp, lp):
            if message == taskbar_restart:
                shell.Shell_NotifyIconW(0, ctypes.byref(notify))
                return 0
            if message == 0x8001:
                if lp in (0x0202, 0x0203):
                    self.emit("tray_show", None)
                elif lp == 0x0205:
                    point = w.POINT()
                    user.GetCursorPos(ctypes.byref(point))
                    menu = user.CreatePopupMenu()
                    for identifier, label in ((1, "Open PROVIS"), (2, "Pause sharing"), (3, "Resume sharing"), (4, "Disconnect safely"), (5, "Quit")):
                        user.AppendMenuW(menu, 0, identifier, label)
                    user.SetForegroundWindow(hwnd)
                    selected = user.TrackPopupMenu(menu, 0x0100 | 0x0002, point.x, point.y, 0, hwnd, None)
                    user.DestroyMenu(menu)
                    user.PostMessageW(hwnd, 0, 0, 0)
                    action = {1: "tray_show", 2: "tray_pause", 3: "tray_resume", 4: "tray_disconnect", 5: "tray_quit"}.get(selected)
                    if action:
                        self.emit(action, None)
                return 0
            if message == 0x0010:
                shell.Shell_NotifyIconW(2, ctypes.byref(notify))
                user.DestroyWindow(hwnd)
                return 0
            if message == 0x0002:
                user.PostQuitMessage(0)
                return 0
            return user.DefWindowProcW(hwnd, message, wp, lp)

        self._procedure = procedure
        instance = kernel.GetModuleHandleW(None)
        name = "ProvisInferenceTray" + str(os.getpid())
        wc = WindowClass(0, procedure, 0, 0, instance, None, None, None, None, name)
        if not user.RegisterClassW(ctypes.byref(wc)):
            raise ctypes.WinError(ctypes.get_last_error())
        hwnd = user.CreateWindowExW(0, name, name, 0, 0, 0, 0, 0, None, None, instance, None)
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = hwnd
        notify.size, notify.hwnd, notify.id = ctypes.sizeof(NotifyData), hwnd, 1
        notify.flags, notify.message = 0x0001 | 0x0002 | 0x0004, 0x8001
        # Small code-native PROVIS mark, no external assets or image dependencies.
        pixels = bytearray()
        for y in range(31, -1, -1):
            for x in range(32):
                letter = 9 <= x < 13 and 7 <= y < 26 or 13 <= x < 23 and (7 <= y < 11 or 15 <= y < 19) or 20 <= x < 24 and 10 <= y < 16
                pixels.extend((255, 255, 255, 255) if letter else (139, 70, 24, 255))
        mask = ctypes.create_string_buffer(bytes(128))
        bitmap = ctypes.create_string_buffer(bytes(pixels))
        icon = user.CreateIcon(instance, 32, 32, 1, 32, mask, bitmap)
        notify.icon = icon or user.LoadIconW(None, ctypes.cast(ctypes.c_void_p(32516), w.LPCWSTR))
        notify.tip = "PROVIS • Private compute"
        if not shell.Shell_NotifyIconW(0, ctypes.byref(notify)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.ready.set()
        message = w.MSG()
        while user.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            user.TranslateMessage(ctypes.byref(message))
            user.DispatchMessageW(ctypes.byref(message))
        user.UnregisterClassW(name, instance)
        if icon:
            user.DestroyIcon(icon)
        self.hwnd = None
