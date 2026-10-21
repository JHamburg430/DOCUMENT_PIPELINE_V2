"""Run MCP from the signed node binary without opening a UI or console."""
import io
import os
import sys


def _windows_pipe(number, mode):
    """Duplicate only an inherited anonymous/named pipe; never attach a console."""
    import ctypes as c
    from ctypes import wintypes as w
    import msvcrt

    kernel = c.WinDLL('kernel32', use_last_error=True)
    kernel.GetStdHandle.argtypes = [w.DWORD];kernel.GetStdHandle.restype = w.HANDLE
    kernel.GetFileType.argtypes = [w.HANDLE];kernel.GetFileType.restype = w.DWORD
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.DuplicateHandle.argtypes = [w.HANDLE,w.HANDLE,w.HANDLE,c.POINTER(w.HANDLE),w.DWORD,w.BOOL,w.DWORD]
    kernel.DuplicateHandle.restype = w.BOOL
    kernel.CloseHandle.argtypes = [w.HANDLE];kernel.CloseHandle.restype = w.BOOL
    handle = kernel.GetStdHandle(number & 0xffffffff)
    if not handle or handle == c.c_void_p(-1).value or kernel.GetFileType(handle) != 3:
        raise OSError('MCP requires inherited stdio pipes')
    process = kernel.GetCurrentProcess()
    duplicate = w.HANDLE()
    if not kernel.DuplicateHandle(process,handle,process,c.byref(duplicate),0,False,2):
        raise OSError('Could not duplicate MCP pipe')
    try:
        fd = msvcrt.open_osfhandle(duplicate.value,os.O_BINARY | (os.O_RDONLY if mode == 'r' else os.O_WRONLY))
    except Exception:
        kernel.CloseHandle(duplicate);raise
    binary = os.fdopen(fd,mode+'b',buffering=0)
    return io.TextIOWrapper(binary,encoding='utf-8',newline='\n',write_through=mode=='w')


def run_stdio():
    try:
        # PyInstaller's Windows windowed bootloader sets these to None, even
        # when the MCP client supplied usable inherited OS pipe handles.
        if os.name == 'nt':
            if sys.stdin is None:sys.stdin = _windows_pipe(-10,'r')
            if sys.stdout is None:sys.stdout = _windows_pipe(-11,'w')
        if sys.stdin is None or sys.stdout is None:
            return 2
        from provis_agent_link_mcp import main
        main()
        return 0
    except (OSError,ValueError):
        # Never open a dialog or fall through into the node UI on transport failure.
        return 2
