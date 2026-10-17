"""Shared Tcl interpreter for desktop UI tests; each test owns its own window."""
import pytest

@pytest.fixture(scope='session')
def inference_gui_root():
    import tkinter as tk
    root=tk.Tk();root.withdraw()
    yield root
    root.destroy()
