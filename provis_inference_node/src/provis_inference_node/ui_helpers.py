"""Backward-compatible import path for shared endpoint display helpers."""

from .endpoint_ui import copy_endpoint_to_clipboard, local_broker_endpoints

__all__ = ["copy_endpoint_to_clipboard", "local_broker_endpoints"]
