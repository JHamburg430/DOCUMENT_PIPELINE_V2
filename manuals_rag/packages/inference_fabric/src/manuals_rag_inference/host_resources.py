"""Local host capacity only; no processes, filenames or workload content."""
from __future__ import annotations

import ctypes
import math
import os
import platform
from pathlib import Path
import shutil
import threading
import time


def validate_host_resources(value):
    """Accept bounded numeric telemetry; older nodes legitimately omit it."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid host resources")
    allowed = {"cpu_count", "cpu_percent", "memory_total_bytes", "memory_available_bytes",
               "disk_total_bytes", "disk_free_bytes", "observed_at_epoch"}
    result = {}
    for key, item in value.items():
        if key not in allowed:
            continue
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError("invalid host resource value")
        if key == "cpu_percent" and item > 100:
            raise ValueError("invalid CPU percentage")
        if key not in {"cpu_percent", "observed_at_epoch"} and (not isinstance(item, int) or item > 2**63 - 1):
            raise ValueError("invalid host resource capacity")
        result[key] = item
    for total, free in (("memory_total_bytes", "memory_available_bytes"), ("disk_total_bytes", "disk_free_bytes")):
        if total in result and free in result and result[free] > result[total]:
            raise ValueError("invalid host resource free capacity")
    return result


class HostResourceSampler:
    def __init__(self):
        self._previous_cpu = None
        self._lock = threading.Lock()

    def sample(self):
        values = {"cpu_count": os.cpu_count(), "observed_at_epoch": time.time()}
        try:
            # Capacity of the volume containing the user's home; never report its path.
            disk = shutil.disk_usage(Path.home())
            values.update(disk_total_bytes=disk.total, disk_free_bytes=disk.free)
        except (OSError, RuntimeError):
            pass
        try:
            if platform.system() == "Windows":
                class MemoryStatus(ctypes.Structure):
                    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                        (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
                memory = MemoryStatus()
                memory.length = ctypes.sizeof(memory)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                    values.update(memory_total_bytes=memory.total, memory_available_bytes=memory.available)
                idle, kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
                if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                    self._cpu(values, kernel.value + user.value, idle.value)
            else:
                memory = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, rest = line.split(":", 1)
                    if key in {"MemTotal", "MemAvailable"}:
                        memory[key] = int(rest.split()[0]) * 1024
                if "MemTotal" in memory:
                    values["memory_total_bytes"] = memory["MemTotal"]
                if "MemAvailable" in memory:
                    values["memory_available_bytes"] = memory["MemAvailable"]
                ticks = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
                # Guest ticks are already included in user/nice.
                self._cpu(values, sum(ticks[:8]), ticks[3] + (ticks[4] if len(ticks) > 4 else 0))
        except (OSError, ValueError, AttributeError, IndexError):
            pass
        return validate_host_resources(values)

    def _cpu(self, values, total, idle):
        with self._lock:
            previous = self._previous_cpu
            self._previous_cpu = (total, idle)
        if previous and total > previous[0] and idle >= previous[1]:
            values["cpu_percent"] = round(max(0, min(100, 100 * (1 - (idle - previous[1]) / (total - previous[0])))), 1)
