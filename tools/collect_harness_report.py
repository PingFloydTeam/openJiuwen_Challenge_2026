#!/usr/bin/env python3
"""Collect the harness_report for an agent-registry-mcp check-in.

Pure stdlib, one file, works on macOS, Linux and Windows:

  env           every JIUWENSWARM_* variable of this process (uppercased;
                Windows env names are case-insensitive)
  process_tree  command lines of our ancestors — the openJiuwen harness
                shows up here when the caller really runs under it

Ancestry backends, first that works wins:
  Linux    /proc/<pid>/stat (ppid) + /proc/<pid>/cmdline (argv)
  Windows  Toolhelp32 snapshot via ctypes (image names + ppid; pip console
           shims like jiuwenswarm-process.exe carry the marker in the name)
  POSIX    `ps -axo pid,ppid,command` (macOS has no /proc)

If every backend fails, process_tree is "" — env markers alone satisfy the
registry gate. Prints exactly two lines: the single-line JSON report, then
the sha256 hex digest of those exact bytes (the protocol's D). Send the
report string verbatim to register_agent and sign nonce:name:version:D.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import subprocess
import sys

MAX_TREE_CHARS = 600
MAX_CHAIN = 16
_CANARY = "hrpt-canary-7c1f"  # our own shell echoes this script; skip that entry


def _proc_map_linux() -> dict[int, tuple[int, str]] | None:
    """{pid: (ppid, cmdline)} from /proc, or None when there is no /proc."""
    try:
        mapping: dict[int, tuple[int, str]] = {}
        for entry in glob.glob("/proc/[0-9]*"):
            pid = int(entry.rsplit("/", 1)[1])
            with open(f"{entry}/stat", encoding="utf-8", errors="replace") as fh:
                stat = fh.read()
            head, _, rest = stat.rpartition(")")  # comm may contain spaces/parens
            fields = rest.split()  # [state, ppid, ...]
            if len(fields) < 2:
                continue
            cmd = ""
            try:
                with open(f"{entry}/cmdline", "rb") as fh:
                    cmd = b" ".join(fh.read().split(b"\0")).decode("utf-8", "replace").strip()
            except OSError:
                pass
            if not cmd:  # kernel threads: fall back to the comm field
                cmd = head.partition("(")[2].strip()
            mapping[pid] = (int(fields[1]), cmd)
        return mapping or None
    except Exception:
        return None


def _proc_map_windows() -> dict[int, tuple[int, str]] | None:
    """{pid: (ppid, image_name)} via Toolhelp32, or None off-Windows/on error."""
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes

        class PE32W(ctypes.Structure):  # PROCESSENTRY32W
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        k32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PE32W)]
        k32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PE32W)]
        snap = k32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
        invalid = (None, 0, ctypes.c_size_t(-1).value)
        if snap in invalid:
            return None
        mapping: dict[int, tuple[int, str]] = {}
        try:
            entry = PE32W()
            entry.dwSize = ctypes.sizeof(PE32W)
            have = k32.Process32FirstW(snap, ctypes.byref(entry))
            while have:
                mapping[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
                have = k32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            k32.CloseHandle(ctypes.c_void_p(snap))
        return mapping or None
    except Exception:
        return None


def _proc_map_ps() -> dict[int, tuple[int, str]] | None:
    """{pid: (ppid, command)} via ps — the macOS fallback."""
    try:
        done = subprocess.run(["ps", "-axo", "pid=,ppid=,command="],
                              capture_output=True, text=True, timeout=5)
        if done.returncode != 0:
            return None
        mapping: dict[int, tuple[int, str]] = {}
        for line in done.stdout.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                mapping[int(parts[0])] = (int(parts[1]), parts[2])
        return mapping or None
    except Exception:
        return None


def _ancestry(mapping: dict[int, tuple[int, str]]) -> str:
    """Command lines of our ancestors, root first.

    Entries containing the canary are skipped: when this script runs through
    a shell (`python3 - <<EOF ...`), the shell's own command line echoes the
    whole script — including the marker strings — and would fake evidence.
    """
    chain: list[str] = []
    pid = os.getppid()
    seen: set[int] = set()
    while pid and pid not in seen and len(chain) < MAX_CHAIN:
        seen.add(pid)
        entry = mapping.get(pid)
        if entry is None:
            break
        pid, cmd = entry
        if cmd and _CANARY not in cmd:
            chain.append(cmd.strip()[:200])
    return " | ".join(reversed(chain))[:MAX_TREE_CHARS]


def main() -> None:
    try:  # Windows consoles are not UTF-8 by default
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    env = {k.upper(): v for k, v in os.environ.items()
           if k.upper().startswith("JIUWENSWARM_")}
    mapping = _proc_map_linux() or _proc_map_windows() or _proc_map_ps() or {}
    report = json.dumps({"env": env, "process_tree": _ancestry(mapping)},
                        separators=(",", ":"))
    print(report)
    print(hashlib.sha256(report.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
