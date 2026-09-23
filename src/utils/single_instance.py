"""B9 S7: one bot per instance, enforced by an OS-held lock.

The lock is released by the operating system when the process dies -- crash
included -- so there is no stale-lock cleanup to forget. Instance-aware: the
live bot and a TBOT_INSTANCE=B dry-run twin each hold their own.
"""

import os
import sys
import logging

from src.utils.instance_paths import suffixed_path as _suffixed_path

logger = logging.getLogger(__name__)
_HANDLE = None


def claim(name: str = "tbot") -> None:
    """Take the lock or exit. Call once, before anything touches MT5 or the log."""
    global _HANDLE
    _path = _suffixed_path(f"data/{name}.lock")
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    try:
        _HANDLE = open(_path, "a+")
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(_HANDLE.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_HANDLE.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        try:
            _HANDLE.seek(0)
            _who = _HANDLE.read().strip() or "unknown"
        except Exception:
            _who = "unknown"
        print(f"[SINGLE-INSTANCE] REFUSED - {name} is already running ({_who}). Exiting.")
        sys.exit(0)
    _HANDLE.seek(0)
    _HANDLE.truncate()
    _HANDLE.write(f"pid={os.getpid()} exe={sys.executable}")
    _HANDLE.flush()
    print(f"[SINGLE-INSTANCE] {name} lock taken by pid {os.getpid()}")
