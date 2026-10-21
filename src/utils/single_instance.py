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
# B14 Part 3A: the one byte every bot locks (Windows) -- a fixed place, 1 MB in, far past the "pid=... exe=..." text.
# Windows allows locking past the end of a file, and the text before it stays readable for the refusal message.
_LOCK_AT = 1 << 20


def claim(name: str = "tbot") -> None:
    """Take the lock or exit. Call once, before anything touches MT5 or the log."""
    global _HANDLE
    _path = _suffixed_path(f"data/{name}.lock")
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    try:
        _HANDLE = open(_path, "a+")
        if sys.platform == "win32":
            import msvcrt
            # B14 Part 3A: always lock the same byte. msvcrt.locking locks from the file's CURRENT position, and "a+"
            # opens at the END of the file, so the locked byte used to move with the length of the line the last bot
            # wrote ("pid=... exe=..."). When two bots' PIDs had a different number of digits they locked different
            # bytes, and the second bot was not refused.
            os.lseek(_HANDLE.fileno(), _LOCK_AT, os.SEEK_SET)
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
        if name == "tbot" and not os.environ.get("TBOT_INSTANCE"):
            # B14 Part 3A: the running bot reports refused starts in its log and on Telegram (main.py reads this file)
            try:
                import datetime as _dt1a
                os.makedirs("logs", exist_ok=True)
                with open(os.path.join("logs", "startup_refused.txt"), "a", encoding="utf-8") as _f1a:
                    _f1a.write("%s | PID %d | the one-bot lock (%s) is held: %s\n" % (
                        _dt1a.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.getpid(), _path, str(_who)[:120]))
            except Exception:
                pass
        sys.exit(0)
    _HANDLE.seek(0)
    _HANDLE.truncate()
    _HANDLE.write(f"pid={os.getpid()} exe={sys.executable}")
    _HANDLE.flush()
    print(f"[SINGLE-INSTANCE] {name} lock taken by pid {os.getpid()}")
