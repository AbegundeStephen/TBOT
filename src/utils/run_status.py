"""
Shared run-status persistence for dashboard-launched background jobs
(backtest.py, scripts/refresh_data.py both write; src/dashboard/server.py
reads). Extracted from backtest.py's original _write_json_atomic /
server.py's original _read_backtest_status once refresh_data.py became a
second writer of the identical pattern -- one implementation, not two
copies drifting apart.
"""

import json
import os
from pathlib import Path
from datetime import datetime


def write_json_atomic(path, data: dict) -> None:
    """
    Write-temp-then-os.replace so a poller reading status.json/result.json
    from a different process never sees a half-written file mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
    os.replace(tmp_path, path)


def read_run_status(base_dir, run_id: str):
    """
    Read <base_dir>/<run_id>/status.json. Returns None if it doesn't exist
    yet. Synthesizes a "failed" state in two cases the writer process can
    never self-report:
      - "running" with a dead PID: a hard-killed subprocess that never got
        the chance to write its own failure state.
      - "starting" for too long (no PID yet -- the launching route
        pre-writes this before spawning): the subprocess crashed before its
        own first status.json write, e.g. argparse rejecting a bad CLI arg
        via sys.exit() -- that happens before --run-id is even parsed, so
        nothing in that process can ever write "failed" for it.
    """
    path = os.path.join(str(base_dir), run_id, "status.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        status = json.load(f)
    if status.get("state") == "running":
        pid = status.get("pid")
        try:
            import psutil
            alive = pid is not None and psutil.pid_exists(pid)
        except ImportError:
            alive = True  # psutil unavailable -- can't check, assume alive
        if not alive:
            status = dict(status)
            status["state"] = "failed"
            status["error"] = "process terminated unexpectedly"
    elif status.get("state") == "starting":
        started_at = status.get("started_at")
        try:
            age_s = (datetime.utcnow() - datetime.fromisoformat(started_at)).total_seconds()
        except (TypeError, ValueError):
            age_s = 0
        if age_s > 30:
            status = dict(status)
            status["state"] = "failed"
            status["error"] = "process failed to start (never reached its first status update)"
    return status
