"""Runtime state that lives in the workspace, not in a process.

Any process that occupies the GPU writes a small JSON heartbeat into the
workspace. Every reader -- the WebUI, the CLI, an agent polling over HTTP --
derives job state from those files alone. Nothing here keeps memory state that
the disk does not, so the web server can restart at any moment and still report
a run that a terminal started.

The contract is documented in ``docs/RUNTIME-CONTRACT.md``:

``runs/<run>/training.json``
    Written by the training process. Step progress plus a terminal status.
``logs/<id>.json``
    Written by whoever spawned a job. Pid, argv, log path and the exit code.
``<workspace>/.gpu.lock``
    One GPU job per machine. Contains the holder's pid, so a dead holder is
    reclaimed instead of blocking every later run.
"""

from __future__ import annotations

import glob
import json
import os
import time

# A heartbeat older than this means the process is alive but not reporting:
# either it is inside a long phase (sampling, export) or it is wedged.
STALE_AFTER = 60.0

# What a heartbeat may claim as a terminal status.
TERMINAL = ("done", "failed", "stopped", "interrupted")

LOCK_NAME = ".gpu.lock"


# -- small file helpers -----------------------------------------------------

def write_json_atomic(path: str, data: dict) -> None:
    """Write JSON so a reader never sees a half-written file.

    A crash in the middle of ``json.dump`` used to leave a truncated heartbeat,
    which readers silently skipped -- losing the one record of a running job.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp = f"{path}.{os.getpid()}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_json(path: str) -> dict | None:
    """Return the JSON object at ``path``, or None when it is missing or torn."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _zombie(pid: int) -> bool:
    """A zombie still has a /proc entry, but it will never make progress again."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            tail = handle.read().rsplit(b")", 1)[1].split()
        return tail[0] == b"Z"
    except (OSError, IndexError):
        return False


def pid_alive(pid) -> bool:
    """True when ``pid`` is a live, non-zombie process.

    Only the kernel answers this. Matching the command line looked safer but
    was wrong in both directions: a wrapper such as ``python -c ...`` is alive
    with no hint in its argv, and a recycled pid inherits a matching one. A
    dead trainer needs no hint to be detected -- its heartbeat stops, and the
    staleness window catches it a minute later.
    """
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, but owned by another user: the pid exists, so it counts.
        pass
    except OSError:
        return False
    return not _zombie(value)


# -- state derivation -------------------------------------------------------

# How long a record may say "starting" without a pid before we call it dead.
SPAWN_GRACE = 30.0


def classify(record: dict, now: float | None = None, stale_after: float = STALE_AFTER) -> dict:
    """Turn one heartbeat or job record into a state.

    This is the only place that decides what "running" means. The WebUI, the
    API and any agent call it (or the same rules), so they cannot disagree.
    """
    now = time.time() if now is None else now
    status = str(record.get("status") or "").lower()
    exit_code = record.get("exit_code")
    beat = float(record.get("time") or 0.0)
    age = round(now - beat, 1) if beat else None
    alive = pid_alive(record.get("pid"))

    if isinstance(exit_code, int):
        if status in ("stopped", "stopping"):
            state = "stopped"
        else:
            state = "done" if exit_code == 0 else "failed"
    elif status in TERMINAL:
        state = status
    elif status == "stopping":
        state = "stopping" if alive else "stopped"
    elif alive:
        state = "running"
    elif status in ("start", "starting") and age is not None and age < SPAWN_GRACE:
        # Spawned by a client but the worker has not reported a pid yet.
        state = "starting"
    elif status or beat or exit_code is not None:
        # It wrote something, so it started; the process is gone without a
        # terminal status: it was killed, or the machine went down.
        state = "interrupted"
    else:
        state = "unknown"

    return {
        "state": state,
        "pid_alive": alive,
        "age": age,
        "stale": bool(alive and age is not None and age > stale_after),
    }


# -- GPU lock ---------------------------------------------------------------

class GpuBusy(RuntimeError):
    """Another process already holds the GPU."""

    def __init__(self, holder: dict, path: str = ""):
        self.holder = holder or {}
        self.path = path
        super().__init__(
            f"GPU busy: pid {self.holder.get('pid') or '?'} "
            f"run {self.holder.get('run_id') or '?'} ({self.holder.get('kind') or 'job'})"
        )


def lock_path(workspace_dir: str) -> str:
    return os.path.join(workspace_dir, LOCK_NAME)


def _sibling_locks(workspace_dir: str) -> list[str]:
    """Locks held by other workspaces: the GPU is shared, the file is not."""
    parent = os.path.dirname(os.path.abspath(workspace_dir))
    mine = os.path.abspath(lock_path(workspace_dir))
    return [path for path in glob.glob(os.path.join(parent, "*", LOCK_NAME))
            if os.path.abspath(path) != mine]


def _live_holder(path: str) -> dict | None:
    holder = read_json(path)
    if not holder:
        return None
    return holder if pid_alive(holder.get("pid")) else None


def check_lock(workspace_dir: str) -> dict | None:
    """Return the holder that blocks this workspace, if any."""
    for path in [lock_path(workspace_dir), *_sibling_locks(workspace_dir)]:
        holder = _live_holder(path)
        if holder:
            return holder
    return None


def acquire_gpu_lock(workspace_dir: str, *, kind: str = "train", run_id: str = "", run_dir: str = "") -> dict:
    """Take the machine-wide GPU lock, reclaiming one whose owner died."""
    mine = lock_path(workspace_dir)
    payload = {
        "pid": os.getpid(),
        "kind": kind,
        "run_id": run_id,
        "run_dir": run_dir,
        "started": time.time(),
    }
    for _ in range(2):
        try:
            handle = os.open(mine, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _live_holder(mine)
            if holder:
                raise GpuBusy(holder, mine) from None
            # The previous holder is gone: take the lock instead of stalling.
            try:
                os.unlink(mine)
            except OSError:
                pass
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(payload, out)
        sibling = next((h for p in _sibling_locks(workspace_dir) if (h := _live_holder(p))), None)
        if sibling:
            release_gpu_lock(workspace_dir)
            raise GpuBusy(sibling, mine)
        return payload
    holder = _live_holder(mine) or read_json(mine) or {}
    raise GpuBusy(holder, mine)


def release_gpu_lock(workspace_dir: str) -> bool:
    """Drop the lock only when this process is the holder."""
    path = lock_path(workspace_dir)
    holder = read_json(path) or {}
    if holder.get("pid") not in (None, os.getpid()):
        return False
    try:
        os.unlink(path)
        return True
    except OSError:
        return False


# -- workspace scan ---------------------------------------------------------

def workspace_root(config) -> str:
    """The workspace a config belongs to, or "" when it lives outside one."""
    base = getattr(config, "_base_dir", "") or ""
    return os.path.abspath(base) if base else ""


def progress_event(record: dict) -> dict | None:
    """Shape the latest metrics line like the SSE progress event the WebUI eats."""
    step = record.get("step")
    if step is None:
        return None
    return {
        "type": "progress",
        "step": int(step),
        "total": int(record.get("total") or 0),
        "loss": float(record.get("train") if record.get("train") is not None else record.get("loss") or 0.0),
        "avg": float(record.get("avg") or 0.0),
        "lr": float(record.get("lr") or 0.0),
        "sec": float(record.get("sec") or 0.0),
        # `vram` stays the allocated peak every older client reads; the other two
        # explain the gap to what the driver reports for the card.
        "vram": float(record.get("peak") or 0.0),
        "vram_reserved": float(record.get("reserved") or 0.0),
        "vram_device": float(record.get("device") or 0.0),
    }


def last_metrics(run_dir: str) -> dict | None:
    """The most recent training line of ``metrics.jsonl``.

    Eval and sample lines carry no loss, so they are skipped: otherwise the last
    thing a finished run shows is a zero loss that never happened.
    """
    path = os.path.join(run_dir, "metrics.jsonl")
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            window = min(size, 8192)
            handle.seek(size - window)
            chunk = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(chunk.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and (data.get("train") is not None or data.get("loss") is not None):
            return data
    return None


def scan_runs(workspace_dir: str) -> list[dict]:
    """Every run directory that has a heartbeat, newest first."""
    out = []
    for beat in glob.glob(os.path.join(workspace_dir, "runs", "*", "training.json")):
        record = read_json(beat)
        if not record:
            continue
        run_dir = os.path.dirname(beat)
        run = os.path.basename(run_dir)
        state = classify(record)
        metrics = last_metrics(run_dir) or {}
        out.append({
            "id": run,
            "kind": "train",
            "source": "cli",
            "state_path": beat,
            "run_id": run,
            "run_dir": run_dir,
            "status": record.get("status") or "",
            "state": state["state"],
            "stale": state["stale"],
            "age": state["age"],
            "pid": record.get("pid"),
            "started": record.get("started") or 0.0,
            "finished": record.get("finished") or 0.0,
            "exit_code": record.get("exit_code"),
            "phase": record.get("phase") or "",
            "step": record.get("step"),
            "total": record.get("total"),
            "last": progress_event({**metrics, "step": record.get("step"), "total": record.get("total")})
            if metrics else progress_event({"step": record.get("step"), "total": record.get("total")}),
        })
    out.sort(key=lambda item: (item.get("started") or 0.0), reverse=True)
    return out


def scan_job_records(workspace_dir: str) -> list[dict]:
    """Job records written by whoever spawned them (train, cache, caption)."""
    out = []
    for path in glob.glob(os.path.join(workspace_dir, "logs", "*.json")):
        record = read_json(path)
        if not record:
            continue
        state = classify(record)
        out.append({
            "id": record.get("id") or os.path.basename(path)[:-5],
            "kind": record.get("kind") or "job",
            "source": "server",
            "state_path": path,
            "workspace": record.get("workspace") or "",
            "run_id": record.get("run_id") or "",
            "run_dir": record.get("run_dir") or "",
            "log": record.get("log") or "",
            "argv": record.get("argv") or [],
            "status": record.get("status") or "",
            "state": state["state"],
            "stale": state["stale"],
            "age": state["age"],
            "pid": record.get("pid"),
            "started": record.get("started") or 0.0,
            "finished": record.get("finished") or 0.0,
            "exit_code": record.get("exit_code"),
            "phase": record.get("phase") or "",
            "stage": record.get("stage") or "",
            "baseline_status": record.get("baseline_status") or "",
            "baseline_error": record.get("baseline_error") or "",
            "config_path": record.get("config_path") or "",
            "profile": record.get("profile") or record.get("workspace") or "",
            "last": record.get("last") or None,
        })
    out.sort(key=lambda item: (item.get("started") or 0.0), reverse=True)
    return out


def scan_jobs(root: str) -> list[dict]:
    """Every job in every workspace under ``root``, keyed by identity.

    One implementation, two readers: the HTTP layer and ``moss-lora status``
    both call this, so a terminal and the WebUI cannot disagree about what is
    running. A job record and a run heartbeat describe the same training, so
    they are linked by run directory rather than listed twice.
    """
    rows: dict[str, dict] = {}
    if not root or not os.path.isdir(root):
        return []
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            continue
        for record in scan_job_records(directory):
            record["workspace"] = record.get("workspace") or name
            rows[record["id"]] = record
        by_run_dir = {os.path.realpath(item["run_dir"]): item for item in rows.values()
                      if item.get("run_dir")}
        for run in scan_runs(directory):
            existing = rows.get(run["id"]) or by_run_dir.get(os.path.realpath(run.get("run_dir") or ""))
            if existing is not None:
                # A job record carries no progress of its own; the heartbeat and
                # the metrics file do. setdefault is not enough: the key exists
                # with a None value.
                if not existing.get("last"):
                    existing["last"] = run.get("last")
                existing["run_dir"] = existing.get("run_dir") or run.get("run_dir")
                existing["run_id"] = existing.get("run_id") or run.get("run_id")
                existing["step"] = run.get("step")
                existing["total"] = run.get("total")
                # The trainer owns the verdict for a training run: it knows the
                # exit code, while the job record only knows the pid went away.
                if run.get("state") in TERMINAL and isinstance(run.get("exit_code"), int):
                    existing["state"] = run["state"]
                    existing["exit_code"] = run["exit_code"]
                    existing["status"] = run.get("status") or existing.get("status")
                continue
            run["workspace"] = name
            rows[run["id"]] = run
    items = sorted(rows.values(), key=lambda item: item.get("started") or 0.0, reverse=True)
    return items


def scan_locks(root: str) -> list[dict]:
    """Every workspace currently holding the GPU lock.

    Two workspaces must not train at once, so the answer is a list rather than
    a single holder.
    """
    out = []
    if not root or not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            continue
        holder = check_lock(directory)
        if holder:
            out.append({**holder, "workspace": holder.get("workspace") or name})
    return out
