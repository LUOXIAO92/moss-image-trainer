# moved from app.py: L62-117, L120, L1726-1746
"""In-memory job registry: one project-wide running slot plus its event log."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field


from .errors import fail


@dataclass
class Job:
    id: str
    kind: str
    config_path: str
    command: list[str]
    status: str = "running"
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    events: list[dict] = field(default_factory=list)
    process: subprocess.Popen | None = None
    last: dict = field(default_factory=dict)
    profile: str = ""
    workspace: str = ""
    run_name: str = ""
    output_dir: str = ""
    log_path: str = ""
    # Where this job publishes its own heartbeat, so the WebUI can report it
    # after a server restart or when the job was started from a terminal.
    state_path: str = ""
    env: dict = field(default_factory=dict)
    # Optional serial pre-steps (currently the no-LoRA baseline).  A failure is
    # reported but never prevents the training command from running.
    pre_commands: list[dict] = field(default_factory=list)
    stage: str = ""
    baseline_status: str = ""
    baseline_error: str = ""
    last_pid: int = 0
    exit_code: int | None = None
    # Resume metadata travels with the job record so a restarted server can
    # still explain how this run was started. ``resume`` is the checkpoint
    # path as shown in the record; ``resume_step`` is parsed from its name.
    resume: str = ""
    resume_step: int | None = None
    # Fresh training truncates its own log; resume keeps the earlier session.
    log_mode: str = "a"
    signal: str = ""
    traceback: str = ""
    persisted_at: float = 0.0

    def push(self, event: dict):
        event["t"] = round(time.time() - self.created, 3)
        self.events.append(event)
        if len(self.events) > 5000:
            del self.events[:1000]


class Jobs:
    def __init__(self):
        self.items: dict[str, Job] = {}
        self.lock = threading.Lock()

    def add(self, job: Job):
        with self.lock:
            self.items[job.id] = job

    def get(self, job_id: str) -> Job:
        job = self.items.get(job_id)
        if job is None:
            raise fail(404, "err.job.not_found", job=job_id)
        return job

    def active(self) -> Job | None:
        """One project-wide slot: training and caching may not overlap."""
        with self.lock:
            return next((j for j in self.items.values() if j.status == "running"), None)

    def all(self) -> list["Job"]:
        with self.lock:
            return list(self.items.values())


JOBS = Jobs()

def _terminate(job: "Job") -> bool:
    """Stop a job and the children it spawned (queue -> train)."""
    import signal

    process = job.process
    if process is None or process.poll() is not None:
        if job.status == "running":
            job.status = "stopped"
            job.push({"type": "log", "line": "terminated by user"})
            return True
        return False
    try:
        # Windows has neither killpg nor process groups; terminate the child directly.
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError):
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    job.status = "stopped"
    job.push({"type": "log", "line": "terminated by user"})
    return True
