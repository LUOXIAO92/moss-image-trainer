"""Backend contracts for resume, run state cleanup and error traces.

No training process and no network: the launcher is stubbed when the API is
exercised, and trainer helpers are called on an uninitialised instance.

    .venv\\Scripts\\python.exe tests/test_train_resume.py
"""

from __future__ import annotations

import asyncio
import faulthandler
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import yaml
from fastapi import HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from moss_lora import cli as cli_module  # noqa: E402
from moss_lora import queue as queue_module  # noqa: E402
from moss_lora import runtime  # noqa: E402
from moss_lora.trainer import Trainer  # noqa: E402
from moss_lora.web import jobs as job_registry  # noqa: E402
from moss_lora.web.routers import jobs as jobs_router  # noqa: E402
from moss_lora.web.routers import workspaces as workspaces_router  # noqa: E402
from moss_lora.web.services import launcher  # noqa: E402
from moss_lora.web.services.workspace import _workspace_root  # noqa: E402

WORKSPACE = "zz-train-resume-check"


class FakeRequest:
    def __init__(self, payload: dict):
        self.payload = payload

    async def json(self):
        return self.payload


def workspace_dir() -> str:
    return os.path.join(_workspace_root(), WORKSPACE)


def config_dict(steps: int = 100, name: str = "demo") -> dict:
    base = workspace_dir()
    return {
        "name": name,
        "model": {
            "dit": os.path.join(base, "dit.safetensors"),
            "text_encoder": "",
            "vae": os.path.join(base, "vae.safetensors"),
        },
        "dataset": {"path": os.path.join(base, "dataset")},
        "cache": {"dir": "train_cache"},
        "output_dir": "runs",
        "train": {"steps": steps, "lr": 1e-4, "batch_size": 1},
        "lora": {"rank": 16},
    }


def write_config(run: str, steps: int = 100) -> str:
    directory = os.path.join(workspace_dir(), "runs", run)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_dict(steps, run), handle, allow_unicode=True, sort_keys=False)
    return directory


def prepare_workspace() -> None:
    base = workspace_dir()
    shutil.rmtree(base, ignore_errors=True)
    for sub in ("dataset", "train_cache", "logs", "runs"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    for name in ("dit.safetensors", "vae.safetensors"):
        with open(os.path.join(base, name), "wb") as handle:
            handle.write(b"\0")


def cleanup_workspace() -> None:
    shutil.rmtree(workspace_dir(), ignore_errors=True)
    job_registry.JOBS.items.clear()


def check_fresh_reset() -> None:
    root = tempfile.mkdtemp(prefix="moss-reset-")
    try:
        os.makedirs(os.path.join(root, "samples"))
        with open(os.path.join(root, "metrics.jsonl"), "w", encoding="utf-8") as handle:
            handle.write('{"step": 1}\n')
        with open(os.path.join(root, "training.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        os.makedirs(os.path.join(root, "checkpoints"))
        with open(os.path.join(root, "checkpoints", "step000001.safetensors"), "wb") as handle:
            handle.write(b"keep")
        os.makedirs(os.path.join(root, "exports"))
        with open(os.path.join(root, "exports", "ship.safetensors"), "wb") as handle:
            handle.write(b"keep")

        trainer = object.__new__(Trainer)
        trainer.out_dir = root
        logs: list[str] = []
        trainer.log = logs.append
        trainer.reset_run_state()

        assert not os.path.exists(os.path.join(root, "metrics.jsonl"))
        assert not os.path.exists(os.path.join(root, "training.json"))
        assert not os.path.exists(os.path.join(root, "samples"))
        assert os.path.isfile(os.path.join(root, "checkpoints", "step000001.safetensors"))
        assert os.path.isfile(os.path.join(root, "exports", "ship.safetensors"))
        assert any("fresh run" in line for line in logs), logs
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_fresh_reset_waits_for_lock() -> None:
    root = tempfile.mkdtemp(prefix="moss-lock-reset-")
    workspace = os.path.join(root, "ws")
    out_dir = os.path.join(workspace, "runs", "live")
    try:
        os.makedirs(os.path.join(out_dir, "samples"))
        for name in ("metrics.jsonl", "training.json"):
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
                handle.write("live run data")
        with open(os.path.join(out_dir, "samples", "step000001_0.png"), "wb") as handle:
            handle.write(b"png")

        runtime.acquire_gpu_lock(workspace, kind="train", run_id="other",
                                 run_dir=os.path.join(workspace, "runs", "other"))
        try:
            trainer = object.__new__(Trainer)
            trainer.workspace_root = workspace
            trainer.out_dir = out_dir
            trainer.config = SimpleNamespace(name="live")
            trainer._lock_held = False
            trainer.log = lambda *args: None
            try:
                trainer.start(reset=True)
            except (SystemExit, runtime.GpuBusy):
                # _acquire_lock wraps GpuBusy in SystemExit for the CLI.
                pass
            else:
                raise AssertionError("start(reset=True) took a held lock")

            assert os.path.isfile(os.path.join(out_dir, "metrics.jsonl"))
            assert os.path.isfile(os.path.join(out_dir, "training.json"))
            assert os.path.isfile(os.path.join(out_dir, "samples", "step000001_0.png"))
        finally:
            runtime.release_gpu_lock(workspace)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_disk_job_signal_fields() -> None:
    root = tempfile.mkdtemp(prefix="moss-job-record-")
    workspace = os.path.join(root, "ws")
    try:
        os.makedirs(os.path.join(workspace, "logs"))
        record = {
            "id": "sig-record",
            "kind": "train",
            "status": "error",
            "exit_code": -11,
            "signal": "SIGSEGV",
            "traceback": "Traceback (most recent call last):\nRuntimeError: boom",
            "resume": "runs/demo/checkpoints/step000800.safetensors",
            "resume_step": 800,
            "time": 1.0,
            "started": 1.0,
            "finished": 2.0,
        }
        with open(os.path.join(workspace, "logs", "sig-record.json"), "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        row = runtime.scan_job_records(workspace)[0]
        assert row["signal"] == "SIGSEGV", row
        assert "RuntimeError: boom" in row["traceback"], row
        assert row["resume_step"] == 800, row
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_prune_after_resume() -> None:
    root = tempfile.mkdtemp(prefix="moss-prune-")
    try:
        with open(os.path.join(root, "metrics.jsonl"), "w", encoding="utf-8") as handle:
            for row in (
                {"step": 799, "train": 1.0},
                {"step": 800, "train": 0.9},
                {"step": 800, "eval": 0.8},
                {"step": 801, "train": 0.7},
                {"step": 840, "eval": 0.6},
            ):
                handle.write(json.dumps(row) + "\n")
        sample_dir = os.path.join(root, "samples")
        os.makedirs(sample_dir)
        for name in ("step000800_0.png", "step000801_0.png", "step000840_0.png", "trajectory.png"):
            with open(os.path.join(sample_dir, name), "wb") as handle:
                handle.write(b"png")

        trainer = object.__new__(Trainer)
        trainer.out_dir = root
        logs: list[str] = []
        trainer.log = logs.append
        trainer._prune_after_resume(800)

        with open(os.path.join(root, "metrics.jsonl"), encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert [row["step"] for row in rows] == [799, 800, 800], rows
        assert os.path.isfile(os.path.join(sample_dir, "step000800_0.png"))
        assert not os.path.exists(os.path.join(sample_dir, "step000801_0.png"))
        assert not os.path.exists(os.path.join(sample_dir, "step000840_0.png"))
        assert not os.path.exists(os.path.join(sample_dir, "trajectory.png"))
        assert any("resume from step 800" in line for line in logs), logs
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_trainer_error_trace() -> None:
    trainer = object.__new__(Trainer)
    trainer.out_dir = tempfile.mkdtemp(prefix="moss-trace-")
    logs: list[str] = []
    beats: list[tuple] = []
    trainer.log = logs.append
    trainer.state = SimpleNamespace(step=0)
    trainer.workspace_root = ""
    trainer._lock_held = False
    trainer._acquire_lock = lambda: None
    trainer._release_lock = lambda: None
    trainer._write_heartbeat = lambda *args, **kwargs: beats.append((args, kwargs))
    trainer._save_on_stop = lambda: None

    def boom(resume=""):
        raise RuntimeError("train boom")

    trainer._train_loop = boom
    try:
        trainer.train()
    except RuntimeError:
        pass
    else:
        raise AssertionError("train() swallowed the exception")
    text = "\n".join(logs)
    assert "training failed" in text, text
    assert "Traceback (most recent call last)" in text, text
    assert "RuntimeError: train boom" in text, text
    assert beats and beats[-1][0][1] == "failed", beats
    shutil.rmtree(trainer.out_dir, ignore_errors=True)

    logs.clear()
    beats.clear()

    def interrupt(resume=""):
        raise KeyboardInterrupt

    trainer._train_loop = interrupt
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    text = "\n".join(logs)
    assert "training stopped by user" in text, text
    assert "Traceback" not in text, text
    assert beats and beats[-1][0][1] == "stopped", beats


def check_launcher_trace_and_signal() -> None:
    assert launcher.signal_name(-11) == "SIGSEGV"
    assert launcher.signal_name(-15) == "SIGTERM"
    assert launcher.signal_name(0) == ""
    assert launcher.signal_name(None) == ""

    root = tempfile.mkdtemp(prefix="moss-launcher-")
    try:
        job = job_registry.Job(
            id="trace",
            kind="train",
            config_path=os.path.join(root, "config.yaml"),
            command=[sys.executable, "-c", "raise RuntimeError('child boom')"],
            log_path=os.path.join(root, "job.log"),
            state_path=os.path.join(root, "job.json"),
            log_mode="w",
        )
        with open(job.log_path, "w", encoding="utf-8") as handle:
            handle.write("old session\n")
        launcher._run(job)
        with open(job.log_path, encoding="utf-8") as handle:
            text = handle.read()
        assert "old session" not in text, text
        assert "Traceback (most recent call last)" in text, text
        assert "RuntimeError: child boom" in text, text
        assert job.exit_code == 1, job.exit_code
        assert job.env.get("PYTHONFAULTHANDLER") == "1", job.env

        parent = job_registry.Job(
            id="parent",
            kind="train",
            config_path=os.path.join(root, "config.yaml"),
            command=[],
            log_path=os.path.join(root, "parent.log"),
            state_path=os.path.join(root, "parent.json"),
            log_mode="w",
        )
        original = launcher._stream_command
        try:
            launcher._stream_command = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("parent boom"))
            launcher._run(parent)
        finally:
            launcher._stream_command = original
        with open(parent.log_path, encoding="utf-8") as handle:
            parent_text = handle.read()
        assert "Traceback (most recent call last)" in parent_text, parent_text
        assert "RuntimeError: parent boom" in parent_text, parent_text
        assert "parent boom" in parent.traceback, parent.traceback
        assert parent.exit_code == 1, parent.exit_code
        with open(parent.state_path, encoding="utf-8") as handle:
            record = json.load(handle)
        assert "parent boom" in record.get("traceback", ""), record

        # A real signal return code is reported by name; no child is killed.
        signalled = job_registry.Job(
            id="sig",
            kind="train",
            config_path=os.path.join(root, "config.yaml"),
            command=[],
            log_path=os.path.join(root, "sig.log"),
            state_path=os.path.join(root, "sig.json"),
            log_mode="w",
        )
        try:
            launcher._stream_command = lambda job, *args, **kwargs: (
                setattr(job, "signal", launcher.signal_name(-11)), -11)[1]
            launcher._run(signalled)
        finally:
            launcher._stream_command = original
        with open(signalled.log_path, encoding="utf-8") as handle:
            sig_text = handle.read()
        assert "[job] error · exit code -11 (SIGSEGV)" in sig_text, sig_text
        assert signalled.exit_code == -11, signalled.exit_code
        assert signalled.signal == "SIGSEGV", signalled.signal
        terminal = [event for event in signalled.events if event.get("type") == "error"][-1]
        assert terminal["signal"] == "SIGSEGV", terminal
        assert terminal["exit_code"] == -11, terminal
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_queue_checkpoint_filter() -> None:
    root = tempfile.mkdtemp(prefix="moss-queue-")
    try:
        os.makedirs(os.path.join(root, "checkpoints"))
        for name in ("step000100.safetensors", "step000100.state.safetensors",
                     "state.safetensors", "final.safetensors"):
            with open(os.path.join(root, "checkpoints", name), "wb") as handle:
                handle.write(b"x")
        found = queue_module._checkpoints(root)
        assert [step for step, _ in found] == [100], found
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_command_resume() -> None:
    checkpoint = os.path.join(tempfile.gettempdir(), "step000800.safetensors")
    command = launcher._command("train", "/tmp/run.yaml", resume=checkpoint)
    assert command[-2:] == ["--resume", os.path.abspath(checkpoint)], command
    assert "--resume" not in launcher._command("train", "/tmp/run.yaml"), "fresh train got --resume"


async def check_api_resume() -> None:
    prepare_workspace()
    run_dir = write_config("demo", 100)
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    for step in (100, 200):
        with open(os.path.join(run_dir, "checkpoints", f"step{step:06d}.safetensors"), "wb") as handle:
            handle.write(b"weights")

    captured: list = []
    original_run = jobs_router._run
    original_summary = jobs_router._cache_summary
    try:
        jobs_router._run = lambda job: captured.append(job)
        jobs_router._cache_summary = lambda config: {
            "index_ready": True, "items": 1, "missing_text": 0,
        }

        latest = await jobs_router.api_start(FakeRequest({
            "kind": "train", "workspace": WORKSPACE, "run": "demo", "resume": "latest",
        }))
        job = captured[-1]
        assert latest["resume_step"] == 200, latest
        assert job.log_mode == "a", job.log_mode
        assert job.resume_step == 200, job.resume_step
        assert os.path.basename(job.command[-1]) == "step000200.safetensors", job.command

        job_registry.JOBS.items.clear()
        fresh = await jobs_router.api_start(FakeRequest({
            "kind": "train", "workspace": WORKSPACE, "run": "demo",
        }))
        fresh_job = captured[-1]
        assert fresh["resume_step"] is None, fresh
        assert fresh_job.log_mode == "w", fresh_job.log_mode
        assert "--resume" not in fresh_job.command, fresh_job.command

        # The API refuses while a job is live, before it inspects the resume
        # value. Clear the stubbed registry so the validation paths are reached.
        job_registry.JOBS.items.clear()

        async def expect(handler, payload, status: int):
            try:
                await handler(FakeRequest(payload))
            except HTTPException as exc:
                assert exc.status_code == status, (payload, exc.status_code, exc.detail)
                return exc
            raise AssertionError(f"expected HTTP {status}: {payload}")

        await expect(jobs_router.api_start, {
            "kind": "cache", "workspace": WORKSPACE, "run": "demo", "resume": "latest",
        }, 400)
        await expect(jobs_router.api_start, {
            "kind": "train", "workspace": WORKSPACE, "resume": "latest",
        }, 400)
        await expect(jobs_router.api_start, {
            "kind": "train", "workspace": WORKSPACE, "run": "demo", "resume": "../evil.safetensors",
        }, 400)
        await expect(jobs_router.api_start, {
            "kind": "train", "workspace": WORKSPACE, "run": "demo", "resume": "step000999.safetensors",
        }, 404)
        await expect(jobs_router.api_start, {
            "kind": "train", "workspace": WORKSPACE, "run": "demo", "resume": "final.safetensors",
        }, 400)
    finally:
        jobs_router._run = original_run
        jobs_router._cache_summary = original_summary
        cleanup_workspace()


def check_run_list_and_dashboard() -> None:
    prepare_workspace()
    try:
        demo_dir = write_config("demo", 100)
        os.makedirs(os.path.join(demo_dir, "checkpoints"))
        with open(os.path.join(demo_dir, "checkpoints", "step000050.safetensors"), "wb") as handle:
            handle.write(b"weights")
        # ``final`` is an export, not a resume target; the API must keep the
        # numbered checkpoint even when the final export is newer.
        with open(os.path.join(demo_dir, "checkpoints", "final.safetensors"), "wb") as handle:
            handle.write(b"export")
        with open(os.path.join(demo_dir, "metrics.jsonl"), "w", encoding="utf-8") as handle:
            handle.write('{"step": 10, "total": 100, "train": 1.0}\n')
        os.makedirs(os.path.join(demo_dir, "samples"))
        with open(os.path.join(demo_dir, "samples", "step000010_0.png"), "wb") as handle:
            handle.write(b"png")
        write_config("fresh", 100)
        queued_dir = write_config("queued", 100)
        os.makedirs(os.path.join(queued_dir, "checkpoints"))
        with open(os.path.join(queued_dir, "checkpoints", "step000050.safetensors"), "wb") as handle:
            handle.write(b"weights")

        runs = workspaces_router.api_workspace_runs(WORKSPACE)["runs"]
        by_name = {run["name"]: run for run in runs}
        assert set(by_name) == {"demo", "fresh", "queued"}, by_name
        assert by_name["demo"]["has_metrics"] is True
        assert by_name["demo"]["checkpoint"] == "step000050.safetensors"
        assert by_name["demo"]["checkpoint_step"] == 50
        assert by_name["demo"]["resumable"] is True
        assert by_name["demo"]["samples"] == 1
        assert by_name["fresh"]["has_metrics"] is False
        assert by_name["fresh"]["step"] == 0
        assert by_name["fresh"]["total"] == 100
        assert by_name["fresh"]["checkpoint"] is None
        assert by_name["fresh"]["resumable"] is False
        assert by_name["queued"]["has_metrics"] is False
        assert by_name["queued"]["checkpoint"] == "step000050.safetensors"
        assert by_name["queued"]["resumable"] is True

        dashboard = workspaces_router.api_dashboard(run="fresh", workspace=WORKSPACE)
        assert dashboard["run"].endswith("/fresh"), dashboard["run"]
        assert dashboard["rows"] == [], dashboard["rows"]
        assert dashboard["samples"] == [], dashboard["samples"]
        listed = {run["run_id"]: run for run in dashboard["runs"]}
        assert listed["fresh"]["has_metrics"] is False, listed["fresh"]
        assert listed["fresh"]["samples"] == 0, listed["fresh"]
        assert listed["fresh"]["resumable"] is False, listed["fresh"]
        assert listed["demo"]["checkpoint"] == "step000050.safetensors", listed["demo"]
        assert listed["demo"]["samples"] == 1, listed["demo"]

        short = workspaces_router.api_dashboard(run="demo", workspace=WORKSPACE)
        assert short["run"].endswith("/demo"), short["run"]
        assert short["rows"], short["rows"]
    finally:
        cleanup_workspace()


def main() -> None:
    # Importing the CLI installs the native-fault handler before any training.
    assert cli_module is not None
    assert faulthandler.is_enabled()
    check_fresh_reset()
    check_fresh_reset_waits_for_lock()
    check_disk_job_signal_fields()
    check_prune_after_resume()
    check_trainer_error_trace()
    check_launcher_trace_and_signal()
    check_queue_checkpoint_filter()
    check_command_resume()
    asyncio.run(check_api_resume())
    check_run_list_and_dashboard()
    print("train resume checks passed")


if __name__ == "__main__":
    main()
