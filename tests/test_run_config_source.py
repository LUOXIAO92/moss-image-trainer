"""一个运行只有一份配置：runs/<run>/config.yaml。

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/test_run_config_source.py

覆盖：configs 路由消失、新建运行、读写 run 配置、训练中冻结、启动时读的是
run 目录那份文件，以及代码里不再出现旧配置目录和旧工作区文件。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time

import httpx
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.config import Config  # noqa: E402
from moss_lora.web import jobs as job_registry  # noqa: E402
from moss_lora.web.routers import jobs as jobs_router  # noqa: E402
from moss_lora.web.server import create_app  # noqa: E402
from moss_lora.web.services.workspace import _workspace_root  # noqa: E402

# 拆成两段，免得这个文件自己成为被 grep 的命中项
FORBIDDEN = ("configs" + "/", "workspace" + ".yaml")

WORKSPACE = "zz-run-config-check"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def workspace_dir() -> str:
    return os.path.join(_workspace_root(), WORKSPACE)


def config_dict(steps: int) -> dict:
    base = workspace_dir()
    return {
        "name": "demo-run",
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


def prepare() -> None:
    base = workspace_dir()
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(os.path.join(base, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(base, "train_cache"), exist_ok=True)
    os.makedirs(os.path.join(base, "logs"), exist_ok=True)
    for name in ("dit.safetensors", "vae.safetensors"):
        with open(os.path.join(base, name), "wb") as handle:
            handle.write(b"\0")
    os.makedirs(os.path.join(base, "runs", "demo-run"), exist_ok=True)
    with open(os.path.join(base, "runs", "demo-run", "config.yaml"), "w", encoding="utf-8") as handle:
        handle.write(yaml.safe_dump(config_dict(2000), allow_unicode=True, sort_keys=False))


def heartbeat(state: str, pid: int | None = None) -> None:
    path = os.path.join(workspace_dir(), "runs", "demo-run", "training.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"run": "demo-run", "pid": pid if pid is not None else os.getpid(),
                   "status": state, "step": 10, "total": 2000,
                   "time": time.time(), "started": time.time() - 60}, handle)


def read_run_config(run: str = "demo-run") -> dict:
    path = os.path.join(workspace_dir(), "runs", run, "config.yaml")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


async def check_configs_routes_gone(http: httpx.AsyncClient) -> None:
    for method, path in (
        ("get", f"/api/workspaces/{WORKSPACE}/configs"),
        ("post", f"/api/workspaces/{WORKSPACE}/configs"),
        ("get", f"/api/workspaces/{WORKSPACE}/configs/portrait"),
        ("delete", f"/api/workspaces/{WORKSPACE}/configs/portrait"),
        ("get", "/api/config"),
        ("post", "/api/config"),
    ):
        request = http.build_request(method.upper(), path, json={})
        response = await http.send(request)
        # The SPA mount answers non-GET with 405; either way no configs API exists.
        assert response.status_code in (404, 405), (method, path, response.status_code)


async def check_create_run(http: httpx.AsyncClient) -> None:
    response = await http.post(f"/api/workspaces/{WORKSPACE}/runs", json={"name": "fresh-run"})
    assert response.status_code == 200, response.text
    body = response.json()
    path = os.path.join(workspace_dir(), "runs", "fresh-run", "config.yaml")
    assert os.path.isfile(path), body
    assert body["path"].endswith("runs/fresh-run/config.yaml"), body
    data = read_run_config("fresh-run")
    assert data["name"] == "fresh-run", data
    # Config() defaults, blank paths: nothing seeds it but the dataclass.
    assert data["dataset"]["path"] == "", data
    assert data["model"]["dit"] == "" and data["model"]["vae"] == "", data
    assert data["train"]["steps"] == Config().train.steps, data

    duplicate = await http.post(f"/api/workspaces/{WORKSPACE}/runs", json={"name": "fresh-run"})
    assert duplicate.status_code == 409, duplicate.text

    copied = await http.post(f"/api/workspaces/{WORKSPACE}/runs",
                             json={"name": "copied-run", "from": "demo-run"})
    assert copied.status_code == 200, copied.text
    copied_data = read_run_config("copied-run")
    assert copied_data["train"]["steps"] == 2000, copied_data
    assert copied_data["name"] == "copied-run", copied_data

    missing = await http.post(f"/api/workspaces/{WORKSPACE}/runs",
                              json={"name": "another", "from": "nope"})
    assert missing.status_code == 404, missing.text


async def check_run_config_read_write(http: httpx.AsyncClient) -> None:
    url = f"/api/workspaces/{WORKSPACE}/runs/demo-run/config"
    fetched = await http.get(url)
    assert fetched.status_code == 200, fetched.text
    body = fetched.json()
    assert body["run"] == "demo-run", body
    assert body["path"].endswith("runs/demo-run/config.yaml"), body
    assert yaml.safe_load(body["yaml"])["train"]["steps"] == 2000, body["yaml"]

    edited = yaml.safe_dump({**config_dict(777), "name": "someone-else"}, allow_unicode=True, sort_keys=False)
    saved = await http.put(url, json={"yaml": edited})
    assert saved.status_code == 200, saved.text
    on_disk = read_run_config()
    assert on_disk["train"]["steps"] == 777, on_disk
    # The run directory names the config; a stale name field must not drift.
    assert on_disk["name"] == "demo-run", on_disk

    broken = await http.put(url, json={"yaml": "train: [unclosed"})
    assert broken.status_code == 400, broken.text
    payload = broken.json()
    assert "detail" in payload and "line" in payload and "column" in payload, payload


async def check_frozen_while_running(http: httpx.AsyncClient) -> None:
    url = f"/api/workspaces/{WORKSPACE}/runs/demo-run/config"
    heartbeat("running")
    frozen = await http.put(url, json={"yaml": yaml.safe_dump(config_dict(999))})
    assert frozen.status_code == 409, frozen.text
    detail = frozen.json()["detail"]
    assert detail["code"] == "err.run.frozen", detail
    assert detail["message"], detail
    assert read_run_config()["train"]["steps"] == 777, read_run_config()

    blocked = await http.post("/api/jobs", json={"kind": "cache", "workspace": WORKSPACE,
                                                 "run": "demo-run", "parts": ["latent"]})
    assert blocked.status_code == 409, blocked.text

    blocked_train = await http.post("/api/jobs", json={"kind": "train", "workspace": WORKSPACE,
                                                      "run": "demo-run"})
    assert blocked_train.status_code == 409, blocked_train.text
    assert blocked_train.json()["detail"]["code"] == "err.run.frozen", blocked_train.text

    # 跑完之后不再拦：谁改谁负责
    heartbeat("done")
    allowed = await http.put(url, json={"yaml": yaml.safe_dump(config_dict(888))})
    assert allowed.status_code == 200, allowed.text


async def check_jobs_read_the_run_file(http: httpx.AsyncClient) -> None:
    """POST /api/jobs 带 run= 时，启动用的正是那个文件此刻的内容。"""
    url = f"/api/workspaces/{WORKSPACE}/runs/demo-run/config"
    edited = yaml.safe_dump({**config_dict(4321), "name": "whatever"}, allow_unicode=True, sort_keys=False)
    assert (await http.put(url, json={"yaml": edited})).status_code == 200

    spawned: list[str] = []
    original_run = jobs_router._run
    try:
        jobs_router._run = lambda job: spawned.append(job.config_path)
        response = await http.post("/api/jobs", json={"kind": "cache", "workspace": WORKSPACE,
                                                      "run": "demo-run", "parts": ["latent"]})
        assert response.status_code == 200, response.text
        body = response.json()
        snapshot = body["config_path"]
        snapshot = snapshot if os.path.isabs(snapshot) else os.path.join(ROOT, snapshot)
        assert os.path.isfile(snapshot), body
        with open(snapshot, encoding="utf-8") as handle:
            snapshot_data = yaml.safe_load(handle)
        assert snapshot_data["train"]["steps"] == 4321, snapshot_data
        assert snapshot_data["name"] == "demo-run", snapshot_data
        assert spawned and spawned[0] == snapshot, spawned
    finally:
        jobs_router._run = original_run
        job_registry.JOBS.items.clear()


def run_dirs() -> list[str]:
    base = os.path.join(workspace_dir(), "runs")
    return sorted(name for name in os.listdir(base)
                  if os.path.isdir(os.path.join(base, name)))


async def wait_for_job(spawned: list) -> None:
    deadline = time.time() + 2
    while not spawned and time.time() < deadline:
        await asyncio.sleep(0.01)
    assert spawned, "the train job never reached the launcher"


async def check_train_targets_the_named_run(http: httpx.AsyncClient) -> None:
    """Train the selected run; only a bare YAML allocates a new run id.

    The endpoint is exercised with the launcher stubbed in the router module,
    so no trainer process starts and no GPU is touched.
    """
    runs_dir = os.path.join(workspace_dir(), "runs")
    before = run_dirs()

    spawned: list = []
    original_run = jobs_router._run
    original_summary = jobs_router._cache_summary
    try:
        jobs_router._run = lambda job: spawned.append(job)
        jobs_router._cache_summary = lambda config: {
            "index_ready": True, "items": 1, "missing_text": 0,
        }

        response = await http.post("/api/jobs", json={"kind": "train", "workspace": WORKSPACE,
                                                     "run": "demo-run"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["run_id"] == "demo-run", body
        assert body["name"] == "demo-run", body
        assert body["output_dir"] == os.path.relpath(os.path.join(runs_dir, "demo-run"), ROOT), body
        assert run_dirs() == before, (before, run_dirs())

        await wait_for_job(spawned)
        job = spawned[-1]
        assert job.run_name == "demo-run", job
        assert job.output_dir == os.path.relpath(os.path.join(runs_dir, "demo-run"), ROOT), job
        assert job.log_path == os.path.join(workspace_dir(), "logs", "demo-run.log"), job
        with open(job.config_path, encoding="utf-8") as handle:
            snapshot = yaml.safe_load(handle)
        assert snapshot["name"] == "demo-run", snapshot
        assert os.path.realpath(snapshot["output_dir"]) == os.path.realpath(runs_dir), snapshot

        # Without a run name the endpoint still allocates a timestamped target,
        # and allocation alone must not create the directory.
        job_registry.JOBS.items.clear()
        spawned.clear()
        posted = yaml.safe_dump(config_dict(123), allow_unicode=True, sort_keys=False)
        response = await http.post("/api/jobs", json={"kind": "train", "workspace": WORKSPACE,
                                                     "yaml": posted})
        assert response.status_code == 200, response.text
        body = response.json()
        run_id = body["run_id"]
        assert run_id and run_id != "demo-run", body
        assert re.fullmatch(r"\d{8}-\d{6}(?:-\d+)?", run_id), run_id
        assert body["name"] == run_id, body
        assert body["output_dir"] == os.path.relpath(os.path.join(runs_dir, run_id), ROOT), body
        assert run_id not in os.listdir(runs_dir), body
        assert run_dirs() == before, (before, run_dirs())
        await wait_for_job(spawned)
        assert spawned[-1].run_name == run_id, spawned[-1]
    finally:
        jobs_router._run = original_run
        jobs_router._cache_summary = original_summary
        job_registry.JOBS.items.clear()


async def check_no_configs_directory_in_code() -> None:
    """Trace every tracked file: one config home, no legacy directory."""
    import subprocess

    listing = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout
    offenders = []
    for rel in listing.splitlines():
        if rel == "tests/test_run_config_source.py":  # this file names the strings it greps
            continue
        if not rel.endswith((".py", ".js", ".html", ".md", ".json", ".yaml")):
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if any(token in line for token in FORBIDDEN):
                    offenders.append(f"{rel}:{number}")
    assert not offenders, f"还有旧配置目录或旧工作区文件的残留：{offenders[:10]}"


async def main() -> None:
    prepare()
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60) as http:
            await check_configs_routes_gone(http)
            await check_create_run(http)
            await check_run_config_read_write(http)
            await check_frozen_while_running(http)
            await check_jobs_read_the_run_file(http)
            await check_train_targets_the_named_run(http)
        await check_no_configs_directory_in_code()
    finally:
        shutil.rmtree(workspace_dir(), ignore_errors=True)
    print("run config source checks passed")


if __name__ == "__main__":
    asyncio.run(main())
