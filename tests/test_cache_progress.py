"""缓存构建的心跳契约：写到工作区，任何人都能读。

不起 GPU、不建缓存，只测写入与投影两条路径。

    .venv/bin/python tests/test_cache_progress.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora import runtime  # noqa: E402
from moss_lora.cache import PROGRESS_NAME, CacheProgress  # noqa: E402
from moss_lora.web.services.cache_cmd import _cache_job  # noqa: E402


def fake_config(workspace: str):
    return types.SimpleNamespace(
        _base_dir=workspace,
        cache=types.SimpleNamespace(dir=os.path.join(workspace, "train_cache")),
    )


def read(workspace: str) -> dict:
    return runtime.read_json(os.path.join(workspace, PROGRESS_NAME)) or {}


def check_writer(tmp):
    workspace = os.path.join(tmp, "ws-writer")
    os.makedirs(workspace)
    progress = CacheProgress(fake_config(workspace))
    path = os.path.join(workspace, PROGRESS_NAME)
    assert progress.path == path, progress.path

    with progress.phase("latent", 4, "img/s") as beat:
        beat.update(2, 5.0)
        mid = read(workspace)
        assert mid["status"] == "running" and mid["phase"] == "latent", mid
        assert mid["done"] == 2 and mid["total"] == 4, mid
        assert mid["pid"] == os.getpid(), mid

    done = read(workspace)
    assert done["status"] == "done" and done["exit_code"] == 0, done
    assert done["done"] == 4 and done["finished"], done
    assert done["history"]["latent"]["complete"] is True, done["history"]

    # 失败也要留下终态，否则读的人只能等 stale
    try:
        with progress.phase("text", 4, "cap/s"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    failed = read(workspace)
    assert failed["status"] == "failed" and failed["exit_code"] == 1, failed
    assert failed["history"]["text"].get("failed") is True, failed["history"]
    print("cache heartbeat     : ok  running / done / failed, atomic records")


def check_continuation(tmp):
    workspace = os.path.join(tmp, "ws-continue")
    os.makedirs(workspace)
    first = CacheProgress(fake_config(workspace))
    with first.phase("latent", 3, "img/s") as beat:
        beat.update(3, 8.0)

    # cache 之后紧接 encode-text：同一份构建，latent 的那条不该归零
    second = CacheProgress(fake_config(workspace))
    assert second.record["history"].get("latent", {}).get("complete") is True, second.record
    assert second.record["started"] == first.record["started"] > 0, second.record["started"]

    # 15 分钟之后的一次构建就不继承了
    stale = runtime.read_json(second.path)
    stale["time"] = time.time() - 3600
    stale["finished"] = time.time() - 3600
    runtime.write_json_atomic(second.path, stale)
    third = CacheProgress(fake_config(workspace))
    assert third.record["history"] == {}, third.record["history"]
    print("phase continuation  : ok  inherited within the window, dropped after it")


def check_projection(tmp):
    workspace = os.path.join(tmp, "ws-projection")
    os.makedirs(workspace)
    assert _cache_job(workspace) is None

    path = os.path.join(workspace, PROGRESS_NAME)
    record = {
        "pid": os.getpid(), "status": "running", "phase": "latent",
        "done": 120, "total": 725, "rate": 9.4, "unit": "img/s",
        "started": time.time() - 30, "time": time.time(), "exit_code": None,
        "history": {},
    }
    runtime.write_json_atomic(path, record)
    job = _cache_job(workspace)
    assert job["state"] == "running" and job["done"] == 120 and job["total"] == 725, job

    # 进程没了、又没有终态：读的人要能看出这是中断
    runtime.write_json_atomic(path, {**record, "pid": 2 ** 22 + 11, "time": time.time() - 120})
    assert _cache_job(workspace)["state"] == "interrupted"

    runtime.write_json_atomic(path, {**record, "status": "done", "phase": "done",
                                     "exit_code": 0, "finished": time.time(),
                                     "time": time.time()})
    assert _cache_job(workspace)["state"] == "done"
    print("cache job projection: ok  running / interrupted / done")


def check_stop(tmp):
    """停止请求：先验进程身份，再发信号；心跳立刻变成 stopping。"""
    import signal
    import subprocess

    from fastapi import HTTPException

    from moss_lora.web.services.cache_cmd import _cache_stop

    workspace = os.path.join(tmp, "ws-stop")
    os.makedirs(workspace)
    path = os.path.join(workspace, PROGRESS_NAME)

    def refused() -> tuple[int, str]:
        try:
            _cache_stop(workspace)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"code": "", "message": str(exc.detail)}
            return exc.status_code, detail.get("code", "")
        return 0, ""

    # 1) 没有心跳 -> 没有可停的任务
    status, code = refused()
    assert status == 404 and code == "err.cache.stop_no_job", (status, code)

    # 2) 心跳指向一个不是缓存进程的 pid（就是本测试进程）-> 拒绝，不发信号
    record = {"pid": os.getpid(), "status": "running", "phase": "latent", "done": 1, "total": 4,
              "rate": 1.0, "unit": "img/s", "started": time.time(), "time": time.time(),
              "cache_dir": os.path.join(workspace, "train_cache"), "exit_code": None, "history": {}}
    runtime.write_json_atomic(path, record)
    status, code = refused()
    assert status == 409 and code == "err.cache.stop_foreign", (status, code)
    assert runtime.read_json(path)["status"] == "running", "a refused stop must not touch the record"

    # 3) 一个带着缓存进程命令行的真进程：信号必须送达，记录必须变成 stopping
    fake = subprocess.Popen(
        ["bash", "-c", 'exec -a "python -m moss_lora.cli cache --config x.yaml" sleep 30'],
        start_new_session=True,
    )
    try:
        runtime.write_json_atomic(path, {**record, "pid": fake.pid, "time": time.time()})
        result = _cache_stop(workspace)
        assert result["ok"] and result["pid"] == fake.pid, result
        assert fake.wait(timeout=10) != 0, "the fake cache process kept running"
        stopped = runtime.read_json(path)
        assert stopped["status"] == "stopping" and stopped["time"] > record["time"], stopped
    finally:
        if fake.poll() is None:
            fake.kill()

    # 4) 已经结束的任务：再点一次要明确说结束，而不是再发一次信号
    runtime.write_json_atomic(path, {**record, "status": "done", "exit_code": 0,
                                     "finished": time.time(), "time": time.time()})
    status, code = refused()
    assert status == 409 and code == "err.cache.stop_finished", (status, code)

    # 5) 进程自己把 KeyboardInterrupt 接住时，终态是 stopped / 130，不是 failed
    progress = CacheProgress(fake_config(workspace))
    try:
        with progress.phase("latent", 4, "img/s"):
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    final = read(workspace)
    assert final["status"] == "stopped" and final["exit_code"] == 130, final
    print("cache stop          : ok  refused for a foreign pid, signals a cache build, stopped=130")


def main():
    tmp = tempfile.mkdtemp(prefix="moss-cache-progress-")
    try:
        check_writer(tmp)
        check_continuation(tmp)
        check_projection(tmp)
        check_stop(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\ncache progress checks passed")


if __name__ == "__main__":
    main()
