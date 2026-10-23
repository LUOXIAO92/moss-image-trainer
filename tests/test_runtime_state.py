"""运行态契约的回归测试：一份磁盘状态，三个读法必须一致。

这里不起服务、不碰 GPU，只测投影层——状态推导、心跳的原子写、作业记录与
心跳的合并、GPU 锁。这三样是 CLI、HTTP 与看板唯一可能分叉的地方。

    .venv/bin/python tests/test_runtime_state.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora import runtime  # noqa: E402

DEAD_PID = 2 ** 22 + 11  # 远超 pid_max 常见上限，永远不可能存在


def check_classify():
    now = 1_000_000.0
    here = os.getpid()

    cases = [
        # (标签, 记录, 期望状态)
        ("live run", {"pid": here, "status": "running", "time": now - 5}, "running"),
        ("dead run", {"pid": DEAD_PID, "status": "running", "time": now - 90}, "interrupted"),
        ("finished ok", {"pid": here, "status": "done", "exit_code": 0, "time": now - 5}, "done"),
        ("finished bad", {"pid": DEAD_PID, "status": "failed", "exit_code": 1, "time": now - 90}, "failed"),
        ("ctrl-c", {"pid": here, "status": "stopped", "exit_code": 130, "time": now - 5}, "stopped"),
        ("stopping, alive", {"pid": here, "status": "stopping", "time": now - 5}, "stopping"),
        ("stopping, gone", {"pid": DEAD_PID, "status": "stopping", "time": now - 90}, "stopped"),
        ("spawn grace", {"status": "starting", "time": now - 5}, "starting"),
        ("never started", {"status": "starting", "time": now - 300}, "interrupted"),
        ("empty", {}, "unknown"),
    ]
    for label, record, expected in cases:
        got = runtime.classify(record, now=now)
        assert got["state"] == expected, f"{label}: {got['state']} != {expected}"

    stale = runtime.classify({"pid": here, "status": "running", "time": now - 120}, now=now)
    assert stale["state"] == "running" and stale["stale"] is True, stale
    print(f"classify            : ok  {len(cases)} states, stale window {runtime.STALE_AFTER:.0f}s")


def check_atomic(tmp: str):
    path = os.path.join(tmp, "beat.json")
    runtime.write_json_atomic(path, {"step": 1, "status": "running"})
    assert runtime.read_json(path)["step"] == 1
    # 反复覆写时，读到的永远是完整的一份
    for step in range(2, 40):
        runtime.write_json_atomic(path, {"step": step, "status": "running"})
        assert runtime.read_json(path) == {"step": step, "status": "running"}
    assert not os.path.exists(path + ".tmp"), "临时文件没有清掉"
    print("atomic heartbeat    : ok  38 rewrites, no partial read")


def check_merge(tmp: str):
    ws = os.path.join(tmp, "ws")
    run_dir = os.path.join(ws, "runs", "run-a")
    os.makedirs(run_dir)
    os.makedirs(os.path.join(ws, "logs"))

    # 训练进程写的心跳
    runtime.write_json_atomic(os.path.join(run_dir, "training.json"), {
        "pid": DEAD_PID, "step": 42, "total": 100, "status": "stopped",
        "phase": "stopped", "run_id": "run-a", "started": 500.0,
        "time": 900.0, "exit_code": 130, "finished": 900.0,
    })
    with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 41, "total": 100, "train": 0.42, "avg": 0.5, "sec": 0.9}) + "\n")
        handle.write(json.dumps({"step": 42, "total": 100, "train": 0.38, "avg": 0.5, "sec": 0.9}) + "\n")
        handle.write(json.dumps({"step": 42, "total": 100, "eval": 0.4}) + "\n")

    # 启动器写的作业记录，指向同一个 run 目录
    with open(os.path.join(ws, "logs", "job-a.json"), "w", encoding="utf-8") as handle:
        json.dump({"id": "job-a", "kind": "train", "workspace": "ws", "run_id": "run-a",
                   "run_dir": run_dir, "pid": DEAD_PID, "started": 500.0,
                   "status": "running", "stage": "train"}, handle)

    rows = runtime.scan_jobs(tmp)
    same = [row for row in rows if (row.get("run_id") == "run-a")]
    assert len(same) == 1, f"同一个 run 出现在 {len(same)} 行里：{same}"
    row = same[0]
    # 训练进程拥有裁决权：它知道退出码，作业记录只知道 pid 没了
    assert row["state"] == "stopped", row["state"]
    assert row["exit_code"] == 130, row["exit_code"]
    # 进度来自心跳和 metrics，而不是作业记录
    assert row["step"] == 42 and (row.get("last") or {}).get("loss") == 0.38, row.get("last")
    print("merge run+record    : ok  one row, trainer verdict, progress filled")


def check_metrics(tmp: str):
    run_dir = os.path.join(tmp, "metrics")
    os.makedirs(run_dir)
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 7, "total": 9, "train": 0.5}) + "\n")
        handle.write(json.dumps({"step": 7, "total": 9, "eval": 0.44}) + "\n")
    last = runtime.last_metrics(run_dir)
    assert last is not None and last.get("train") == 0.5, last  # eval 行不能把 loss 抹成 0
    assert runtime.last_metrics(os.path.join(tmp, "nowhere")) is None
    print("metrics tail        : ok  eval lines skipped")


def check_run_config(tmp: str):
    """The trainer pins the effective config into the run directory."""
    import types

    from moss_lora.config import Config
    from moss_lora.trainer import Trainer

    out_dir = os.path.join(tmp, "runs", "run-b")
    os.makedirs(out_dir)
    config = Config()
    config.name = "run-b"
    config.lora.rank = 24
    trainer = types.SimpleNamespace(
        out_dir=out_dir, config=config, log=lambda message: None,
    )
    Trainer._write_config_snapshot(trainer)
    path = os.path.join(out_dir, "config.yaml")
    assert os.path.isfile(path), "the run directory has no config snapshot"
    written = Config.load(path)
    assert written.name == "run-b" and written.lora.rank == 24, (written.name, written.lora.rank)
    # 同一份配置再写一次不会丢字段
    Trainer._write_config_snapshot(trainer)
    assert Config.load(path).lora.rank == 24
    print("run config snapshot : ok  runs/<run>/config.yaml round-trips")


def check_lock(tmp: str):
    ws = os.path.join(tmp, "lockws")
    os.makedirs(ws)
    assert runtime.check_lock(ws) is None
    holder = runtime.acquire_gpu_lock(ws, kind="train", run_id="run-a", run_dir="/tmp/run-a")
    assert holder["pid"] == os.getpid(), holder
    assert runtime.check_lock(ws)["run_id"] == "run-a"
    try:
        runtime.acquire_gpu_lock(ws, kind="train", run_id="run-b")
    except runtime.GpuBusy as busy:
        assert busy.holder["run_id"] == "run-a", busy.holder
    else:
        raise AssertionError("第二次获取锁没有报忙")
    assert runtime.release_gpu_lock(ws) is True
    assert runtime.check_lock(ws) is None

    # 上一个持有者已经死了：锁可以被接管，不需要人工删文件
    with open(runtime.lock_path(ws), "w", encoding="utf-8") as handle:
        json.dump({"pid": DEAD_PID, "kind": "train", "run_id": "gone",
                   "workspace": "lockws", "run_dir": "", "started": time.time() - 999}, handle)
    taken = runtime.acquire_gpu_lock(ws, kind="train", run_id="run-c")
    assert taken["run_id"] == "run-c", taken
    runtime.release_gpu_lock(ws)
    print("gpu lock            : ok  exclusive, reclaims a dead holder")


def main():
    tmp = tempfile.mkdtemp(prefix="moss-state-")
    try:
        check_classify()
        check_atomic(tmp)
        check_merge(tmp)
        check_metrics(tmp)
        check_run_config(tmp)
        check_lock(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nruntime state checks passed")


if __name__ == "__main__":
    main()
