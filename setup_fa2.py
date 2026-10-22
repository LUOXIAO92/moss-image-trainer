"""setup 阶段：FlashAttention-2 预编译轮子解析与安装。

为什么需要它
------------
训练要「多条不同长度的序列装进一次前向」就必须有 varlen attention；PyTorch 原生
SDPA 只能靠稠密 block-diagonal mask 表达，而带 mask 就走不了 flash kernel。所以
打包训练的前提是装上真正的 varlen kernel。

FA2 是目前唯一跨代际可用的选择：sm_80/86/89/90/100/120 都有 kernel（上游
``setup.py`` 默认 ``80;90;100;120``），且 varlen 前后向齐全。FA3 只编 sm_90a，消费级
Blackwell（sm_120）跑不了；FA4 虽然给 sm_120 写了专用 kernel，但仍是
beta，且 30/40 系只有未优化的 sm80 移植。

轮子的文件名里编码了全部约束，例如::

    flash_attn-2.8.3+cu130torch2.14-cp314-cp314-linux_x86_64.whl
                   ^^^^^  ^^^^^^^^^  ^^^^^^         ^^^^^^^^^^^^^
                   CUDA    PyTorch    CPython      平台

本脚本读当前环境（解释器 / torch / CUDA / 显卡），挑出唯一匹配的轮子并装进
``.venv``。找不到匹配时不猜、不降级安装，只报告原因。

它属于**环境准备**，不是训练核心：``moss_lora`` 只用 ``flash_attn`` 本身，不 import
这个文件，也不依赖它存在。

用法（``setup.sh`` 末尾会自动调用，也可单独跑）::

    .venv/bin/python setup_fa2.py check          # 只报告环境与匹配结果，不下载
    .venv/bin/python setup_fa2.py install        # 下载 + 安装 + 验证
    .venv/bin/python setup_fa2.py install --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_mod
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = "mjun0812/flash-attention-prebuild-wheels"
API = f"https://api.github.com/repos/{REPO}/releases"
CACHE_FILE = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "moss_lora" / "fa2_release_index.json"
CACHE_TTL = 24 * 3600

# 已知 tag → torch 小版本（API 被限流时的兜底；用 HEAD 探测确认文件存在）
KNOWN_TAGS = {"2.14": "v0.10.0", "2.13": "v0.9.47", "2.12": "v0.9.17"}

WHEEL_RE = re.compile(
    r"^flash_attn-(?P<fa>\d+\.\d+(?:\.\d+)?)"
    r"\+(?P<cuda>cu\d+)torch(?P<torch>\d+\.\d+)"
    r"-(?P<py>cp\d+)-(?P<abi>cp\d+|abi3)-(?P<plat>.+)\.whl$"
)


@dataclass(frozen=True)
class Wheel:
    name: str
    fa: str
    cuda: str
    torch: str
    py: str
    plat: str
    url: str
    size: int = 0

    @property
    def gb(self) -> float:
        return self.size / 1e9


@dataclass
class Env:
    python_tag: str
    torch: str
    cuda: str
    platform: str
    gpu: str
    cc: tuple[int, int] | None
    problems: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def detect() -> Env:
    """读当前解释器 + torch + GPU，得出轮子需要匹配的四个标签。"""
    problems: list[str] = []
    warnings: list[str] = []
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"

    machine = {"x86_64": "x86_64", "AMD64": "x86_64", "aarch64": "aarch64"}.get(
        platform_mod.machine()
    )
    if machine is None:
        problems.append(f"不支持的 CPU 架构：{platform_mod.machine()}")
        mach = platform_mod.machine()
    else:
        mach = machine
    plat = {"linux": f"linux_{mach}", "win32": f"win_{mach}"}.get(sys.platform)
    if plat is None:
        problems.append(f"不支持的操作系统：{sys.platform}")

    torch_ver = cuda = "unknown"
    gpu = "unknown"
    cc: tuple[int, int] | None = None
    try:
        import torch
    except Exception as exc:  # pragma: no cover - 环境问题
        problems.append(f"无法 import torch：{exc}")
        torch = None
    else:
        torch_ver = ".".join(torch.__version__.split(".")[:2])
        if not torch.version.cuda:
            problems.append("torch 是 CPU/ROCm 构建，FA2 轮子需要 CUDA 版本")
        else:
            major, minor = torch.version.cuda.split(".")[:2]
            cuda = f"cu{major}{minor}"
        if not torch.cuda.is_available():
            # 轮子选择不依赖算力，只有「算力过低」才是致命问题；看不到卡时无法判定，
            # 记成警告继续，安装后由 verify() 实测。
            warnings.append("看不到 GPU（容器/沙箱或驱动限制），跳过算力检查")
        else:
            cc = tuple(torch.cuda.get_device_capability(0))
            gpu = torch.cuda.get_device_name(0)
            if cc[0] < 8:
                problems.append(
                    f"{gpu} 是 sm_{cc[0]}{cc[1]}，FA2 只覆盖 sm_80 及以上"
                    "（Ampere/Ada/Hopper/Blackwell）"
                )
    return Env(py, torch_ver, cuda, plat or "unknown", gpu, cc, problems, warnings)


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "moss-lora-fa2"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def release_index(refresh: bool = False) -> list[dict]:
    """取全部 release 及其资产；带 24h 磁盘缓存，避免打爆 GitHub 限流。"""
    if not refresh and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL:
            try:
                return json.loads(CACHE_FILE.read_text())
            except Exception:
                pass
    items: list[dict] = []
    page = 1
    while True:
        try:
            batch = _http_json(f"{API}?per_page=100&page={page}")
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise RuntimeError(
                    "GitHub API 限流（403），稍后用 --refresh 重试，或走兜底探测"
                ) from exc
            raise
        if not batch:
            break
        for rel in batch:
            for asset in rel.get("assets", []):
                items.append(
                    {
                        "tag": rel["tag_name"],
                        "name": asset["name"],
                        "url": asset["browser_download_url"],
                        "size": asset.get("size", 0),
                        "published": rel.get("published_at", ""),
                    }
                )
        page += 1
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(items))
    return items


def parse_wheel(name: str, url: str, size: int = 0) -> Wheel | None:
    m = WHEEL_RE.match(name)
    if not m:
        return None
    if m.group("abi") not in (m.group("py"), "abi3"):
        return None
    return Wheel(
        name=name,
        fa=m.group("fa"),
        cuda=m.group("cuda"),
        torch=m.group("torch"),
        py=m.group("py"),
        plat=m.group("plat"),
        url=url,
        size=size,
    )


def _version_key(ver: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", ver))


def resolve(env: Env, assets: list[dict]) -> tuple[Wheel | None, list[str]]:
    """严格匹配：平台、Python 标签、torch 小版本、CUDA 版本四者都要一致。"""
    notes: list[str] = []
    pool = [
        w
        for a in assets
        if (w := parse_wheel(a["name"], a["url"], a.get("size", 0)))
        and w.plat == env.platform
    ]
    if not pool:
        return None, [f"没有任何 {env.platform} / {env.python_tag} 的轮子"]
    same_torch = [w for w in pool if w.torch == env.torch]
    if not same_torch:
        seen = sorted({f"torch{w.torch}/{w.cuda}" for w in pool})
        notes.append(f"没有 torch {env.torch} 的轮子；仓库里有：{', '.join(seen[:8])}")
        return None, notes
    same_cuda = [w for w in same_torch if w.cuda == env.cuda]
    if not same_cuda:
        seen = sorted({w.cuda for w in same_torch})
        notes.append(f"没有 {env.cuda} 的轮子；同 torch 版本只有：{', '.join(seen)}")
        return None, notes
    same_cuda.sort(key=lambda w: _version_key(w.fa), reverse=True)
    return same_cuda[0], notes


def _head(url: str) -> int | None:
    """探测文件是否存在；存在则返回 Content-Length（可能为 0/未知）。"""
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "moss-lora-fa2"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return None
            return int(resp.headers.get("Content-Length") or 0)
    except Exception:
        return None


def resolve_without_api(env: Env) -> Wheel | None:
    """API 被限流时的兜底：按已知 tag 拼文件名，HEAD 探测。"""
    tag = KNOWN_TAGS.get(env.torch)
    if not tag:
        return None
    for fa in ("2.8.3", "2.7.4"):
        name = (
            f"flash_attn-{fa}+{env.cuda}torch{env.torch}-"
            f"{env.python_tag}-{env.python_tag}-{env.platform}.whl"
        )
        url = f"https://github.com/{REPO}/releases/download/{tag}/{name.replace('+', '%2B')}"
        size = _head(url)
        if size is not None:
            return Wheel(
                name, fa, env.cuda, env.torch, env.python_tag, env.platform, url, size
            )
    return None


def describe(env: Env) -> None:
    print("环境探测")
    print(f"  Python     {sys.version.split()[0]}  ({env.python_tag})")
    print(f"  PyTorch    {env.torch}  CUDA {env.cuda}")
    print(f"  平台       {env.platform}")
    if env.cc:
        print(f"  显卡       {env.gpu}  sm_{env.cc[0]}{env.cc[1]}")
    else:
        print(f"  显卡       {env.gpu}")
    for w in env.warnings:
        print(f"  注意：{w}")
    if env.problems:
        print("  问题：")
        for p in env.problems:
            print(f"    - {p}")


def download(wheel: Wheel, dest: Path) -> Path:
    target = dest / wheel.name
    if wheel.size and target.exists() and target.stat().st_size == wheel.size:
        print(f"  已缓存 {target}")
        return target
    print(f"  下载 {wheel.name}（{wheel.gb:.2f} GB）")
    req = urllib.request.Request(wheel.url, headers={"User-Agent": "moss-lora-fa2"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(target, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total:
                print(
                    f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB  {done * 100 // total:3d}%",
                    end="",
                )
    print()
    if not zipfile.is_zipfile(target):
        raise RuntimeError(f"下载的文件不是合法 wheel：{target}")
    return target


def install(wheel_path: Path) -> None:
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
        str(wheel_path),
    ]
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def verify() -> int:
    try:
        import flash_attn
        import torch
        from flash_attn import flash_attn_varlen_func
    except Exception as exc:
        print(f"  验证失败：{exc}")
        return 1
    print(f"  flash_attn {flash_attn.__version__} 已导入")
    if not torch.cuda.is_available():
        print("  跳过数值验证：当前不可见 GPU")
        return 0
    torch.manual_seed(0)
    lens = [4096, 1024]
    q = torch.randn(sum(lens), 8, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, lens[0], lens[0] + lens[1]], device="cuda", dtype=torch.int32)
    out = flash_attn_varlen_func(q, q, q, cu, cu, max(lens), max(lens)).float()
    ref = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                q[cu[i] : cu[i + 1]].transpose(0, 1).unsqueeze(0).float(),
                q[cu[i] : cu[i + 1]].transpose(0, 1).unsqueeze(0).float(),
                q[cu[i] : cu[i + 1]].transpose(0, 1).unsqueeze(0).float(),
            )
            .squeeze(0)
            .transpose(0, 1)
            for i in range(len(lens))
        ]
    )
    print(f"  数值对比（varlen vs SDPA，bf16）最大绝对误差 {(out - ref).abs().max().item():.4f}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    env = detect()
    describe(env)
    if not env.ok:
        print("\n匹配中止：环境探测有未解决的问题。")
        return 2
    try:
        assets = release_index(refresh=args.refresh)
        print(f"\n发布索引：{len(assets)} 个资产")
        wheel, notes = resolve(env, assets)
    except RuntimeError as exc:
        print(f"\n{exc}\n改用兜底探测…")
        wheel, notes = resolve_without_api(env), []
    for note in notes:
        print(f"  ! {note}")
    if not wheel:
        print("没有可用轮子。")
        return 3
    print(f"\n匹配到：{wheel.name}")
    print(f"  大小 {wheel.gb:.2f} GB\n  {wheel.url}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    env = detect()
    describe(env)
    if not env.ok:
        print("\n不安装：先解决上面列出的问题。")
        return 2
    try:
        assets = release_index(refresh=args.refresh)
        wheel, notes = resolve(env, assets)
    except RuntimeError as exc:
        print(f"{exc}\n改用兜底探测…")
        wheel, notes = resolve_without_api(env), []
    for note in notes:
        print(f"  ! {note}")
    if not wheel:
        print("没有可用轮子，未安装任何东西。")
        return 3
    print(f"\n安装 {wheel.name}")
    if args.dry_run:
        print(f"  [dry-run] {wheel.url}")
        return 0
    with tempfile.TemporaryDirectory(prefix="fa2-") as tmp:
        install(download(wheel, Path(tmp)))
    print("\n验证")
    return verify()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="setup_fa2.py", description=__doc__.splitlines()[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("install", cmd_install)):
        p = sub.add_parser(name)
        p.add_argument("--refresh", action="store_true", help="忽略缓存，重新拉发布索引")
        p.add_argument("--dry-run", action="store_true", help="只解析，不下载不安装")
        p.set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
