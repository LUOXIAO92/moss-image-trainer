"""Latent + text-embedding cache.

Caching is what makes the 9B text encoder and the VAE disappear from the training
process: latents and prompt embeddings are computed once, in two separate phases so
only one big model is resident at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from .data import ALIGN, assign_sizes, scan, sizing_spec
from .weights import LatentCodec, load_vae


def _text_len(path: str) -> int:
    """Length of the cached text conditioning, used by the token budget."""
    from safetensors import safe_open

    if not os.path.exists(path):
        return 0
    try:
        with safe_open(path, framework="pt") as f:
            if "text_mask" in f.keys():
                return int(f.get_slice("text_mask").get_shape()[0])
    except Exception:
        pass
    return 0


def reg_cache_dir(config) -> str:
    """正则集缓存目录：默认与训练集缓存并排在工作区里（``reg_cache``）。

    两者都放在 ``workspaces/<ws>/`` 下，各自独立目录，训练时一起读取；
    ``cache.reg_dir`` 显式设置时以配置为准。
    """
    explicit = os.path.expanduser(os.path.expandvars(getattr(config.cache, "reg_dir", "") or ""))
    if explicit:
        return os.path.abspath(explicit)
    cache_dir = os.path.abspath(os.path.expanduser(config.cache.dir))
    # 默认与训练集缓存并排：workspaces/<ws>/train_cache 与 workspaces/<ws>/reg_cache
    return os.path.join(os.path.dirname(cache_dir), "reg_cache")


def item_uid(item) -> str:
    return hashlib.sha1(item.key.encode()).hexdigest()[:16]


def _load_image(path: str, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((width, height), Image.BICUBIC)
        array = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


class CacheBuilder:
    def __init__(self, config, log=print):
        self.config = config
        self.log = log
        base = scan(config.dataset.path, config.dataset.caption_ext, config.dataset.metadata,
                    moss_caption_source=getattr(config.dataset, "moss_caption_source", "tag_caption"))
        reg_path = os.path.expanduser(os.path.expandvars(config.dataset.reg_path or ""))
        if reg_path:
            if not os.path.isdir(reg_path):
                raise FileNotFoundError(f"dataset.reg_path not found: {reg_path}")
            reg = scan(reg_path, config.dataset.caption_ext, "", config.dataset.reg_caption or "",
                       moss_caption_source=getattr(config.dataset, "moss_caption_source", "tag_caption"))
            # 重复次数优先用目录名（<n>_name），reg_repeat 只作为额外倍数
            multiplier = max(1, int(config.dataset.reg_repeat or 1))
            for item in reg:
                item.repeats = max(1, int(item.repeats or 1)) * multiplier
                item.reg = True
            self.log(
                f"regularisation set: {len(reg)} images "
                f"(repeats from folder names x{multiplier})"
            )
            base = base + reg
        self.sizing = sizing_spec(config.dataset.max_area, ALIGN)
        self.items = assign_sizes(base, area=config.dataset.max_area, align=ALIGN)
        self.dir = os.path.abspath(os.path.expanduser(config.cache.dir))
        self.reg_dir = reg_cache_dir(config)
        os.makedirs(self.dir, exist_ok=True)
        # 尺寸规则变了，旧 latent 的宽高就作废：全部重编，文本缓存保持不变
        cached = self._cached_sizing()
        self.sizing_changed = cached != self.sizing
        if self.sizing_changed and cached is not None:
            self.log(
                f"cache sizing changed {cached} -> {self.sizing}: re-encoding every latent"
            )
        if any(getattr(item, "reg", False) for item in self.items):
            os.makedirs(self.reg_dir, exist_ok=True)

    def _cached_sizing(self) -> dict | None:
        import json

        path = os.path.join(self.dir, "index.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return (json.load(handle) or {}).get("sizing")
        except (OSError, ValueError):
            return None

    # -- paths -----------------------------------------------------------
    def path_for(self, item) -> str:
        base = self.reg_dir if getattr(item, "reg", False) else self.dir
        return os.path.join(base, f"{item_uid(item)}.safetensors")

    def _pending(self, tensor: str):
        """Items whose cache file is missing the requested tensor."""
        from safetensors import safe_open

        pending = []
        rebuild = self.config.cache.overwrite or (tensor == "latent" and self.sizing_changed)
        for item in self.items:
            path = self.path_for(item)
            if rebuild or not os.path.exists(path):
                pending.append(item)
                continue
            with safe_open(path, framework="pt") as f:
                if tensor not in f.keys():
                    pending.append(item)
        return pending

    # -- phases ----------------------------------------------------------
    @torch.no_grad()
    def run_latents(self, batch_size: int | None = None, workers: int = 6):
        """Encode latents with CPU image loading overlapped with GPU work.

        The VAE itself does ~10 img/s at 1024²; decoding+resizing 3 MB JPEGs on one
        thread is slower than the GPU, so a small thread pool prefetches batches.
        ``batch_size`` above 4 falls off a cliff on WSL (mapping limit), so it is
        clamped.
        """
        pending = self._pending("latent")
        if not pending:
            self.log("latents: nothing to do")
            return 0
        from .memory import free_gib, latent_batch

        if batch_size is None:
            # one batch size per image size; same-size images are grouped below
            sample = pending[0]
            batch_size = latent_batch(*sample.size)
        batch_size = max(1, min(int(batch_size), 4))
        self.log(f"latents: {len(pending)} images (batch {batch_size}, free {free_gib():.1f} GiB)")
        codec = LatentCodec(load_vae(self.config.model.vae, dtype=torch.float32, device="cuda"))
        groups: dict[tuple[int, int], list] = {}
        for item in pending:
            groups.setdefault(item.size, []).append(item)

        batches = [group[offset: offset + batch_size]
                   for group in groups.values()
                   for offset in range(0, len(group), batch_size)]

        def load(batch):
            return torch.stack([_load_image(i.image, *i.size) for i in batch])

        from concurrent.futures import ThreadPoolExecutor

        start = time.time()
        done = 0
        executor = ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            futures = [executor.submit(load, batch) for batch in batches[:workers]]
            for index, batch in enumerate(batches):
                images = futures[index].result().cuda()
                if index + workers < len(batches):
                    futures.append(executor.submit(load, batches[index + workers]))
                latents = codec.encode(images)
                for item, latent in zip(batch, latents):
                    self._write(item, latent=latent.to(torch.float16).cpu())
                done += len(batch)
                if done % 40 < batch_size or done == len(pending):
                    rate = done / max(time.time() - start, 1e-6)
                    self.log(f"latents {done}/{len(pending)}  {rate:.2f} img/s")
        finally:
            executor.shutdown(wait=False)
        del codec
        torch.cuda.empty_cache()
        return len(pending)

    def run_text(self):
        raise SystemExit(
            "text embeddings are produced by `encode-text` with the local HF encoder:\n"
            "  python -m moss_lora.cli encode-text --config <this yaml>\n"
            "latents and training run in the same product venv."
        )

    # -- helpers ---------------------------------------------------------
    def _caption(self, item) -> str:
        prefix = self.config.dataset.caption_prefix
        suffix = self.config.dataset.caption_suffix
        return " ".join(part for part in (prefix, item.caption, suffix) if part).strip()

    def item_metadata(self, item) -> dict:
        """Image + caption 元数据；写 latent 和写文本都要带上。

        ``encode-text`` 靠这里的 ``caption`` 判断文本缓存是否过期（换了
        ``moss_caption_source`` 或改了前缀/后缀）。
        """
        return {
            "image": item.image,
            "caption": self._caption(item),
            "width": str(item.size[0]),
            "height": str(item.size[1]),
            "repeats": str(item.repeats),
        }

    def _write(self, item, **tensors):
        path = self.path_for(item)
        existing = {}
        if os.path.exists(path):
            from safetensors.torch import load_file

            existing = dict(load_file(path))
        existing.update(tensors)
        save_file(existing, path, metadata=self.item_metadata(item))

    def write_index(self):
        from safetensors import safe_open

        index = {
            "schema": 2,          # 2 = 训练集/正则集分目录，file 为相对路径
            "cache_dir": self.dir,
            "reg_dir": self.reg_dir,
            "caption_source": getattr(self.config.dataset, "moss_caption_source", "tag_caption"),
            # Geometry that produced every latent below; readers compare it with
            # the current config before trusting the shapes.
            "sizing": self.sizing,
            "items": [
                {
                    "file": os.path.relpath(self.path_for(item), self.dir),
                    "image": item.image,
                    "caption": self._caption(item),
                    "width": item.size[0],
                    "height": item.size[1],
                    "repeats": item.repeats,
                    "reg": bool(getattr(item, "reg", False)),
                    "text_len": _text_len(self.path_for(item)),
                }
                for item in self.items
            ],
        }
        path = os.path.join(self.dir, "index.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)
        self.log(f"index: {path} ({len(index['items'])} items)")

    def run(self):
        self.run_latents()
        self.write_index()
