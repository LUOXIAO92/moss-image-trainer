"""Text-embedding generation with the local Qwen3.5-9B encoder.

Two entry points, both exposed through ``python -m moss_lora.cli``:

    encode-text    写某个配置的缓存（caption → text_embeds/text_mask，另外生成 preview）
    encode-prompt  把任意提示词编码成一个小 bundle（采样/对比用）

The heavy lifting is ``moss_lora.hf_text.HfTextEncoder``; this module only handles
cache layout, batching and the preview bundle.
"""

from __future__ import annotations

import os
import time

import torch
from safetensors.torch import load_file, save_file

from .config import Config, expand


def _text_encoder(config: Config, text_model: str):
    from .hf_text import HfTextEncoder

    source = expand(text_model or config.model.text_encoder)
    if not source or "${" in source:
        raise SystemExit("text encoder not configured: pass --text-model or set model.text_encoder")
    if not os.path.exists(source):
        raise SystemExit(f"text encoder not found: {source}")
    config_hint = expand(config.model.text_encoder_config or "")
    return HfTextEncoder(source, dtype=torch.bfloat16, config_dir=config_hint)


def run_encode_text(
    config_path: str,
    overwrite: bool = False,
    text_model: str = "",
    quantize: str = "none",
    batch_size: int | None = None,
    preview_only: bool = False,
    log=print,
) -> int:
    config = Config.load(config_path).resolve_paths()
    cache_dir = os.path.abspath(os.path.expanduser(config.cache.dir))
    os.makedirs(cache_dir, exist_ok=True)

    from .cache import CacheBuilder

    # CacheBuilder knows about the regularisation set; a bare scan() would skip it
    builder = CacheBuilder(config, log=log)
    items = builder.items

    def caption(item):
        parts = [config.dataset.caption_prefix, item.caption, config.dataset.caption_suffix]
        return " ".join(p for p in parts if p).strip()

    # 训练集与正则集分目录：路径一律走 CacheBuilder.path_for()，
    # 否则正则集的文本嵌入会写进训练集缓存，训练时读不到（KeyError: text_embeds）
    def path_for(item):
        return builder.path_for(item)

    def cached_caption(path: str) -> str:
        from safetensors import safe_open

        try:
            with safe_open(path, framework="pt") as handle:
                return str((handle.metadata() or {}).get("caption", ""))
        except Exception:
            return ""

    pending = []
    changed = 0
    for item in items:
        path = path_for(item)
        if overwrite or not os.path.exists(path):
            pending.append(item)
            continue
        if "text_embeds" not in load_file(path, device="cpu"):
            pending.append(item)
            continue
        # caption 变了（换了 moss_caption_source、改了前缀/后缀、重写了 sidecar）就得重编码
        cached = cached_caption(path)
        if cached != caption(item):
            if cached:      # 没有元数据的老缓存不算「变了」，只是升级
                changed += 1
            pending.append(item)

    log(f"text: {len(pending)}/{len(items)} captions"
        + (f"（其中 {changed} 条 caption 变了）" if changed else ""))
    # 文本都齐了但 preview 还缺时也要继续跑：preview 只在提示词变更/缺失时需要重做
    preview_path = os.path.join(cache_dir, "preview.safetensors")
    want_preview = bool(config.preview.prompts) and (overwrite or not os.path.exists(preview_path))
    if not pending and not preview_only and not want_preview:
        return 0

    encoder = _text_encoder(config, text_model)
    if quantize != "none":
        log(f"quantize: {encoder.quantize(quantize)} layers -> {quantize}")
    if batch_size is None:
        from .memory import free_gib, text_batch

        fixed = int(getattr(config.cache, "text_batch", 0) or 0)
        if fixed > 0:
            batch_size = fixed
            log(f"encode-text: batch {batch_size} (cache.text_batch, free {free_gib():.1f} GiB)")
        else:
            # Size the batch after the encoder is on the device. Sizing it first read
            # the empty card's 30 GiB, chose 64 prompts, and then asked for activation
            # room next to an 18 GB model -- which is what ran the card out of memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = text_batch()
            log(f"encode-text: batch {batch_size} (free {free_gib():.1f} GiB after loading the encoder)")
    # A bigger batch is summed by a different kernel, so the same caption encodes
    # to a slightly different vector. Stamp the batch into the cache index.
    builder.record_batch("text", batch_size)
    encode = lambda texts: encoder.encode(texts, batch_size=batch_size)  # noqa: E731
    if preview_only:
        pending = []

    pending.sort(key=lambda item: len(caption(item)))
    start = time.time()
    done = 0
    with builder.progress.phase("text", len(pending), "cap/s") as beat:
        beat.update(0, 0.0)
        for offset in range(0, len(pending), max(1, batch_size)):
            batch = pending[offset: offset + max(1, batch_size)]
            embeds_batch, mask_batch = encode([caption(item) for item in batch])
            for index, item in enumerate(batch):
                path = path_for(item)
                data = dict(load_file(path, device="cpu")) if os.path.exists(path) else {}
                data.update({"text_embeds": embeds_batch[index], "text_mask": mask_batch[index]})
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # 元数据必须一起写：下次靠 caption 是否变化来决定要不要重编码
                save_file(data, path, metadata=builder.item_metadata(item))
            done += len(batch)
            rate = done / max(time.time() - start, 1e-6)
            beat.update(done, rate)
            log(f"text {done}/{len(pending)}  {rate:.2f} cap/s")

    if config.preview.prompts:
        prompts = list(config.preview.prompts)
        embeds, mask = encode(prompts)
        negative, negative_mask = encode([config.preview.negative_prompt or ""])
        save_file(
            {"embeds": embeds, "mask": mask, "negative_embeds": negative, "negative_mask": negative_mask},
            os.path.join(cache_dir, "preview.safetensors"),
        )
        log(f"preview prompts: {len(prompts)} cached")
    builder.write_index()
    return 0


def run_encode_prompt(
    config_path: str,
    out: str,
    prompts: list[str],
    negative: str = "",
    text_model: str = "",
    log=print,
) -> int:
    config = Config.load(config_path).resolve_paths()
    encoder = _text_encoder(config, text_model)
    embeds, mask = encoder.encode(prompts)
    negative_embeds, negative_mask = encoder.encode([negative])

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_file(
        {
            "embeds": embeds,
            "mask": mask,
            "negative_embeds": negative_embeds,
            "negative_mask": negative_mask,
        },
        out,
        metadata={"prompts": "\n".join(prompts)},
    )
    log(f"wrote {out}: {len(prompts)} prompts, max {mask.shape[1]} tokens")
    return 0
