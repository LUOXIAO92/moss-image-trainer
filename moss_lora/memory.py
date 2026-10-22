"""Pick batch sizes from the GPU that is actually free.

The VAE encode and the text encoder both scale roughly linearly with batch size, and
the failure mode when we guess too high is not a clean OOM but a WSL mapping stall
(the GPU falls to <1 img/s).  So the default is: take 80% of the free VRAM, divide by
a per-item estimate, and clamp to a tested ceiling.
"""

from __future__ import annotations

import os

import torch

# empirical, fp32 Flux2 VAE at 1024²: batch 4 peaks near 10 GiB,
# larger batches hit the WSL mapping limit and get slower
VAE_GIB_PER_MEGAPIXEL = 2.5
VAE_BATCH_CAP = 4
TEXT_GIB_PER_PROMPT = 0.12  # 4096-token prompt through the 9B encoder
TEXT_BATCH_CAP = 64


def free_gib() -> float:
    """Free VRAM in GiB; ``MOSS_FREE_VRAM_GB`` overrides it for tests."""
    override = os.environ.get("MOSS_FREE_VRAM_GB")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def auto_batch(per_item_gib: float, cap: int, ratio: float = 0.8, minimum: int = 1) -> int:
    """Largest batch that fits ``ratio`` of the free VRAM, clamped to [1, cap]."""
    budget = free_gib() * ratio
    if per_item_gib <= 0:
        return max(minimum, cap)
    fits = int(budget // per_item_gib)
    return max(minimum, min(cap, fits))


def latent_batch(width: int, height: int, cap: int = VAE_BATCH_CAP) -> int:
    megapixels = max(0.25, (width * height) / 1e6)
    return auto_batch(VAE_GIB_PER_MEGAPIXEL * megapixels, cap)


def text_batch(cap: int = TEXT_BATCH_CAP) -> int:
    return auto_batch(TEXT_GIB_PER_PROMPT, cap)
