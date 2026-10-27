"""VRAM arithmetic: pick batch sizes from free memory, and size a run from its config.

The first half answers "how many items fit right now" for the cache stages. The
second half, :func:`estimate_vram`, is the opposite question: given a finished
configuration, how much memory will the trainer and a preview need? It is pure --
it reads the config object only, never a file or a CUDA device -- so the WebUI
can call it while the user is still editing the form.

The returned figures are allocator peaks in GiB. The driver adds the CUDA
context and the cuBLAS/flash workspaces on top, so those are returned as a
separate ``context`` term and added once in ``total``.
"""

from __future__ import annotations

from types import SimpleNamespace

import os

import torch

# empirical, fp32 Flux2 VAE at 1024²: each image peaks near 2.5 GiB
VAE_GIB_PER_MEGAPIXEL = 2.5
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


def latent_batch(width: int, height: int, cap: int) -> int:
    """Largest VAE batch that fits the free VRAM.

    ``cap`` is the caller's ceiling -- normally the number of images waiting --
    not a platform assumption. A 4-image ceiling here used to encode a WSL2
    mapping limit as if every machine shared it.
    """
    megapixels = max(0.25, (width * height) / 1e6)
    return auto_batch(VAE_GIB_PER_MEGAPIXEL * megapixels, cap)


def text_batch(cap: int = TEXT_BATCH_CAP) -> int:
    return auto_batch(TEXT_GIB_PER_PROMPT, cap)


# ---------------------------------------------------------------------------
# Peak memory for a run, from its configuration
# ---------------------------------------------------------------------------

# The transformer is 42 DiT blocks. The bf16 footprint below is the block half
# of the model only; the conditioning half (text embedder, modulations, time
# embedding, final projection) is counted separately and stays bf16 unless
# ``vram.quantize_conditioning`` names a storage format.
TOTAL_BLOCKS = 42
BLOCK_WEIGHTS_BF16_GIB = 8.86
CONDITIONING_BF16_GIB = 0.62

# Weight-only storage formats. The factors are of the bf16 footprint: 4-bit
# packs two values per byte plus per-group scales, fp8 stores one byte plus one
# row scale. ``quantize_dit`` also decides how much of the block weights is
# converted: the scope table in quant.py puts nf4 ffn at 4.5 GiB of a 9.5 GiB
# bf16 block half, so the FFN projections carry about 73% of the block weights.
QUANT_FACTORS = {"nf4": 0.28, "fp8": 0.53}
FFN_WEIGHT_SHARE = 0.73

# Activation memory with checkpointing on has two terms. The checkpoint keeps
# every block's input -- the boundary it recomputes from -- until backward:
# 42 blocks x tokens x 3072 hidden x 2 bytes, about 0.96 GiB at 4096 tokens. On
# top of that a block holds its own projections and attention buffers while it
# runs, about 0.85 GiB at the same length, and recomputation cannot remove that
# term. ``vram.offload_activations`` moves the boundaries to pinned host memory,
# so only the block-local term stays on the card.
HIDDEN_SIZE = 3072
BLOCK_TRANSIENT_GIB = 0.85
ACTIVATION_TOKENS = 4096
DEFAULT_MAX_AREA = 1024 * 1024

# 42 blocks x 6 projections at rank 1 is about 2.3M parameters; the rank 16 run
# reported 37.2M.
LORA_PARAMS_PER_RANK = 2.325e6
DEFAULT_LORA_TARGETS = 6
ADAMW_BYTES_PER_PARAM = 8.0
EIGHT_BIT_BYTES_PER_PARAM = 3.0
EMA_BYTES_PER_PARAM = 4.0

# Allocator workspaces, plus the weights a streamed run keeps on the card: the
# block being computed, the one being copied in by the lookahead, and the one
# buffer set the pool holds for the next upload.
WORKSPACE_GIB = 0.5
COMPILE_WORKSPACE_GIB = 0.2
SWAP_WINDOW_BLOCKS = 3

# Sampling keeps no autograd graph, so its activations are smaller than the
# training ones. The VAE decode is the fp32 Flux2 cost already used by
# ``latent_batch``; fp16 convolution halves it.
SAMPLER_GIB_PER_MEGAPIXEL = 0.5
VAE_GIB_PER_MEGAPIXEL = 2.5
ACCEL_LORA_GIB = 0.5

# The CUDA context and the cuBLAS/flash workspaces are outside PyTorch's
# allocator accounting but are on the card all the same.
CUDA_CONTEXT_GIB = 1.5


def _number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _whole(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _quant_scope(vram) -> tuple[float, str]:
    """Multiplier for the block weights, plus a label for the notes."""
    scope = str(getattr(vram, "quantize_dit", "none") or "none").strip().lower()
    if scope not in ("ffn", "all"):
        return 1.0, "bf16"
    dtype = str(getattr(vram, "quantize_dtype", "nf4") or "nf4").strip().lower()
    factor = QUANT_FACTORS.get(dtype)
    if factor is None:
        return 1.0, "bf16"
    share = 1.0 if scope == "all" else FFN_WEIGHT_SHARE
    return 1.0 - share * (1.0 - factor), f"{dtype} {scope}"


def _conditioning_storage(vram) -> tuple[float, str]:
    """Multiplier for the conditioning half, plus its storage label.

    The conditioning control names a storage format rather than a scope, and like
    ``quantize_dtype`` it only applies while the DiT quantization is on: the form
    disables it when ``quantize_dit`` is ``none``, and the trainer keeps the half
    in bf16 in that case.
    """
    scope = str(getattr(vram, "quantize_dit", "none") or "none").strip().lower()
    if scope not in ("ffn", "all"):
        return 1.0, "bf16"
    dtype = str(getattr(vram, "quantize_conditioning", "none") or "none").strip().lower()
    factor = QUANT_FACTORS.get(dtype)
    if factor is None:
        return 1.0, "bf16"
    return factor, dtype


def _resident_blocks(vram) -> int:
    swapped = max(0, min(TOTAL_BLOCKS, _whole(getattr(vram, "blocks_to_swap", 0) or 0)))
    if swapped == 0:
        return TOTAL_BLOCKS
    # Streamed blocks leave the card, but the lookahead and its buffer pool keep
    # three of them resident; stream every block and those three are all that is
    # left.
    return min(TOTAL_BLOCKS, TOTAL_BLOCKS - swapped + SWAP_WINDOW_BLOCKS)


def _checkpointing(vram) -> bool:
    value = getattr(vram, "gradient_checkpointing", True)
    if isinstance(value, str):
        # A config that still says "auto" is normalised on load; treat it as on
        # here too, because streaming cannot run without it.
        text = value.strip().lower()
        if text in ("", "auto"):
            return True
        return text not in ("0", "false", "no", "off")
    return bool(value)


def _max_area(dataset) -> int:
    area = _whole(getattr(dataset, "max_area", 0) or 0)
    return area if area > 0 else DEFAULT_MAX_AREA


def _activation_gib(config) -> tuple[float, int]:
    """Checkpointed activation peak, and the token budget it was sized for."""
    area = _max_area(config.dataset)
    budget = _whole(getattr(config.train, "bin_tokens", 0) or 0)
    if budget <= 0:
        batch = max(1, _whole(getattr(config.train, "batch_size", 1) or 1))
        budget = batch * max(1, area // 256)  # 16x16 latent grid
    tokens = max(1, budget)
    scale = tokens / ACTIVATION_TOKENS
    # fp32 weights would double the boundaries; the trainer's dtypes are bf16 or
    # fp16 (2 bytes) in every preset, and fp32 runs are a debugging case.
    dtype = str(getattr(getattr(config, "model", None), "dtype", "bfloat16") or "bfloat16")
    item_bytes = 4 if dtype.lower() == "float32" else 2
    boundary = TOTAL_BLOCKS * tokens * HIDDEN_SIZE * item_bytes / 2**30
    activation = BLOCK_TRANSIENT_GIB * scale + boundary
    if not _checkpointing(config.vram):
        activation *= 3.0
    elif bool(getattr(config.vram, "offload_activations", False)):
        activation -= boundary
    return activation, tokens


def _lora_params(config) -> tuple[float, int, int]:
    """Trainable parameters: rank and the number of targeted projections."""
    lora = config.lora
    rank = max(0, _whole(getattr(lora, "rank", 0)))
    targets = [str(item) for item in (getattr(lora, "targets", None) or []) if str(item).strip()]
    target_count = len(targets) or DEFAULT_LORA_TARGETS
    layers = set()
    for item in getattr(lora, "layers", None) or []:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < TOTAL_BLOCKS:
            layers.add(index)
    layer_factor = len(layers) / TOTAL_BLOCKS if layers else 1.0
    params = LORA_PARAMS_PER_RANK * rank * (target_count / DEFAULT_LORA_TARGETS) * layer_factor
    return params, target_count, rank


def _optimizer_bytes(config) -> float:
    name = str(getattr(config.train, "optimizer", "adamw") or "adamw").strip().lower()
    eight_bit = name in ("adamw8bit", "8bit") or bool(
        getattr(config.vram, "optimizer_in_8bit", False))
    per_param = EIGHT_BIT_BYTES_PER_PARAM if eight_bit else ADAMW_BYTES_PER_PARAM
    if _number(getattr(config.train, "ema", 0) or 0) > 0:
        per_param += EMA_BYTES_PER_PARAM  # fp32 EMA copy
    return per_param


def _workspace_gib(vram) -> tuple[float, bool]:
    # torch.compile is refused for a streamed run (the copy events cannot be
    # traced) and for an activation offload (saves go through autograd hooks), so
    # its workspace is only charged when the trainer will really compile.
    streaming = int(getattr(vram, "blocks_to_swap", 0) or 0) > 0
    offloading = bool(getattr(vram, "offload_activations", False))
    compiled = bool(getattr(vram, "compile", False)) and not streaming and not offloading
    return WORKSPACE_GIB + (COMPILE_WORKSPACE_GIB if compiled else 0.0), compiled


def _preview_gib(preview, resident_gib: float) -> tuple[float, float, float, float, int]:
    """Sampling peak: resident weights plus what one preview adds on top."""
    width = max(1, _whole(getattr(preview, "width", 1024) or 1024))
    height = max(1, _whole(getattr(preview, "height", 1024) or 1024))
    megapixels = width * height / 1048576.0

    batch = max(0, _whole(getattr(preview, "batch", 0) or 0))
    factor = float(max(1, batch)) if batch > 0 else 1.0
    if _number(getattr(preview, "guidance", 1.0), 1.0) != 1.0:
        factor *= 2.0  # the unconditional branch is skipped only at guidance 1
    sampler = SAMPLER_GIB_PER_MEGAPIXEL * megapixels * factor

    device = str(getattr(preview, "vae_device", "gpu") or "gpu").strip().lower()
    vae = 0.0
    if device != "cpu":
        vae = VAE_GIB_PER_MEGAPIXEL * megapixels
        if str(getattr(preview, "vae_dtype", "fp32") or "fp32").strip().lower() == "fp16":
            vae *= 0.5

    # No stat here: the estimate runs while the form is being edited, and a path
    # that does not exist yet must not fail it.
    lora = ACCEL_LORA_GIB if str(getattr(preview, "lora", "") or "").strip() else 0.0
    return resident_gib + sampler + vae + lora, sampler, vae, lora, batch


def estimate_vram(config) -> dict:
    """Expected allocator peak for one configuration, in GiB.

    ``train`` is a training step and ``preview`` is one sampling pass; both
    include the resident model weights. ``total`` adds the CUDA context once to
    the larger of the two, which is what the driver reports for the process.

    Pure: it reads the config only -- no files, no CUDA -- so the WebUI can call
    it while the form is still being edited.
    """
    vram = config.vram
    resident = _resident_blocks(vram)
    streamed = TOTAL_BLOCKS - resident
    block_factor, quant_label = _quant_scope(vram)
    block_gib = BLOCK_WEIGHTS_BF16_GIB * block_factor * (resident / TOTAL_BLOCKS)
    conditioning_factor, conditioning_label = _conditioning_storage(vram)
    conditioning_gib = CONDITIONING_BF16_GIB * conditioning_factor
    activation_gib, tokens = _activation_gib(config)
    params, target_count, rank = _lora_params(config)
    bytes_per_param = _optimizer_bytes(config)
    workspace_gib, compiled = _workspace_gib(vram)

    train_gib = (block_gib + conditioning_gib + activation_gib
                 + params * bytes_per_param / 2**30 + workspace_gib)
    # A preview can move blocks for its own pass and put them back, so its
    # resident weight term comes from its own count when it sets one.
    preview_swap = int(getattr(config.preview, "blocks_to_swap", -1))
    if preview_swap >= 0:
        preview_resident = _resident_blocks(SimpleNamespace(blocks_to_swap=preview_swap))
        preview_block_gib = BLOCK_WEIGHTS_BF16_GIB * block_factor * (preview_resident / TOTAL_BLOCKS)
    else:
        preview_block_gib = block_gib
    resident_gib = block_gib + conditioning_gib
    preview_gib, sampler_gib, vae_gib, accel_gib, preview_batch = _preview_gib(
        config.preview, preview_block_gib + conditioning_gib)

    train = round(train_gib, 2)
    preview = round(preview_gib, 2)
    total = round(max(train, preview) + CUDA_CONTEXT_GIB, 2)

    notes = [
        (f"{streamed} of {TOTAL_BLOCKS} blocks stream from host memory"
         if streamed else f"all {TOTAL_BLOCKS} blocks are resident"),
    ]
    notes += [
        f"{quant_label}: {block_gib:.2f} GiB of block weights",
        f"conditioning: {conditioning_gib:.2f} GiB {conditioning_label}",
        (f"activations: {activation_gib:.2f} GiB for {tokens} tokens, "
         f"gradient checkpointing {'on' if _checkpointing(vram) else 'off'}"
         + (", block boundaries on host memory"
            if bool(getattr(vram, "offload_activations", False)) else "")),
        (f"lora: {params / 1e6:.1f}M params at rank {rank} x {target_count} targets, "
         f"{bytes_per_param:.0f} B each with optimizer state"),
        f"workspace: {workspace_gib:.1f} GiB" + (", torch.compile on" if compiled else ""),
        (f"preview: {sampler_gib:.2f} GiB sampler + {vae_gib:.2f} GiB VAE decode"
         + (f" + {accel_gib:.2f} GiB adapter" if accel_gib else "")
         + (f", {preview_batch} prompt(s) per forward" if preview_batch else "")),
        f"CUDA context: {CUDA_CONTEXT_GIB:.1f} GiB, added once to the larger figure",
    ]
    optimizer = str(getattr(config.train, "optimizer", "adamw") or "adamw").strip().lower()
    if optimizer in ("muon", "adafactor"):
        notes.append(f"optimizer {optimizer}: state sized as AdamW, which is an upper bound")
    if bool(getattr(config.lora, "include_text_embedder", False)):
        notes.append("text-embedder adapter: parameters not sized separately")

    return {
        "train": train,
        "preview": preview,
        "context": CUDA_CONTEXT_GIB,
        "total": total,
        "resident_blocks": resident,
        "streamed_blocks": streamed,
        "notes": notes,
    }
