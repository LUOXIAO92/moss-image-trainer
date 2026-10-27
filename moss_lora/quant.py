"""Weight-only quantization for QLoRA-style training.

The base weights are frozen, so only the forward value of each Linear matters:
the adapter gradient flows through the input, and bitsandbytes' 4-bit matmul and
our fp8 storage both support that. LoRA adapters are never quantized.

Scope decides which frozen Linears are replaced:

    none  every Linear stays bf16                     9.5 GiB of block weights
    ffn   only ffn_up_gate / ffn_down                 4.5 GiB nf4 / 7.3 GiB fp8
    all   every frozen Linear in ``layers.*``         2.5 GiB nf4 / 4.7 GiB fp8

Storage format decides the trade-off:

    nf4   4-bit NormalFloat, smallest, needs bitsandbytes, most quality risk
    fp8   e4m3 with one scale per output row, twice the size, error is far below
          bf16's own rounding for the dot products the blocks compute

Only ``layers.*`` is touched by the scope arguments. ``text_embedder``, both
modulations, ``time_embedding`` and ``final_linear_video`` stay bf16 unless
:func:`quantize_conditioning` is called explicitly: together they are 0.62 GiB,
and quantizing a per-block scale/shift that every layer multiplies into would
risk the picture for a rounding error, so that stays opt-in.
"""

from __future__ import annotations

import torch
from torch import nn

SCOPES = ("none", "ffn", "all")
DTYPES = ("nf4", "fp8")
FP8_MAX = 448.0
FFN_MARKERS = ("ffn_up_gate", "ffn_down")
# The conditioning half: the text projection plus its two blocks, the shared
# per-block modulation, the head modulation, the time embedding and the final
# projection. ``video_embedder`` is deliberately absent -- it is the latent
# input projection, not conditioning.
CONDITIONING_MARKERS = (
    "text_embedder",
    "shared_block_modulation",
    "video_head_modulation",
    "time_embedding",
    "final_linear_video",
)


class FP8Linear(nn.Linear):
    """Frozen linear stored as fp8 e4m3 with one scale per output row.

    The weight is dequantized to the activation dtype per forward pass. That
    keeps the stored footprint at one byte per parameter and costs one cast of
    the block's largest matrix, which is why this is usable for training and not
    only for inference.

    It subclasses ``nn.Linear`` on purpose: PEFT decides whether an adapter can
    attach with ``isinstance(target, torch.nn.Linear)``, and a plain ``nn.Module``
    here made ``quantize_dtype: fp8`` unusable -- every scope that fp8 replaces
    is also a LoRA target. ``nn.Linear.__init__`` runs on the meta device so the
    subclass wiring costs no allocation; the real weight replaces it below.
    """

    def __init__(self, linear: nn.Linear, compute_dtype: torch.dtype = torch.bfloat16):
        super().__init__(linear.in_features, linear.out_features,
                         bias=linear.bias is not None, device="meta",
                         dtype=compute_dtype)
        self.compute_dtype = compute_dtype
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / FP8_MAX
        # A Parameter, not a buffer, although it never receives a gradient: block
        # streaming walks ``module.parameters()`` to decide what can leave the card,
        # and a buffer would silently stay resident, making blocks_to_swap a no-op
        # for fp8 runs.
        self.weight = nn.Parameter((weight / scale).to(torch.float8_e4m3fn), requires_grad=False)
        self.register_buffer("scale", scale.to(torch.bfloat16), persistent=False)
        self.bias = linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16, torch.float32) else self.compute_dtype
        weight = self.weight.to(dtype) * self.scale.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x.to(dtype), weight, bias)


def _to_linear4bit(linear: nn.Linear, dtype: torch.dtype = torch.bfloat16,
                   device: torch.device | str | None = None) -> nn.Module:
    """Pack one Linear to nf4.

    ``device`` selects the kernel that does the packing: the CUDA kernel runs at
    about 40 GiB/s and the CPU kernel at 0.2, so a 5B model takes 0.2 seconds
    one way and 45 the other. The two kernels agree on every code and disagree
    on the scale of about 0.1% of the blocks by one step, so CPU packing is what
    reproduces the pre-streaming bytes exactly.

    The packed weight then moves to wherever the module lives. Passing the pack
    device to ``from_prequantized`` matters: bnb packs a plain ``Params4bit``
    lazily inside ``.to(device)``, and that pack runs on the destination device,
    so a CPU pack that ends with a move to the card would silently pack on the
    card instead.
    """
    import bitsandbytes as bnb

    pack_device = torch.device(device) if device is not None else torch.device("cpu")

    # Build on the meta device: nn.Linear's constructor initialises a full
    # in_features x out_features matrix that this function immediately replaces,
    # and for the 5B model that random init is the bulk of the packing time.
    quantized = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
        device="meta",
    )
    weight = linear.weight.data.detach().to(device=pack_device, dtype=dtype)
    packed, state = bnb.functional.quantize_4bit(
        weight, quant_type="nf4", compress_statistics=True, blocksize=64)
    quantized.weight = bnb.nn.Params4bit.from_prequantized(
        packed, state.as_dict(), requires_grad=False, device=pack_device, module=quantized)
    if linear.bias is not None:
        quantized.bias = nn.Parameter(
            linear.bias.data.detach().to(device=pack_device, dtype=dtype), requires_grad=False)
    del weight

    target = linear.weight.device or torch.device("cpu")
    same_device = target.type == pack_device.type and (
        pack_device.index is None or target.index == pack_device.index)
    if not same_device:
        quantized = quantized.to(target)
    return quantized


def _frozen_linears(
    model: nn.Module,
    scope: str,
    skip: tuple[str, ...],
    only: tuple[str, ...] | None,
    prefix: str = "layers.",
) -> list[tuple[str, nn.Linear]]:
    """The Linears this scope may replace, in module order.

    ``prefix`` names the part of the model the scope applies to. The whole model
    uses ``"layers."`` so the conditioning half stays accurate; a single block
    passed in on its own (the streamed loader quantizes one block at a time)
    passes ``""`` and every Linear inside it is in scope.
    """
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue
        if any(token in name for token in skip):
            continue
        module.weight.requires_grad = False  # base weights are frozen either way
        if only is not None and not any(marker in name for marker in only):
            continue
        if only is None and prefix and not name.startswith(prefix):
            continue  # the conditioning half of the model stays accurate
        if scope == "ffn" and not any(marker in name for marker in FFN_MARKERS):
            continue
        targets.append((name, module))
    return targets


def _replace(model: nn.Module, targets: list[tuple[str, nn.Linear]], build) -> int:
    for name, module in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, build(module))
    return len(targets)


def quantize_nf4(
    model: nn.Module,
    scope: str = "all",
    dtype: torch.dtype = torch.bfloat16,
    skip: tuple[str, ...] = ("lora_",),
    only: tuple[str, ...] | None = None,
    prefix: str = "layers.",
    device: torch.device | str | None = None,
) -> int:
    """Replace frozen Linears with nf4 ones. Returns how many were replaced."""
    if scope not in ("ffn", "all"):
        return 0
    targets = _frozen_linears(model, scope, skip, only, prefix)
    return _replace(model, targets, lambda module: _to_linear4bit(module, dtype, device))


def quantize_fp8(
    model: nn.Module,
    scope: str = "all",
    dtype: torch.dtype = torch.bfloat16,
    skip: tuple[str, ...] = ("lora_",),
    only: tuple[str, ...] | None = None,
    prefix: str = "layers.",
) -> int:
    """Replace frozen Linears with fp8 e4m3 storage. Returns how many."""
    if scope not in ("ffn", "all"):
        return 0
    targets = _frozen_linears(model, scope, skip, only, prefix)
    return _replace(model, targets, lambda module: FP8Linear(module, compute_dtype=dtype))


def quantize_conditioning(
    model: nn.Module,
    dtype: str = "fp8",
    compute_dtype: torch.dtype = torch.bfloat16,
    skip: tuple[str, ...] = ("lora_",),
) -> int:
    """Replace the conditioning half's frozen Linears. Returns how many were replaced.

    ``dtype`` is the storage format (``fp8`` or ``nf4``), matching the
    ``vram.quantize_conditioning`` control. The conditioning half is
    ``text_embedder`` (projection plus two blocks), ``shared_block_modulation``,
    ``video_head_modulation``, ``time_embedding`` and ``final_linear_video`` --
    16 Linears on the real model. ``layers.*`` and ``video_embedder`` are never
    touched, and anything already carrying a LoRA adapter is skipped.
    """
    if dtype not in DTYPES:
        return 0
    targets = _frozen_linears(model, "all", skip, CONDITIONING_MARKERS)
    if dtype == "nf4":
        return _replace(model, targets, lambda module: _to_linear4bit(module, compute_dtype))
    return _replace(model, targets, lambda module: FP8Linear(module, compute_dtype=compute_dtype))
