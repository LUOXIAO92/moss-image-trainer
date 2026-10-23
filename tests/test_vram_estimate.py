"""Calibration for the VRAM estimate: five measured runs, plus its accounting rules.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/test_vram_estimate.py

Each expected value below is the ``peak`` a 12-step run recorded at its end: the
larger of the pure training peak and the preview peak, because the run takes one
preview at step 6. The estimator reports those two separately, so the asserted
quantity is ``max(train, preview)``.

The coefficients that make the five rows land are the module constants in
``moss_lora.memory``: block weights 8.86 GiB bf16 (nf4 x0.28, fp8 x0.53),
conditioning 0.62 GiB, activations 1.0 GiB per 4096 image tokens with
checkpointing on (x3 without), LoRA 2.325M params per rank for six targets with
8 B/param AdamW (3 B for the 8-bit optimizer, +4 B with EMA), workspace 0.5 GiB
plus 0.2 with torch.compile, sampler activations 0.5 GiB per megapixel per
forward (x2 when guidance != 1; ``preview.batch`` multiplies it only when set),
VAE decode 2.5 GiB per megapixel on the GPU (x0.5 for fp16, nothing on the CPU),
0.5 GiB for an acceleration LoRA, and 1.5 GiB of CUDA context added once in
``total``.

The five runs are 1024x1024 previews at 50 steps / guidance 3 with the VAE on the
GPU in fp32, ``preview.batch = 0``, three prompts; the turbo row uses 8 steps and
guidance 1. That last row decodes in fp32 as well: the recorded 6.6 GiB matches
the fp32 VAE term, while the fp16 option puts the same preview at 5.35 GiB and
no shared sampler coefficient can lift it back to 6.6 without pushing the
streamed row past 4.3 GiB.

The rest of the file locks down the rules that are easy to break silently: the
context is added exactly once, the streaming path keeps three blocks' weights (window plus pool), a preview
on the CPU is smaller than one on the GPU, guidance 1 skips the unconditional
branch, the batch factor applies only when set, and an adapter path is never
stat-ed.
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.config import Config  # noqa: E402
from moss_lora.memory import estimate_vram  # noqa: E402

TOLERANCE = 0.10
CONTEXT = 1.5

BASE = {
    "name": "estimate-check",
    "model": {"dtype": "bfloat16"},
    "dataset": {"max_area": 1024 * 1024},
    "train": {
        "batch_size": 1,
        "bin_tokens": 0,
        "optimizer": "adamw",
        "ema": 0.0,
        "steps": 12,
    },
    "lora": {
        "rank": 16,
        "targets": ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"],
    },
    "vram": {
        "preset": "32",
        "gradient_checkpointing": True,
        "blocks_to_swap": 0,
        "swap_window": 2,
        "optimizer_in_8bit": False,
        "quantize_dit": "none",
        "quantize_dtype": "nf4",
        "compile": True,
    },
    "preview": {
        "prompts": ["a character, standing", "a character, close-up", "a character, sitting"],
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "guidance": 3.0,
        "batch": 0,
        "vae_device": "gpu",
        "vae_dtype": "fp32",
        "lora": "",
    },
}


def make(**sections) -> Config:
    """A fresh Config with the sections replaced/merged, never shared."""
    data = copy.deepcopy(BASE)
    for section, values in sections.items():
        if isinstance(values, dict):
            data.setdefault(section, {}).update(copy.deepcopy(values))
        else:
            data[section] = copy.deepcopy(values)
    return Config.from_dict(data)


# name, overrides, pure training peak (context only), recorded run peak
CALIBRATION = [
    (
        "bf16 resident",
        {},
        12.2,
        13.4,
    ),
    (
        "nf4 all",
        {"vram": {"quantize_dit": "all", "quantize_dtype": "nf4"}},
        5.7,
        6.9,
    ),
    (
        "fp8 all",
        {"vram": {"quantize_dit": "all", "quantize_dtype": "fp8"}},
        7.3,
        9.0,
    ),
    (
        "nf4 all, 42 blocks streamed",
        {
            "vram": {
                "quantize_dit": "all",
                "quantize_dtype": "nf4",
                "blocks_to_swap": 42,
                "optimizer_in_8bit": True,
                "compile": False,
            },
        },
        3.4,
        3.94,
    ),
    (
        "nf4 all + 8-step turbo preview",
        {
            "vram": {"quantize_dit": "all", "quantize_dtype": "nf4"},
            "preview": {
                "steps": 8,
                "guidance": 1.0,
                # Deliberately not on disk: a preview adapter is sized whether
                # or not the file exists, so the estimate never stats it.
                "lora": "turbo-adapter.safetensors",
            },
        },
        5.7,
        6.6,
    ),
]


def check_calibration() -> list[str]:
    failures = []
    print(f"{'row':36s} {'train':>6s} {'pure':>5s} {'preview':>7s} {'peak':>6s} "
          f"{'target':>6s} {'delta':>7s}")
    for name, overrides, pure, target in CALIBRATION:
        estimate = estimate_vram(make(**overrides))
        peak = max(estimate["train"], estimate["preview"])
        delta = (peak - target) / target
        print(f"{name:36s} {estimate['train']:6.2f} {pure:5.1f} "
              f"{estimate['preview']:7.2f} {peak:6.2f} {target:6.2f} {delta:+7.1%}")
        if abs(delta) > TOLERANCE:
            failures.append(f"{name}: peak {peak:.2f} GiB vs recorded {target:.2f} "
                            f"({delta:+.1%}, tolerance {TOLERANCE:.0%})")
    return failures


def check_shape() -> None:
    estimate = estimate_vram(make())
    assert set(estimate) == {
        "train", "preview", "context", "total",
        "resident_blocks", "streamed_blocks", "notes",
    }, sorted(estimate)
    assert estimate["context"] == CONTEXT
    assert isinstance(estimate["resident_blocks"], int)
    assert isinstance(estimate["streamed_blocks"], int)
    assert isinstance(estimate["notes"], list) and estimate["notes"]
    assert all(isinstance(note, str) and note for note in estimate["notes"])
    assert estimate["train"] > 0 and estimate["preview"] > 0
    expected = round(max(estimate["train"], estimate["preview"]) + CONTEXT, 2)
    assert estimate["total"] == expected, (estimate["total"], expected)


def check_streaming_window() -> None:
    resident = lambda swap: estimate_vram(make(vram={"blocks_to_swap": swap}))["resident_blocks"]
    assert resident(0) == 42
    assert resident(40) == 5, resident(40)  # 2 kept + 2 lookahead + 1 pooled
    assert resident(42) == 3, resident(42)
    # Never more than the whole model, whatever the config asks for.
    assert resident(1) == 42, resident(1)
    assert estimate_vram(make(vram={"blocks_to_swap": 42}))["streamed_blocks"] == 39


def check_preview_decoding() -> None:
    turbo = {
        "vram": {"quantize_dit": "all", "quantize_dtype": "nf4"},
        "preview": {"steps": 8, "guidance": 1.0, "batch": 1,
                    "lora": "turbo-adapter.safetensors"},
    }
    gpu = estimate_vram(make(**{**turbo, "preview": {**turbo["preview"], "vae_device": "gpu"}}))
    cpu = estimate_vram(make(**{**turbo, "preview": {**turbo["preview"], "vae_device": "cpu"}}))
    assert cpu["preview"] < gpu["preview"], (cpu["preview"], gpu["preview"])
    fp16 = estimate_vram(make(**{**turbo, "preview": {
        **turbo["preview"], "vae_device": "gpu", "vae_dtype": "fp16"}}))
    assert fp16["preview"] < gpu["preview"], (fp16["preview"], gpu["preview"])


def check_guidance() -> None:
    default = {"vram": {"quantize_dit": "all", "quantize_dtype": "nf4"},
               "preview": {"vae_device": "gpu", "steps": 8, "batch": 1}}
    skip = estimate_vram(make(**{**default, "preview": {**default["preview"], "guidance": 1.0}}))
    cfg = estimate_vram(make(**{**default, "preview": {**default["preview"], "guidance": 3.0}}))
    assert skip["preview"] < cfg["preview"], (skip["preview"], cfg["preview"])


def check_batch_factor() -> None:
    base = {"vram": {"quantize_dit": "all", "quantize_dtype": "nf4"},
            "preview": {"guidance": 1.0, "vae_device": "cpu", "batch": 0}}
    unset = estimate_vram(make(**base))
    one = estimate_vram(make(**{**base, "preview": {**base["preview"], "batch": 1}}))
    two = estimate_vram(make(**{**base, "preview": {**base["preview"], "batch": 2}}))
    # Unset (all prompts in one pass) and 1 are the same single forward; 2 doubles
    # the sampler's activations, i.e. adds one sampler unit at 1024x1024.
    assert unset["preview"] == one["preview"], (unset["preview"], one["preview"])
    assert two["preview"] > one["preview"], (two["preview"], one["preview"])


def check_adapter_path_not_read() -> None:
    base = {"vram": {"quantize_dit": "all", "quantize_dtype": "nf4"},
            "preview": {"guidance": 1.0, "vae_device": "gpu", "batch": 1}}
    without = estimate_vram(make(**{**base, "preview": {**base["preview"], "lora": ""}}))
    missing = estimate_vram(make(**{**base, "preview": {
        **base["preview"], "lora": os.path.join("no-such-dir", "turbo.safetensors")}}))
    assert missing["preview"] == round(without["preview"] + 0.5, 2), (
        missing["preview"], without["preview"])
    assert any("adapter" in note for note in missing["notes"]), missing["notes"]


def main() -> int:
    failures = check_calibration()
    if failures:
        print("\nCALIBRATION FAILED")
        for failure in failures:
            print("  -", failure)
        return 1
    check_shape()
    check_streaming_window()
    check_preview_decoding()
    check_guidance()
    check_batch_factor()
    check_adapter_path_not_read()
    print("\nvram estimate calibration passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
