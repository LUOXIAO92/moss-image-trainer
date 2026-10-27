"""Presets are bundles of values, and nothing else decides anything.

    .venv/bin/python tests/test_preset_bundle.py

A preset writes its VRAM values and the preview VAE dtype into the config once,
when a human picks one. The trainer never resolves a preset, so a config that
says `quantize_dit: none` runs unquantized no matter which tier was applied
earlier, and a config with no preset key at all is a complete run.

The old behaviour this file guards against: `vram.preset: auto` detected the card
and filled in whatever was still a sentinel, so the YAML, the form and the run
could disagree. Two sentinels are retired and folded on load ("auto" to on, -1 to
0); everything else is the value in the file.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.config import (PRESET_TIERS, TRANSFORMER_BLOCKS, Config, apply_preset,  # noqa: E402
                              preset_bundle)


def fresh() -> Config:
    return Config()


def check_bundle_shape() -> None:
    for tier in PRESET_TIERS:
        bundle = preset_bundle(tier)
        for key in ("gradient_checkpointing", "blocks_to_swap", "optimizer_in_8bit",
                    "offload_activations", "quantize_dit", "quantize_dtype",
                    "quantize_conditioning", "compile"):
            assert key in bundle, f"{tier}: missing {key}"
        assert "preview" in bundle and "vae_dtype" in bundle["preview"], tier
        assert 0 <= bundle["blocks_to_swap"] <= TRANSFORMER_BLOCKS, tier
        if bundle["blocks_to_swap"] > 0:
            # torch.compile is skipped for a streamed run, so a preset that
            # streams must not claim it: the form would show a setting the
            # trainer silently ignores.
            assert bundle["compile"] is False, f"{tier}: streaming cannot compile"
    # 16 GB and 12 GB stream a slice of the tail and sample in fp16; the small
    # tiers stream everything. These are the differences the tiers exist for.
    assert preset_bundle("8")["blocks_to_swap"] == TRANSFORMER_BLOCKS
    assert preset_bundle("8")["preview"]["vae_dtype"] == "fp16"
    assert preset_bundle("8")["compile"] is False
    assert preset_bundle("32")["blocks_to_swap"] == 0
    assert preset_bundle("32")["preview"]["vae_dtype"] == "fp32"
    assert preset_bundle("32")["compile"] is True
    # 16 GB and 12 GB keep the DiT in bf16 instead of quantizing it.
    assert preset_bundle("16")["quantize_dit"] == "none"
    assert preset_bundle("16")["blocks_to_swap"] == 6
    assert preset_bundle("16")["gradient_checkpointing"] is True
    assert preset_bundle("16")["optimizer_in_8bit"] is False
    assert preset_bundle("16")["compile"] is False
    assert preset_bundle("16")["preview"]["vae_dtype"] == "fp16"
    assert preset_bundle("16")["preview"]["vae_dtype"] == "fp16"
    assert preset_bundle("12")["quantize_dit"] == "none"
    assert preset_bundle("12")["blocks_to_swap"] == 22
    assert preset_bundle("12")["gradient_checkpointing"] is True
    assert preset_bundle("12")["optimizer_in_8bit"] is False
    assert preset_bundle("12")["compile"] is False
    assert preset_bundle("12")["preview"]["vae_dtype"] == "fp16"
    assert preset_bundle("6")["quantize_dit"] == "all"
    # "8GB" and 8 mean the same thing; anything else is a typo the caller made.
    assert preset_bundle("8GB") == preset_bundle("8")
    try:
        preset_bundle("auto")
    except ValueError:
        pass
    else:
        raise AssertionError("the auto preset is gone; it must not resolve")


def check_apply_overwrites() -> None:
    config = fresh()
    config.vram.quantize_dtype = "fp8"
    config.preview.vae_dtype = "fp32"
    apply_preset(config, "8")
    # Every value the bundle owns is written, including one the user had changed:
    # the preset is a stated starting point, not a merge with hidden state.
    assert config.vram.quantize_dtype == "nf4"
    assert config.vram.quantize_dit == "all"
    assert config.vram.blocks_to_swap == 42
    assert config.vram.compile is False
    assert config.preview.vae_dtype == "fp16"
    apply_preset(config, "12")
    assert config.vram.quantize_dit == "none"
    assert config.vram.blocks_to_swap == 22
    assert config.vram.compile is False
    assert config.preview.vae_dtype == "fp16"
    apply_preset(config, "32")
    assert config.vram.blocks_to_swap == 0
    assert config.vram.quantize_dit == "none"
    assert config.vram.compile is True
    assert config.preview.vae_dtype == "fp32"
    # The preview's own residency is a user choice the preset resets to "follow".
    assert config.preview.blocks_to_swap == -1


def check_no_runtime_resolution() -> None:
    """A config that mentions a tier but carries its own values runs those values."""
    config = Config.from_dict({"vram": {"preset": "8", "quantize_dit": "none",
                                        "blocks_to_swap": 0, "compile": True}})
    assert config.vram.quantize_dit == "none"
    assert config.vram.blocks_to_swap == 0
    assert config.vram.compile is True
    # The retired key is ignored rather than silently honoured.
    assert not hasattr(config.vram, "preset")


def check_legacy_sentinels() -> None:
    config = Config.from_dict({"vram": {"gradient_checkpointing": "auto", "blocks_to_swap": -1}})
    assert config.vram.gradient_checkpointing is True
    assert config.vram.blocks_to_swap == 0
    config = Config.from_dict({"vram": {"gradient_checkpointing": "false", "blocks_to_swap": 50}})
    assert config.vram.gradient_checkpointing is False
    assert config.vram.blocks_to_swap == TRANSFORMER_BLOCKS


def check_validation() -> None:
    def base() -> Config:
        config = fresh()
        config.dataset.path = "/nope"
        config.model.dit = "/nope/dit.safetensors"
        config.model.vae = "/nope/vae.safetensors"
        return config

    config = base()
    config.vram.blocks_to_swap = TRANSFORMER_BLOCKS + 1
    config.preview.blocks_to_swap = -2
    codes = {problem.code for problem in config.validate()}
    assert "err.config.blocks_to_swap" in codes, codes
    assert "err.config.preview_blocks_to_swap" in codes, codes
    config = base()
    config.preview.blocks_to_swap = TRANSFORMER_BLOCKS
    codes = {problem.code for problem in config.validate()}
    assert "err.config.preview_blocks_to_swap" not in codes, codes
    # Offloading activations is only defined together with checkpointing: the
    # boundaries it moves are the ones checkpointing keeps.
    config = base()
    config.vram.gradient_checkpointing = False
    config.vram.offload_activations = True
    codes = {problem.code for problem in config.validate()}
    assert "err.config.offload_checkpointing" in codes, codes
    config.vram.gradient_checkpointing = True
    codes = {problem.code for problem in config.validate()}
    assert "err.config.offload_checkpointing" not in codes, codes


def main() -> int:
    check_bundle_shape()
    check_apply_overwrites()
    check_no_runtime_resolution()
    check_legacy_sentinels()
    check_validation()
    print("preset bundle checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
