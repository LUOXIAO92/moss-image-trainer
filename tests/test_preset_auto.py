"""`vram.preset: auto` must resolve to the same knobs the explicit tier sets.

    .venv/bin/python tests/test_preset_auto.py

An explicit tier overwrites every field in `VRAM_PRESETS`, so `preset: 8` and
`preset: auto` on an 8 GB card have to agree. The one field without a "follow
the preset" sentinel is `compile`: it defaults to true, so auto can only ever
turn it off -- which is exactly what the streaming tiers need, because a compiled
block loop that copies weights between host and card re-traces on every shape.

The check also pins the boundary arithmetic (a card one GB below a tier still
gets that tier) and that `detect=False` never touches `torch.cuda`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moss_lora.config as config_module  # noqa: E402
from moss_lora.config import VRAM_PRESETS, Config, apply_preset  # noqa: E402

CARD_FOR = {"32": 31.0, "24": 23.5, "16": 15.5, "12": 11.5, "8": 8.0, "6": 6.0}


def fresh() -> Config:
    config = Config()
    config.vram.preset = "auto"
    return config


def main() -> int:
    for tier, card in CARD_FOR.items():
        preset = VRAM_PRESETS[tier]
        config = apply_preset(fresh(), card)
        assert config.vram.blocks_to_swap == preset["blocks_to_swap"], (tier, config.vram)
        assert config.vram.quantize_dit == preset["quantize_dit"], (tier, config.vram)
        assert config.vram.optimizer_in_8bit == preset["optimizer_in_8bit"], (tier, config.vram)
        assert config.vram.gradient_checkpointing == preset["gradient_checkpointing"], (tier, config.vram)
        if preset["compile"] is False:
            assert config.vram.compile is False, f"{tier} keeps torch.compile on while streaming every block"
        else:
            assert config.vram.compile is True, (tier, config.vram.compile)

    # An explicit tier wins over everything, including compile.
    config = Config()
    config.vram.preset = "8"
    config.vram.compile = True
    config.vram.blocks_to_swap = 3
    config = apply_preset(config)
    assert config.vram.compile is False and config.vram.blocks_to_swap == 42, config.vram

    # Auto leaves deliberate choices alone: only sentinels are filled.
    config = fresh()
    config.vram.quantize_dit = "all"
    config.vram.blocks_to_swap = 5
    config.vram.optimizer_in_8bit = True
    config.vram.gradient_checkpointing = "false"
    config = apply_preset(config, 31.0)
    assert config.vram.quantize_dit == "all", config.vram
    assert config.vram.blocks_to_swap == 5, config.vram
    assert config.vram.optimizer_in_8bit is True, config.vram
    assert config.vram.gradient_checkpointing is False, config.vram

    # A card one GB below a tier still lands on it; below 5 GB falls through to 6.
    assert apply_preset(fresh(), 23.0).vram.blocks_to_swap == VRAM_PRESETS["24"]["blocks_to_swap"]
    assert apply_preset(fresh(), 22.9).vram.quantize_dit == VRAM_PRESETS["16"]["quantize_dit"]
    assert apply_preset(fresh(), 2.0).vram.quantize_dit == VRAM_PRESETS["6"]["quantize_dit"]

    # detect=False must not reach for torch.cuda; without a capacity the config
    # keeps its dataclass defaults.
    def explode() -> float | None:
        raise AssertionError("apply_preset(detect=False) called detect_vram_gb()")

    real = config_module.detect_vram_gb
    config_module.detect_vram_gb = explode
    try:
        config = apply_preset(fresh(), None, detect=False)
    finally:
        config_module.detect_vram_gb = real
    assert config.vram.quantize_dit == "none" and config.vram.blocks_to_swap == -1, config.vram

    print("preset auto checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
