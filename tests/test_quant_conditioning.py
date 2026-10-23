"""The conditioning half must be quantizable without touching the DiT blocks.

    .venv/bin/python tests/test_quant_conditioning.py

``vram.quantize_conditioning`` names fp8 or nf4 storage for the 16 frozen Linears
outside ``layers.*``: the text projection and its two blocks, both modulations,
the time embedding and the final projection. The toy model below mirrors that
layout one-for-one, so the returned count is the same 16 the real model reports.
The blocks, ``video_embedder`` and anything carrying a LoRA adapter stay bf16.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora import quant  # noqa: E402
from moss_lora.config import Config  # noqa: E402

# Every conditioning Linear, in module order; the real model has the same 16.
CONDITIONING = (
    "text_embedder.proj",
    "text_embedder.blocks.0.qkv",
    "text_embedder.blocks.0.proj",
    "text_embedder.blocks.0.ffn_gate",
    "text_embedder.blocks.0.ffn_up",
    "text_embedder.blocks.0.ffn_down",
    "text_embedder.blocks.1.qkv",
    "text_embedder.blocks.1.proj",
    "text_embedder.blocks.1.ffn_gate",
    "text_embedder.blocks.1.ffn_up",
    "text_embedder.blocks.1.ffn_down",
    "time_embedding.0",
    "time_embedding.2",
    "shared_block_modulation",
    "video_head_modulation",
    "final_linear_video",
)


class _TextBlock(nn.Module):
    """The five Linears of a real text block, plus one LoRA-marked Linear."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24, bias=False, dtype=torch.bfloat16)
        self.proj = nn.Linear(8, 8, dtype=torch.bfloat16)
        self.ffn_gate = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        self.ffn_up = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        self.ffn_down = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
        # PEFT names its adapter projections lora_A/lora_B; those are Linears too
        # and must survive every quantization pass.
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(8, 2, bias=False, dtype=torch.bfloat16)}
        )


class _Block(nn.Module):
    """One DiT block, reduced to the projection the scope assertions need."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_embedder = nn.Linear(4, 8, dtype=torch.bfloat16)
        self.text_embedder = nn.Module()
        self.text_embedder.proj = nn.Linear(8, 8, dtype=torch.bfloat16)
        self.text_embedder.blocks = nn.ModuleList([_TextBlock(), _TextBlock()])
        self.time_embedding = nn.Sequential(
            nn.Linear(8, 8, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(8, 8, dtype=torch.bfloat16),
        )
        self.shared_block_modulation = nn.Linear(8, 48, bias=False, dtype=torch.bfloat16)
        self.video_head_modulation = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        self.final_linear_video = nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([_Block(), _Block()])


def _module(model: nn.Module, path: str) -> nn.Module:
    node = model
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _check_untouched(model: nn.Module) -> None:
    assert isinstance(model.layers[0].q, nn.Linear), "the DiT blocks are out of scope"
    assert isinstance(model.layers[1].q, nn.Linear), "the DiT blocks are out of scope"
    assert isinstance(model.video_embedder, nn.Linear), "video_embedder is not conditioning"
    assert isinstance(model.text_embedder.blocks[0].lora_A["default"], nn.Linear), \
        "LoRA-marked modules must never be replaced"
    assert isinstance(model.text_embedder.blocks[1].lora_A["default"], nn.Linear), \
        "LoRA-marked modules must never be replaced"


def check_fp8():
    model = _Toy()
    replaced = quant.quantize_conditioning(model, "fp8")
    assert replaced == 16, replaced
    for path in CONDITIONING:
        assert isinstance(_module(model, path), quant.FP8Linear), path
    _check_untouched(model)
    print("fp8: 16 conditioning Linears replaced, blocks and LoRA left alone")


def check_nf4():
    import bitsandbytes as bnb

    model = _Toy()
    replaced = quant.quantize_conditioning(model, "nf4")
    assert replaced == 16, replaced
    for path in CONDITIONING:
        assert isinstance(_module(model, path), bnb.nn.Linear4bit), path
    assert all(isinstance(_module(model, path).weight, bnb.nn.Params4bit)
               for path in CONDITIONING), "nf4 weights must stay packed"
    _check_untouched(model)
    print("nf4: 16 conditioning Linears replaced, blocks and LoRA left alone")


def check_unknown_format_is_a_noop():
    model = _Toy()
    assert quant.quantize_conditioning(model, "int8") == 0
    assert quant.quantize_conditioning(model, "none") == 0
    for path in CONDITIONING:
        assert isinstance(_module(model, path), nn.Linear), path
    print("unknown format: nothing replaced")


def check_config_validation():
    code = "err.config.quantize_conditioning"
    assert not [p for p in Config().validate() if p.code == code], \
        "the default none must validate"
    for value in ("none", "fp8", "nf4"):
        config = Config.from_dict({"vram": {"quantize_conditioning": value}})
        assert not [p for p in config.validate() if p.code == code], value
    bad = Config.from_dict({"vram": {"quantize_conditioning": "int8"}})
    problems = [p for p in bad.validate() if p.code == code]
    assert len(problems) == 1, problems
    assert problems[0].params == {"value": "int8"}, problems[0].params
    assert "int8" in problems[0].render(), problems[0].render()
    print(f"config: {code} rejects int8 and leaves the three formats alone")


def check_estimate_follows_format():
    """The VRAM estimate must charge the conditioning half at the chosen format."""
    from moss_lora.memory import estimate_vram

    def estimate(conditioning: str) -> dict:
        config = Config.from_dict({
            "train": {"batch_size": 1, "bin_tokens": 0},
            "vram": {
                "preset": "32",
                "gradient_checkpointing": True,
                "blocks_to_swap": 0,
                "quantize_dit": "all",
                "quantize_dtype": "nf4",
                "quantize_conditioning": conditioning,
                "compile": True,
            },
        })
        return estimate_vram(config)

    bf16, nf4, fp8 = estimate("none"), estimate("nf4"), estimate("fp8")
    assert nf4["train"] < fp8["train"] < bf16["train"], (
        bf16["train"], fp8["train"], nf4["train"])
    assert any("nf4" in note and note.startswith("conditioning:") for note in nf4["notes"]), \
        nf4["notes"]
    print(f"estimate: train {bf16['train']:.2f} -> fp8 {fp8['train']:.2f} "
          f"-> nf4 {nf4['train']:.2f} GiB")


def check_real_model_has_sixteen():
    """The architecture itself, not just the toy, puts 16 Linears in that half."""
    from moss_lora.model import MOSSImageTransformer

    model = MOSSImageTransformer(
        dim=64, ffn_dim=128, num_heads=8, num_layers=2, text_dim=64,
        freq_dim=8, latent_channels=8, dtype=torch.bfloat16,
    )
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(marker in name for marker in quant.CONDITIONING_MARKERS)
    ]
    assert len(names) == 16, names
    print("architecture: the conditioning markers cover 16 Linears")


def check_trainer_runs_the_pass():
    """The trainer applies the pass after the blocks and logs the count."""
    import bitsandbytes as bnb

    from moss_lora.trainer import Trainer

    trainer = Trainer.__new__(Trainer)  # no cache, no weights, no GPU
    trainer.config = Config.from_dict({"vram": {
        "quantize_dit": "all",
        "quantize_dtype": "fp8",
        "quantize_conditioning": "nf4",
    }})
    trainer.dtype = torch.bfloat16
    trainer.model = _Toy()
    logs = []
    trainer.log = logs.append
    trainer._maybe_quantize()
    assert "conditioning: 16 Linears -> nf4" in logs, logs
    assert any("2 frozen Linears -> fp8" in line for line in logs), logs
    assert isinstance(_module(trainer.model, CONDITIONING[0]), bnb.nn.Linear4bit)
    print("trainer: blocks first, then 16 conditioning Linears -> nf4")


def main() -> int:
    check_fp8()
    check_nf4()
    check_unknown_format_is_a_noop()
    check_config_validation()
    check_estimate_follows_format()
    check_real_model_has_sixteen()
    check_trainer_runs_the_pass()
    print("conditioning quantization checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
