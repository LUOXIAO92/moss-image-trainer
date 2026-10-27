"""One seed binds every random draw a training run makes.

    .venv/bin/python tests/test_reproducible_seed.py

``train.seed`` used to seed only the CPU generator that picked batches and
sampled sigma. The flow-matching noise, the LoRA adapter initialisation, and LoRA
dropout read the process-global CPU/CUDA generators, which were never seeded, so
two runs of one config diverged at the first step and no run could be repeated.
This file locks down the contract: seeding is a pure function of the configured
seed, a resume restores the same streams, and the value is validated before a run
starts.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import moss_lora.model as attention_module  # noqa: E402
from moss_lora.config import Config  # noqa: E402
from moss_lora.trainer import seed_runtime  # noqa: E402

CUDA = torch.cuda.is_available()


def check_global_state_is_seeded():
    seed_runtime(1234)
    first = torch.randn(8)
    seed_runtime(1234)
    second = torch.randn(8)
    assert torch.equal(first, second), "same seed drew different CPU noise"
    seed_runtime(1235)
    third = torch.randn(8)
    assert not torch.equal(first, third), "a different seed drew the same CPU noise"
    print("cpu state: reproducible from the seed")


def check_cuda_state_is_seeded():
    if not CUDA:
        print("cuda state: skipped, no device")
        return
    seed_runtime(1234)
    first = torch.randn(8, device="cuda")
    seed_runtime(1234)
    second = torch.randn(8, device="cuda")
    assert torch.equal(first, second), "same seed drew different CUDA noise"
    seed_runtime(1235)
    third = torch.randn(8, device="cuda")
    assert not torch.equal(first, third), "a different seed drew the same CUDA noise"
    print("cuda state: reproducible from the seed")


def check_resume_carries_the_stream():
    """What the checkpoint stores is what a resumed process restores."""
    seed_runtime(7)
    torch.randn(4)  # move the CPU stream forward, as the first steps would
    if CUDA:
        torch.randn(4, device="cuda")
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if CUDA else None
    expected_cpu = torch.randn(4)
    expected_cuda = torch.randn(4, device="cuda") if CUDA else None

    seed_runtime(999)  # a resumed process starts somewhere else entirely
    torch.set_rng_state(cpu_state)
    if CUDA:
        torch.cuda.set_rng_state(cuda_state)
    assert torch.equal(torch.randn(4), expected_cpu), "CPU stream did not resume"
    if CUDA:
        assert torch.equal(torch.randn(4, device="cuda"), expected_cuda), \
            "CUDA stream did not resume"
    print("resume: both streams continue where the checkpoint left them")


def check_validation():
    config = Config()
    assert "err.config.train_seed" not in {p.code for p in config.validate()}

    for bad in ("abc", -1, 2**63, 1.5, True):
        codes = {p.code for p in Config.from_dict({"train": {"seed": bad}}).validate()}
        assert "err.config.train_seed" in codes, (bad, codes)

    # A form field can hand over a numeric string; the trainer converts it.
    config = Config.from_dict({"train": {"seed": "42"}})
    assert "err.config.train_seed" not in {p.code for p in config.validate()}
    print("validation: seed must be a 0..2^63-1 whole number")


def check_attention_backward_is_deterministic():
    """FlashAttention-2 accumulates dq with atomics unless told not to."""
    if not CUDA or not attention_module.flash_attn_available():
        print("attention: skipped, needs CUDA and FlashAttention-2")
        return
    seen: dict = {}
    real = attention_module._flash_attn_varlen

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    attention_module._flash_attn_varlen = spy
    try:
        q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
        attention_module.optimized_attention(q, q, q, 4, cu_seqlens=cu, max_seqlens=256)
    finally:
        attention_module._flash_attn_varlen = real
    assert seen.get("deterministic") is True, seen
    print("attention: varlen backward requests deterministic accumulation")


def check_state_file_restores_both_streams():
    """A resume has to restore the noise stream, not only the batch stream."""
    from moss_lora.trainer import Trainer

    trainer = object.__new__(Trainer)
    trainer.generator = torch.Generator("cpu").manual_seed(5)
    expected_cpu = torch.randn(4, generator=trainer.generator)

    trainer.generator.manual_seed(5)
    data = {"rng.cpu": trainer.generator.get_state()}
    if CUDA:
        torch.cuda.manual_seed(5)
        expected_cuda = torch.randn(4, device="cuda")
        torch.cuda.manual_seed(5)
        data["rng.cuda.0"] = torch.cuda.get_rng_state()

    # Move both streams somewhere else, the way a fresh process would.
    trainer.generator.manual_seed(999)
    if CUDA:
        torch.cuda.manual_seed(999)
    trainer._restore_rng_state(data)

    assert torch.equal(torch.randn(4, generator=trainer.generator), expected_cpu), \
        "the batch stream did not resume"
    if CUDA:
        assert torch.equal(torch.randn(4, device="cuda"), expected_cuda), \
            "the noise stream did not resume"
    print("state file: both rng streams restore")


def main() -> int:
    check_global_state_is_seeded()
    check_cuda_state_is_seeded()
    check_resume_carries_the_stream()
    check_state_file_restores_both_streams()
    check_validation()
    check_attention_backward_is_deterministic()
    print("seed reproducibility checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
