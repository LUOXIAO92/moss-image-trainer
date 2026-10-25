"""The preview OOM ladder climbs one rung at a time, and a real failure survives.

    .venv/bin/python tests/test_preview_fallback.py

A preview must never kill a run that was training fine, but the fallback must
also be cheap: one prompt per pass on the current device first, then fp16 on the
GPU, then the CPU decoder in fp32. No CUDA and no model are involved: the test
builds ``Trainer`` with ``__new__`` and replaces the codec, the sampling call and
the capacity probe.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import moss_lora.config as config_module  # noqa: E402
import moss_lora.memory as memory_module  # noqa: E402
from moss_lora.config import Config  # noqa: E402
from moss_lora.cache import preview_metadata  # noqa: E402
from moss_lora.trainer import Trainer  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)


def make_trainer(root: str, *, batch: int, vae_dtype: str = "fp32",
                 vae_device: str = "gpu") -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.config = Config()
    trainer.config.preview.prompts = ["one", "two", "three", "four"]
    trainer.config.preview.batch = batch
    trainer.config.preview.vae_dtype = vae_dtype
    trainer.config.preview.vae_device = vae_device
    trainer.config.preview.width = trainer.config.preview.height = 1024
    trainer.model = Tiny()
    trainer.device = "cuda"
    trainer.dtype = torch.bfloat16
    trainer.cache_dir = root
    trainer.out_dir = os.path.join(root, "out")
    os.makedirs(trainer.out_dir, exist_ok=True)
    trainer._eager_forward = None
    trainer._codec = None
    trainer.messages: list[str] = []
    trainer.log = trainer.messages.append
    # A bundle the gate accepts: vectors plus the prompts they were encoded from.
    from safetensors.torch import save_file

    torch.manual_seed(0)
    save_file(
        {"embeds": torch.randn(4, 1, 8, dtype=torch.bfloat16),
         "mask": torch.ones(4, 8, dtype=torch.long),
         "negative_embeds": torch.randn(1, 1, 8, dtype=torch.bfloat16),
         "negative_mask": torch.ones(1, 8, dtype=torch.long)},
        os.path.join(root, "preview.safetensors"),
        metadata=preview_metadata(trainer.config.preview.prompts, ""),
    )
    return trainer


def install_fakes(trainer: Trainer, failures: int):
    """Replace the codec and sampler, raising OOM for the first ``failures`` calls."""
    codec_calls: list[dict] = []
    sample_calls: list[tuple[str, str, int]] = []
    preview_lora_calls: list[bool] = []
    error = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    def fake_codec():
        on_cpu = (str(trainer.config.preview.vae_device).lower() == "cpu"
                  or getattr(trainer, "_vae_cpu_fallback", False))
        dtype = "fp32" if on_cpu else trainer._preview_dtype()
        device = "cpu" if on_cpu else "cuda"
        codec_calls.append({"rebuilt": trainer._codec is None,
                            "device": device, "dtype": dtype})
        trainer._codec = types.SimpleNamespace(device=device, dtype=dtype)
        return trainer._codec

    def fake_sample(codec, chunk: int, preview_lora: bool = True):
        sample_calls.append((codec.device, codec.dtype, chunk))
        preview_lora_calls.append(preview_lora)
        if len(sample_calls) <= failures:
            raise error
        return torch.zeros(1, 3, 8, 8)

    trainer._preview_codec = fake_codec
    trainer._sample_preview = fake_sample
    return codec_calls, sample_calls, preview_lora_calls, error


@contextlib.contextmanager
def no_cuda_empty_cache():
    calls: list[None] = []
    real = torch.cuda.empty_cache
    torch.cuda.empty_cache = lambda: calls.append(None)
    try:
        yield calls
    finally:
        torch.cuda.empty_cache = real


@contextlib.contextmanager
def patched_probe(capacity, estimate):
    real_detect = config_module.detect_vram_gb
    real_estimate = memory_module.estimate_vram
    config_module.detect_vram_gb = lambda: capacity
    memory_module.estimate_vram = lambda config: estimate
    try:
        yield
    finally:
        config_module.detect_vram_gb = real_detect
        memory_module.estimate_vram = real_estimate


def check_ladder_order() -> None:
    with tempfile.TemporaryDirectory() as root, no_cuda_empty_cache() as empty_calls:
        trainer = make_trainer(root, batch=4)
        codecs, samples, preview_lora_calls, _ = install_fakes(trainer, failures=2)
        paths = trainer.preview(1)

        # Configured pass -> one prompt per pass -> fp16 VAE on the GPU.
        assert samples == [
            ("cuda", "fp32", 4),
            ("cuda", "fp32", 1),
            ("cuda", "fp16", 1),
        ], samples
        assert [call["device"] for call in codecs] == ["cuda", "cuda", "cuda"], codecs
        assert [call["dtype"] for call in codecs] == ["fp32", "fp32", "fp16"], codecs
        assert all(call["rebuilt"] for call in codecs), \
            "a rung reused the codec instead of rebuilding at the new setting"
        assert preview_lora_calls == [True, True, True], preview_lora_calls
        assert len(empty_calls) >= 2, "a rung did not empty the allocator"

        # The fp16 rung is a decision cached on the trainer, not a config rewrite.
        assert trainer.config.preview.vae_dtype == "fp32", trainer.config.preview.vae_dtype
        assert trainer._preview_dtype() == "fp16", trainer._preview_dtype()
        assert getattr(trainer, "_vae_fp16_fallback", False) is True
        assert not getattr(trainer, "_vae_cpu_fallback", False), \
            "the fp16 rung set the CPU fallback"

        logs = "\n".join(trainer.messages)
        assert "preview: out of memory -> one prompt at a time" in logs, logs
        assert "preview: out of memory -> fp16 VAE on the GPU" in logs, logs
        assert len(paths) == 1 and os.path.exists(paths[0]), paths


def check_baseline_reuses_preview() -> None:
    """Baseline output and no preview adapter, through the same preview path."""
    with tempfile.TemporaryDirectory() as root, no_cuda_empty_cache():
        trainer = make_trainer(root, batch=1)
        _, samples, preview_lora_calls, _ = install_fakes(trainer, failures=0)
        out_dir = os.path.join(root, "baseline")
        paths = trainer.preview(0, out_dir=out_dir, prefix="baseline",
                                preview_lora=False)
        assert samples == [("cuda", "fp32", 1)], samples
        assert preview_lora_calls == [False], preview_lora_calls
        assert paths and os.path.basename(paths[0]) == "baseline_0.png", paths
        assert os.path.dirname(paths[0]) == out_dir, paths[0]


def check_skips() -> None:
    # Default already batch 1: no identical first rung, fp16 comes next.
    with tempfile.TemporaryDirectory() as root, no_cuda_empty_cache():
        trainer = make_trainer(root, batch=1)
        _, samples, _, _ = install_fakes(trainer, failures=1)
        trainer.preview(1)
        assert samples == [("cuda", "fp32", 1), ("cuda", "fp16", 1)], samples

    # fp16 already configured: skip the fp16 rung and go to the CPU decoder.
    with tempfile.TemporaryDirectory() as root, no_cuda_empty_cache():
        trainer = make_trainer(root, batch=4, vae_dtype="fp16")
        _, samples, _, _ = install_fakes(trainer, failures=2)
        trainer.preview(1)
        assert samples == [
            ("cuda", "fp16", 4),
            ("cuda", "fp16", 1),
            ("cpu", "fp32", 1),
        ], samples
        assert trainer.config.preview.vae_dtype == "fp16", trainer.config.preview.vae_dtype
        assert getattr(trainer, "_vae_fp16_fallback", False) is False
        assert getattr(trainer, "_vae_cpu_fallback", False) is True

    # The VAE is already on the CPU: fp16 is pointless there, and the CPU rung
    # would repeat the configured pass. Only the chunk rung remains.
    with tempfile.TemporaryDirectory() as root, no_cuda_empty_cache():
        trainer = make_trainer(root, batch=4, vae_device="cpu")
        _, samples, _, error = install_fakes(trainer, failures=2)
        try:
            trainer.preview(1)
        except torch.cuda.OutOfMemoryError as caught:
            assert caught is error, caught
        else:
            raise AssertionError("a CPU-configured preview with no rungs left swallowed the OOM")
        assert samples == [("cpu", "fp32", 4), ("cpu", "fp32", 1)], samples


def check_total_failure_reraises() -> None:
    with tempfile.TemporaryDirectory() as root, no_cuda_empty_cache():
        trainer = make_trainer(root, batch=4)
        _, samples, _, error = install_fakes(trainer, failures=99)
        try:
            trainer.preview(1)
        except torch.cuda.OutOfMemoryError as caught:
            assert caught is error, "the ladder replaced the original OOM"
        else:
            raise AssertionError("every rung failed but preview() returned")
        assert samples == [
            ("cuda", "fp32", 4),
            ("cuda", "fp32", 1),
            ("cuda", "fp16", 1),
            ("cpu", "fp32", 1),
        ], samples
        assert getattr(trainer, "_vae_cpu_fallback", False) is True


def check_preview_budget_warning() -> None:
    over = {"preview": 7.9, "context": 1.5, "total": 10.0}
    fits = {"preview": 7.0, "context": 1.5, "total": 8.0}

    with tempfile.TemporaryDirectory() as root:
        trainer = make_trainer(root, batch=1)
        with patched_probe(8.0, over):
            trainer._warn_preview_budget()
        assert len(trainer.messages) == 1, trainer.messages
        line = trainer.messages[0]
        assert "estimated 9.4 GB with the fp32 VAE at 1024x1024" in line, line
        assert "lower preview.width/height" in line and "preview.vae_dtype: fp16" in line, line

        # Within half a GB over the card, say nothing.
        trainer.messages.clear()
        with patched_probe(9.0, over):
            trainer._warn_preview_budget()
        assert not trainer.messages, trainer.messages

        trainer.messages.clear()
        with patched_probe(8.0, fits):
            trainer._warn_preview_budget()
        assert not trainer.messages, trainer.messages

        # No capacity detected: no line at all.
        trainer.messages.clear()
        with patched_probe(None, over):
            trainer._warn_preview_budget()
        assert not trainer.messages, trainer.messages

        # An fp16 VAE is already the cheap setting: do not recommend it again.
        trainer.config.preview.vae_dtype = "fp16"
        trainer.messages.clear()
        with patched_probe(8.0, over):
            trainer._warn_preview_budget()
        assert len(trainer.messages) == 1, trainer.messages
        assert "preview.vae_dtype: fp16" not in trainer.messages[0], trainer.messages[0]


def main() -> int:
    check_ladder_order()
    check_baseline_reuses_preview()
    check_skips()
    check_total_failure_reraises()
    check_preview_budget_warning()
    print("preview fallback checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
