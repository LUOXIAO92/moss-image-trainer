"""Warm-starting a fresh adapter from an existing LoRA file.

    .venv/bin/python tests/test_lora_init.py

``lora.init_from`` copies another run's adapter into the parameters training is
about to optimize. Three contracts hold it together: the file's rank must equal
the configured rank (the parameters were just created with that shape), a
different alpha is folded into ``lora_B`` so the loaded model computes what the
file computed, and modules the file does not carry keep the fresh init instead of
failing the run.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402

from moss_lora.config import Config  # noqa: E402
from moss_lora.trainer import Trainer  # noqa: E402


def toy_model(rank: int = 4, alpha: int = 8):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Module()
            self.proj.q = nn.Linear(16, 16)
            self.proj.ffn_down = nn.Linear(32, 16)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

    model = get_peft_model(Toy(), LoraConfig(
        r=rank, lora_alpha=alpha, target_modules=r"layers\.\d+\.proj\.q$"))
    return model


def fake_trainer(model, rank: int, alpha: int):
    trainer = object.__new__(Trainer)
    trainer.model = model
    trainer.config = type("Config", (), {
        "lora": type("Lora", (), {"rank": rank, "alpha": alpha, "init_from": ""})()})()
    trainer.log = lambda *_args, **_kwargs: None
    return trainer


def write_adapter(path: str, rank: int, alpha: float, layers=(0, 1), extra: bool = False):
    state = {}
    for layer in layers:
        state[f"lora_unet_layers_{layer}_proj_q.lora_down.weight"] = torch.randn(rank, 16)
        state[f"lora_unet_layers_{layer}_proj_q.lora_up.weight"] = torch.randn(16, rank)
        state[f"lora_unet_layers_{layer}_proj_q.alpha"] = torch.tensor(float(alpha))
    if extra:
        state["lora_unet_layers_9_proj_q.lora_down.weight"] = torch.randn(rank, 16)
        state["lora_unet_layers_9_proj_q.lora_up.weight"] = torch.randn(16, rank)
        state["lora_unet_layers_9_proj_q.alpha"] = torch.tensor(float(alpha))
    save_file(state, path)
    return state


def params(model):
    return dict(model.named_parameters())


def check_copy_and_alpha_scaling(tmp: str):
    model = toy_model(rank=4, alpha=8)
    path = os.path.join(tmp, "adapter.safetensors")
    state = write_adapter(path, rank=4, alpha=16.0)  # file alpha 16, config 8
    fake_trainer(model, rank=4, alpha=8)._load_init_lora(path)

    weights = params(model)
    for layer in (0, 1):
        key = f"base_model.model.layers.{layer}.proj.q"
        down = weights[f"{key}.lora_A.default.weight"]
        up = weights[f"{key}.lora_B.default.weight"]
        assert torch.equal(down, state[f"lora_unet_layers_{layer}_proj_q.lora_down.weight"]), layer
        expected = state[f"lora_unet_layers_{layer}_proj_q.lora_up.weight"] * 2.0
        # 16/8 = 2: the loaded model keeps the file's alpha/rank scaling.
        assert torch.allclose(up, expected, atol=1e-6), (layer, (up - expected).abs().max())
    print("copy: matched modules loaded, B rescaled from the file's alpha")


def check_rank_mismatch_is_fatal(tmp: str):
    model = toy_model(rank=4, alpha=8)
    path = os.path.join(tmp, "rank8.safetensors")
    write_adapter(path, rank=8, alpha=8.0)
    try:
        fake_trainer(model, rank=4, alpha=8)._load_init_lora(path)
    except SystemExit as exc:
        assert "rank mismatch" in str(exc), exc
        print("rank mismatch: rejected with the file and config ranks")
        return
    raise AssertionError("a different rank must fail the load")


def check_missing_modules_stay_fresh(tmp: str):
    model = toy_model(rank=4, alpha=8)
    before = params(model)["base_model.model.layers.1.proj.q.lora_A.default.weight"].clone()
    path = os.path.join(tmp, "partial.safetensors")
    write_adapter(path, rank=4, alpha=8.0, layers=(0,), extra=True)
    fake_trainer(model, rank=4, alpha=8)._load_init_lora(path)
    after = params(model)["base_model.model.layers.1.proj.q.lora_A.default.weight"]
    assert torch.equal(before, after), "a module the file does not carry must stay as initialized"
    print("partial file: unlisted module keeps its init, unknown module is skipped")


def check_missing_file_is_reported():
    config = Config.from_dict({"lora": {"init_from": "/no/such/adapter.safetensors"}})
    problems = config.validate()
    assert any(problem.code == "err.config.field_missing"
               and problem.params.get("field") == "lora.init_from" for problem in problems), problems
    print("validation: a missing init file is reported before the run starts")


def check_inspect_endpoint(tmp: str):
    from moss_lora.web.routers.files import api_lora_inspect

    path = os.path.join(tmp, "inspect.safetensors")
    write_adapter(path, rank=4, alpha=16.0, layers=(0, 1))
    data = api_lora_inspect(path)
    assert data["ranks"] == [4], data
    assert data["alphas"] == [16.0], data
    assert data["modules"] == 2, data
    print("inspect: ranks and alphas reach the start-time warning")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        check_copy_and_alpha_scaling(tmp)
        check_rank_mismatch_is_fatal(tmp)
        check_missing_modules_stay_fresh(tmp)
        check_inspect_endpoint(tmp)
    check_missing_file_is_reported()
    print("lora init checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
