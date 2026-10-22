"""Weight-format adapters.

The architecture lives in ``moss_lora.model`` (upstream code). This module only maps
the on-disk layouts used by the shipped repack and the stock Qwen3.5 checkpoint
onto the modules we build, plus the latent packing the reference pipeline expects.
"""

from __future__ import annotations

import glob
import json
import os

import torch
from safetensors.torch import load_file

from .model import MOSSImageTransformer

BN_EPS = 1e-4


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------

def split_fused_qkv(state: dict) -> dict:
    """The shipped repack merges q/k/v into one [3*dim, dim] weight; the reference keeps them apart."""
    if not any(k.endswith(".proj.qkv.weight") for k in state):
        return state
    out = {}
    for key, value in state.items():
        if key.startswith("layers.") and key.endswith(".proj.qkv.weight"):
            q, k, v = value.chunk(3, dim=0)
            base = key[: -len("qkv.weight")]
            out[base + "q.weight"] = q.contiguous()
            out[base + "k.weight"] = k.contiguous()
            out[base + "v.weight"] = v.contiguous()
        else:
            out[key] = value
    return out


def load_transformer(path: str, dtype=torch.bfloat16, device: str | torch.device = "cpu") -> MOSSImageTransformer:
    if os.path.isdir(path):
        state = _load_dir_state(path, "diffusion_pytorch_model")
    else:
        state = load_file(path)
    state = split_fused_qkv(state)
    state.pop("rope.bands", None)  # constant rebuilt by _RoPE
    model = MOSSImageTransformer(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # ``rope.bands`` is a derived constant; the model recomputes it every forward.
    missing = [k for k in missing if k != "rope.bands"]
    if unexpected or missing:
        raise RuntimeError(f"DiT key mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    del state
    return model.eval()


def _load_dir_state(path: str, prefix: str) -> dict:
    index_path = os.path.join(path, f"{prefix}.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        files = sorted({os.path.join(path, name) for name in index["weight_map"].values()})
    else:
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    state = {}
    for file in files:
        state.update(load_file(file))
    return state


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B,32,H,W) -> (B,128,H/2,W/2)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B,128,H,W) -> (B,32,2H,2W)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


def legacy_vae_to_diffusers(state: dict) -> dict:
    """Legacy repacked VAE keys -> diffusers ``AutoencoderKLFlux2``."""
    out = {}
    for key, value in state.items():
        if ".mid.attn_1." in key:
            prefix, _, rest = key.partition(".mid.attn_1.")
            head, _, tail = rest.partition(".")
            name = {"q": "to_q", "k": "to_k", "v": "to_v", "proj_out": "to_out.0", "norm": "group_norm"}.get(head, head)
            if value.ndim == 4:  # legacy 1x1 conv == Linear
                value = value.reshape(value.shape[0], value.shape[1])
            out[f"{prefix}.mid_block.attentions.0.{name}.{tail}"] = value
            continue
        if key.startswith(("encoder.norm_out.", "decoder.norm_out.")):
            side = key.split(".")[0]
            out[f"{side}.conv_norm_out.{key[len(f'{side}.norm_out.'):]}"] = value
            continue
        if key.startswith("encoder.quant_conv."):
            out["quant_conv." + key[len("encoder.quant_conv."):]] = value
            continue
        if key.startswith("decoder.post_quant_conv."):
            out["post_quant_conv." + key[len("decoder.post_quant_conv."):]] = value
            continue
        mapped = False
        for side, blocks, store in (("encoder", "down", "down_blocks"), ("decoder", "up", "up_blocks")):
            prefix = f"{side}.{blocks}."
            if not key.startswith(prefix):
                continue
            index, _, tail = key[len(prefix):].partition(".")
            index = int(index)
            if side == "decoder":  # legacy order lists the highest resolution first
                index = 3 - index
            parts = tail.replace("block.", "").split(".")
            if parts[0] == "downsample":
                out[f"{side}.{store}.{index}.downsamplers.0.conv.{parts[-1]}"] = value
            elif parts[0] == "upsample":
                out[f"{side}.{store}.{index}.upsamplers.0.conv.{parts[-1]}"] = value
            else:
                name = parts[1].replace("nin_shortcut", "conv_shortcut")
                out[f"{side}.{store}.{index}.resnets.{parts[0]}.{name}.{parts[2]}"] = value
            mapped = True
            break
        if mapped:
            continue
        if ".mid.block_1." in key:
            out[key.replace(".mid.block_1.", ".mid_block.resnets.0.")] = value
            continue
        if ".mid.block_2." in key:
            out[key.replace(".mid.block_2.", ".mid_block.resnets.1.")] = value
            continue
        out[key] = value
    return out


def load_vae(path: str, dtype=torch.float16, device: str | torch.device = "cpu"):
    """Return a diffusers ``AutoencoderKLFlux2`` with the reference latent transform."""
    from diffusers import AutoencoderKLFlux2

    if os.path.isdir(path):
        vae = AutoencoderKLFlux2.from_pretrained(path, torch_dtype=dtype)
    else:
        config = {
            "act_fn": "silu", "batch_norm_eps": BN_EPS, "batch_norm_momentum": 0.1,
            "block_out_channels": [128, 256, 512, 512],
            "decoder_block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D"] * 4, "force_upcast": True,
            "in_channels": 3, "latent_channels": 32, "layers_per_block": 2,
            "mid_block_add_attention": True, "norm_num_groups": 32, "out_channels": 3,
            "patch_size": [2, 2], "sample_size": 1024,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "use_post_quant_conv": True, "use_quant_conv": True,
        }
        vae = AutoencoderKLFlux2.from_config(config)
        missing, unexpected = vae.load_state_dict(legacy_vae_to_diffusers(load_file(path)), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"VAE key mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    return vae.eval().requires_grad_(False).to(dtype).to(device)


class LatentCodec:
    """Patchify + BatchNorm normalisation, matching MOSSImagePipeline.__call__."""

    def __init__(self, vae, dtype=torch.float16):
        self.vae = vae
        self.dtype = dtype
        bn = vae.bn
        self.mean = bn.running_mean.detach().float().view(1, -1, 1, 1)
        self.std = (bn.running_var.detach().float() + BN_EPS).sqrt().view(1, -1, 1, 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
        dist = self.vae.encode(images.to(self.vae.dtype)).latent_dist
        latents = dist.sample() if sample else dist.mode()
        latents = pack_latents(latents.float())
        mean = self.mean.to(latents.device, latents.dtype)
        std = self.std.to(latents.device, latents.dtype)
        return ((latents - mean) / std).to(self.dtype)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, chunk: int = 1) -> torch.Tensor:
        mean = self.mean.to(latents.device, latents.dtype)
        std = self.std.to(latents.device, latents.dtype)
        device = next(self.vae.parameters()).device
        z = unpack_latents(latents * std + mean)
        # The fp32 decoder holds several full resolution feature maps per sample,
        # so a three image preview costs ~9 GiB more than one image does. Decode
        # one sample at a time: same output, flat peak.
        if chunk <= 0 or z.shape[0] <= chunk:
            return self.vae.decode(z.to(device=device, dtype=self.vae.dtype)).sample
        parts = [
            self.vae.decode(z[i:i + chunk].to(device=device, dtype=self.vae.dtype)).sample
            for i in range(0, z.shape[0], chunk)
        ]
        return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Text encoder (Qwen3.5-9B, frozen)
# ---------------------------------------------------------------------------
