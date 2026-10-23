"""Text encoding with a local Qwen3.5-9B HuggingFace checkpoint.

This module loads the plain HuggingFace checkpoint and reproduces the prompt
format used by the reference tokenizer::

    <|im_start|>user\\n{caption}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n

Only the text tower is built (no LM head, no vision tower, no MTP head). Weights can stay
in bf16 or be quantized in memory at load time -- nothing is written back to disk.

``directory`` may also be a single ``.safetensors`` file: ConvRot int8 weights are
dequantized on load (``convrot``), and a plain packed HF state dict works as-is.
In that case the config/tokenizer come from the file's directory, or from ``MOSS_TE_CONFIG``.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

PROMPT = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"
PAD_ID = 248044
MAX_TOKENS = 4096
WEIGHT_PREFIX = "model.language_model."
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
QUANT_MODES = ("none", "fp8", "nf4_ffn", "nf4")


def _weights_files(source: str, config_hint: str = "") -> tuple[list[str], str]:
    """Return (files, config_dir) for a HF directory or a single weights file."""
    if os.path.isdir(source):
        return _shard_files(source), source
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    config_dir = os.path.dirname(os.path.abspath(source))
    # a packed file often ships next to its config; otherwise reuse the HF dir
    if config_hint and os.path.isdir(config_hint):
        config_dir = config_hint
    elif not os.path.exists(os.path.join(config_dir, "config.json")):
        fallback = os.environ.get("MOSS_TE_CONFIG", "")
        if fallback and os.path.isdir(fallback):
            config_dir = fallback
        else:
            # shipped with the product: Qwen tokenizer + text-tower config, so a
            # single .safetensors is enough to encode
            config_dir = ASSETS_DIR
    return [source], config_dir


def _shard_files(directory: str) -> list[str]:
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
        return sorted({os.path.join(directory, name) for name in index["weight_map"].values()})
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".safetensors")
    )


def _iter_raw_weights(source: str, dtype: torch.dtype = torch.bfloat16, config_hint: str = "",
                      device: str = "cpu"):
    """Yield ``(name, tensor, scale, marker)`` straight from the checkpoint.

    ``scale`` is the ConvRot per-row scale for int8 weights and ``None`` for
    everything else, so the caller decides whether to dequantise now or keep the
    int8 tensor on the device and dequantise per forward pass.
    """
    from safetensors import safe_open

    from . import convrot

    files, _config_dir = _weights_files(source, config_hint)
    for path in files:
        with safe_open(path, "pt") as handle:
            keys = list(handle.keys())
            quantized = convrot.is_quantized(keys)
            markers: dict[str, dict] = {}
            for key in keys:
                if key.endswith(".comfy_quant"):
                    markers[key[: -len(".comfy_quant")]] = convrot.quant_marker(handle.get_tensor(key))
                elif key.endswith("_scale"):
                    continue
                else:
                    name = convrot.normalize_key(key)
                    if name is None:
                        continue
                    tensor = handle.get_tensor(key)
                    if tensor.dtype == torch.int8 and quantized and key.endswith(".weight"):
                        base = key[: -len(".weight")]
                        scale_key = base + ".weight_scale"
                        scale = handle.get_tensor(scale_key) if scale_key in keys else torch.tensor(1.0)
                        yield name, tensor.to(device), scale.to(device), markers.get(base, {})
                    elif tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                        yield name, tensor.to(device=device, dtype=dtype), None, {}


def _iter_text_weights(source: str, dtype: torch.dtype = torch.bfloat16, config_hint: str = "",
                       device: str = "cpu"):
    """Yield ``(name, tensor)`` with every tensor dequantised to ``dtype``."""
    from . import convrot

    for name, tensor, scale, marker in _iter_raw_weights(source, dtype=dtype,
                                                         config_hint=config_hint, device=device):
        if scale is None:
            yield name, tensor
        else:
            yield name, convrot.dequantize(tensor, scale, 256, marker).to(dtype)


def _load_text_weights(source: str, dtype: torch.dtype = torch.bfloat16, config_hint: str = "",
                       device: str = "cpu") -> dict[str, torch.Tensor]:
    """The whole text tower as one dict. Prefer ``_iter_raw_weights`` for a 9B model."""
    return dict(_iter_text_weights(source, dtype=dtype, config_hint=config_hint, device=device))


def load_text_model(directory: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
                    config_dir: str = ""):
    """Build Qwen3_5TextModel from the local checkpoint (text tower only)."""
    from transformers import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    from .hf_text import _weights_files  # noqa: F811 - keep the helper local to this module

    _files, resolved_config = _weights_files(directory, config_dir)
    if os.path.basename(resolved_config) == "assets":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        with open(os.path.join(ASSETS_DIR, "qwen35_9b_text_config.json"), encoding="utf-8") as handle:
            config = Qwen3_5TextConfig(**json.load(handle))
    else:
        config = Qwen3_5Config.from_pretrained(resolved_config, local_files_only=True).text_config
    build_on = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    # Build on meta: it allocates nothing, and every tensor is then placed exactly
    # once -- the 200 linear layers as int8 (6.4 GiB), the embedding and the norms
    # as bf16 (1.9 GiB). Dequantising the whole tower up front doubles that to
    # 14.8 GiB for no benefit.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = Qwen3_5TextModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    # RoPE frequencies are buffers filled in __init__, so on meta they hold nothing.
    # Recompute them on the device rather than leaving uninitialised memory.
    inv_freq, _scaling = model.rotary_emb.compute_default_rope_parameters(config, device=build_on)
    model.rotary_emb.inv_freq = inv_freq
    model.rotary_emb.original_inv_freq = inv_freq.clone()

    from . import convrot
    from .quant_linear import QuantLinear

    quantized = 0
    missing = []
    with torch.no_grad():
        for name, tensor, scale, marker in _iter_raw_weights(directory, dtype=dtype,
                                                             config_hint=config_dir, device=build_on):
            module_path, _, leaf = name.rpartition(".")
            try:
                parent = model.get_submodule(module_path)
            except AttributeError:
                missing.append(name)
                del tensor
                continue
            if scale is not None:
                holder_path, _, holder_name = module_path.rpartition(".")
                container = model.get_submodule(holder_path) if holder_path else model
                if isinstance(getattr(container, holder_name, None), nn.Embedding):
                    # The embedding is looked up by row, so dequantising it per
                    # forward would cost 3.9 GiB of float32 every batch. Keep it
                    # bf16 (1.9 GiB) and quantise only the linear layers.
                    parent.weight = nn.Parameter(
                        convrot.dequantize(tensor, scale, 256, marker).to(dtype), requires_grad=False)
                else:
                    setattr(container, holder_name, QuantLinear(tensor, scale, marker))
                    quantized += 1
            elif isinstance(parent, nn.Embedding) and leaf == "weight":
                parent.weight = nn.Parameter(tensor, requires_grad=False)
            elif leaf in parent._parameters:
                parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
            else:
                parent._buffers[leaf] = tensor
            del tensor
    if missing:
        raise RuntimeError(f"unexpected text-encoder weights: {missing[:4]}")
    still_meta = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
                  if value is not None and value.is_meta]
    if still_meta:
        raise RuntimeError(f"text-encoder weights left on meta: {still_meta[:4]}")
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _quantize(model: nn.Module, mode: str) -> int:
    """Quantize in memory; never touches disk. Call before moving to the device if possible."""
    import moss_lora.quant as quant

    if mode == "fp8":
        replaced = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and not name.endswith("embed_tokens"):
                parent_name, _, leaf = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, leaf, quant.FP8Linear(module))
                replaced += 1
        return replaced
    if mode == "nf4_ffn":
        return quant.quantize_nf4(model, "all", dtype=next(model.parameters()).dtype,
                                  only=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    if mode == "nf4":
        return quant.quantize_nf4(model, "all", dtype=next(model.parameters()).dtype)
    return 0


class HfTextEncoder:
    """Prompt encoder matching the reference tokenizer output layout."""

    def __init__(self, directory: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
                 config_dir: str = ""):
        from transformers import AutoTokenizer

        self.device = device
        self.dtype = dtype
        _files, resolved = _weights_files(directory, config_dir)
        if os.path.basename(resolved) == "assets":
            resolved = os.path.join(ASSETS_DIR, "qwen35_tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
        self.model = load_text_model(directory, dtype=dtype, device=device, config_dir=config_dir)

    def quantize(self, mode: str = "none") -> int:
        if mode not in QUANT_MODES:
            raise ValueError(f"unknown quantization {mode!r}")
        return _quantize(self.model, mode)

    @torch.no_grad()
    def encode(self, prompts: list[str], max_tokens: int = MAX_TOKENS, batch_size: int = 1,
               group_by_length: bool = False, bucket_step: int = 32):
        """Encode prompts with right padding and an attention mask.

        ``batch_size=1`` encodes each prompt alone. Larger batches change results by
        ~0.5 max abs on vectors with norm ~715 (cos 0.99985) purely from batching the
        kernels -- padding does not leak: padding a prompt to 211 tokens moves the
        result no more than a batch of two equal-length prompts does.
        """
        if not prompts:
            return torch.zeros(0, 0, 4096, device=self.device), torch.zeros(0, 0, dtype=torch.bool, device=self.device)
        texts = [PROMPT.format(text=prompt) for prompt in prompts]
        groups: list[list[int]] = []
        if batch_size > 1 and group_by_length:
            # only worth it when lengths cluster; with varied lengths bucketing leaves
            # few prompts per bucket, so batches shrink while plain padding keeps them full
            # bucket lengths like the image side buckets sizes: pad up to the next
            # multiple of bucket_step, group by that, so padding waste is bounded
            by_length: dict[int, list[int]] = {}
            for position, text in enumerate(texts):
                length = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
                bucket = max(bucket_step, ((length + bucket_step - 1) // bucket_step) * bucket_step)
                by_length.setdefault(bucket, []).append(position)
            for bucket, positions in sorted(by_length.items()):
                for start in range(0, len(positions), batch_size):
                    groups.append((bucket, positions[start : start + batch_size]))
        else:
            groups = [(None, list(range(start, min(start + max(1, batch_size), len(texts)))))
                      for start in range(0, len(texts), max(1, batch_size))]
        chunks = []
        for bucket, group in groups:
            texts_in_group = [texts[position] for position in group]
            batch = self.tokenizer(
                texts_in_group,
                padding="max_length" if bucket else "longest",
                max_length=min(bucket or max_tokens, max_tokens),
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            hidden = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].bool().to(self.device),
            ).last_hidden_state.float()
            chunks.append((group, hidden, batch["attention_mask"].bool().to(self.device)))
        width = max(chunk.shape[1] for _, chunk, _ in chunks)
        out = torch.zeros(len(prompts), width, chunks[0][1].shape[-1], device=self.device, dtype=torch.float32)
        mask = torch.zeros(len(prompts), width, device=self.device, dtype=torch.bool)
        for group, chunk, chunk_mask in chunks:
            index = torch.tensor(group, device=self.device)
            out[index, : chunk.shape[1]] = chunk
            mask[index, : chunk.shape[1]] = chunk_mask
        return out, mask

