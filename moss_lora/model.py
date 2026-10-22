"""MOSSImage dense text-to-image transformer.

Copied from the official MOSS-Image reference and
adapted for standalone training in two places only:
  * ``optimized_attention`` (reference dispatcher) -> ``F.scaled_dot_product_attention``
  * ``operations`` (reference op namespace) is optional and defaults to plain torch ops
Everything else, including the per-sample attention-mask handling, is unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:  # 可选依赖：打包训练用的 varlen kernel（安装见 setup_fa2.py）
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen
except Exception:  # pragma: no cover - 没装就退回 SDPA
    _flash_attn_varlen = None


def flash_attn_available() -> bool:
    """是否能用 FlashAttention-2 的 varlen 路径。"""
    return _flash_attn_varlen is not None


def optimized_attention(q, k, v, heads, skip_reshape=True, mask=None,
                        cu_seqlens=None, max_seqlens=None, **kwargs):
    """Stand-in for the reference attention dispatcher.

    两个后端，按调用方是否给出 ``cu_seqlens`` 分流：

    * 给了 ``cu_seqlens`` 且装得上 FlashAttention-2 → varlen kernel，只在各自的
      样本区间内做注意力，不需要稠密 mask，也不会算被遮掉的那部分
    * 否则 → ``F.scaled_dot_product_attention`` + 传入的 mask

    ``skip_reshape=True`` means q/k/v already arrive as [B, H, L, D]; the flattened
    [B, L, H*D] output is what the caller expects, matching the reference contract.
    """
    if cu_seqlens is not None:
        if _flash_attn_varlen is None:
            raise RuntimeError(
                "packed attention requires FlashAttention-2; "
                "run `.venv/bin/python setup_fa2.py install`"
            )
        if q.shape[0] != 1:
            raise ValueError("varlen attention expects one packed sequence (B=1)")
        length = q.shape[2]
        out = _flash_attn_varlen(
            q[0].transpose(0, 1).contiguous(),
            k[0].transpose(0, 1).contiguous(),
            v[0].transpose(0, 1).contiguous(),
            cu_seqlens,
            cu_seqlens,
            max_seqlens,
            max_seqlens,
        )
        return out.reshape(1, length, heads * out.shape[-1])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    b, _, length, dim = out.shape
    return out.transpose(1, 2).reshape(b, length, heads * dim)


class _TorchOps:
    """Default ``operations`` namespace: plain torch modules."""

    Linear = nn.Linear
    RMSNorm = nn.RMSNorm


def _module_dtype(module) -> torch.dtype:
    """Compute dtype of a linear, tolerating quantized (non-float) weights."""
    base = getattr(module, "base_layer", module)
    dtype = base.weight.dtype
    if dtype.is_floating_point:
        return dtype
    compute = getattr(base, "compute_dtype", None)  # bitsandbytes Linear4bit
    return compute if isinstance(compute, torch.dtype) else torch.bfloat16


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, mask=None, cu_seqlens=None, max_seqlens=None):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return optimized_attention(
        q, k, v, heads, skip_reshape=True, mask=mask,
        cu_seqlens=cu_seqlens, max_seqlens=max_seqlens,
    )


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = (
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
        )
        self.q_norm, self.k_norm = (
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
        )
        self.qkv = operations.Linear(dim, 3 * dim, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.ffn_gate, self.ffn_up = (
            operations.Linear(dim, ffn_dim, device=device, dtype=dtype),
            operations.Linear(dim, ffn_dim, device=device, dtype=dtype),
        )
        self.ffn_down = operations.Linear(ffn_dim, dim, device=device, dtype=dtype)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads, device, dtype, operations):
        super().__init__()
        self.proj = operations.Linear(text_dim, dim, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [_TextBlock(dim, ffn_dim, heads, device, dtype, operations) for _ in range(2)]
        )

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads, device, dtype, operations):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = (
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype),
        )
        self.q_norm, self.k_norm = (
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
            operations.RMSNorm(dim // heads, eps=1e-6, device=device, dtype=dtype),
        )
        self.proj = nn.ModuleDict(
            {
                name: operations.Linear(dim, dim, bias=False, device=device, dtype=dtype)
                for name in ("q", "k", "v", "o")
            }
        )
        self.proj["ffn_up_gate"] = operations.Linear(dim, 2 * ffn_dim, bias=False, device=device, dtype=dtype)
        self.proj["ffn_down"] = operations.Linear(ffn_dim, dim, bias=False, device=device, dtype=dtype)

    def attn(self, x, rope, modulation, mask=None, cu_seqlens=None, max_seqlens=None):
        """Attention half of the block (the split exists for selective checkpointing)."""
        shift_a, scale_a, gate_a = modulation.unbind(1)[:3]
        dtype = _module_dtype(self.proj["q"])
        y = (self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]).to(dtype)
        # quantized/LoRA layers can hand back mixed dtypes; attention needs one
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2).to(dtype)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2).to(dtype)
        out = _attention(q, k, self.proj["v"](y).to(dtype), rope, self.heads, mask,
                         cu_seqlens, max_seqlens)
        return x + self.proj["o"](out) * gate_a[:, None]

    def ffn(self, x, modulation):
        """FFN half of the block (~60% of its FLOPs)."""
        shift_f, scale_f, gate_f = modulation.unbind(1)[3:]
        dtype = _module_dtype(self.proj["q"])
        y = (self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]).to(dtype)
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]

    def forward(self, x, rope, modulation, mask=None, cu_seqlens=None, max_seqlens=None):
        return self.ffn(
            self.attn(x, rope, modulation, mask, cu_seqlens, max_seqlens), modulation
        )


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", torch.empty(self.num_bands))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def forward(self, height, width, text_length, device, center=None):
        bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        image = (
            torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1)
            .flatten(0, 1)
            .float()
        )
        # ``center`` 让画布 (height, width) 与真实图像 (h, w) 不同时仍按真实尺寸居中，
        # padding 到统一画布的实验需要这一点，否则 RoPE 相位会随画布大小漂移。
        ch, cw = center if center is not None else (height, width)
        image[:, 2] -= (ch - 1) / 2
        image[:, 3] -= (cw - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(nn.Module):
    """MOSSImage DiT with training-compatible keys and reference attention.

    Each sample is evaluated at its true text length, including text refinement;
    padded tokens cannot influence either text or joint attention.
    """

    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
        device=None,
        dtype=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        operations = operations or _TorchOps
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.dtype = dtype
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        # local additions: per-block activation checkpointing and block swap hooks
        self.gradient_checkpointing = False
        self.block_swap = None
        # local addition: vectorise equal-length text batches instead of looping
        self.batch_mode = False
        # 打包注意力的后端：auto（有 FA2 就用）/ flash / sdpa（强制稠密 mask）
        self.attention_backend = "auto"
        # SLG: layer indices bypassed (identity) for the unconditional branch
        self.skip_layers: tuple[int, ...] = ()

        self.video_embedder = operations.Linear(latent_channels, dim, device=device, dtype=dtype)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads, device, dtype, operations)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.shared_block_modulation = operations.Linear(dim, 6 * dim, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [_Block(dim, ffn_dim, num_heads, device, dtype, operations) for _ in range(num_layers)]
        )
        self.final_norm_video = operations.RMSNorm(dim, eps=1e-6, device=device, dtype=dtype)
        self.final_linear_video = operations.Linear(
            dim, latent_channels, bias=False, device=device, dtype=dtype
        )
        self.video_head_modulation = operations.Linear(dim, 2 * dim, device=device, dtype=dtype)

    @staticmethod
    def packed_attention_backend() -> str:
        """打包训练实际会走哪个注意力后端（供日志/自检使用）。"""
        return "flash_attn varlen" if flash_attn_available() else "sdpa dense-mask"

    def _time_mods(self, timestep, device):
        """Shared (mod, head) for one scalar timestep; batch dim broadcast later."""
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(1)
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(1, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(1, 2, -1)
        return mod, head

    def forward_packed(self, latents, timestep, contexts, masks=None, checkpoint_blocks=False):
        """Pack several samples of different sizes into one sequence.

        All samples share one timestep (one modulation), which is what makes packing
        compatible with the block interface. Sample spans are enforced by the attention
        backend: FlashAttention-2 ``cu_seqlens`` when available (no mask, no wasted
        compute), otherwise a dense block-diagonal mask on top of SDPA.
        """
        dtype = self.video_embedder.weight.dtype
        device = latents[0].device
        pieces, ropes, spans = [], [], []
        for latent, context in zip(latents, contexts):
            height, width = latent.shape[-2:]
            n_img = height * width
            video = self.video_embedder(latent.flatten(2).transpose(1, 2).to(dtype))
            rope = self.rope(height, width, context.shape[1], device).to(dtype)
            text = self.text_embedder(context, rope[:, n_img:])
            pieces.append(torch.cat((video, text), 1))
            ropes.append(rope)
            spans.append(video.shape[1] + text.shape[1])
        x = torch.cat(pieces, 1)
        rope = torch.cat(ropes, 1)
        # 样本区间：能走 varlen 就只传边界，否则退化成稠密块对角 mask
        use_flash = self.attention_backend != "sdpa" and flash_attn_available() and x.is_cuda
        if use_flash:
            spans_t = torch.tensor(spans, device=device, dtype=torch.int32)
            cu_seqlens = torch.cat((spans_t.new_zeros(1), spans_t.cumsum(0, dtype=torch.int32)))
            max_seqlens = int(max(spans))
            mask = None
        else:
            doc = torch.repeat_interleave(
                torch.arange(len(spans), device=device), torch.tensor(spans, device=device)
            )
            mask = (doc[:, None] == doc[None, :])[None, None]
            cu_seqlens, max_seqlens = None, None
        mod, head = self._time_mods(timestep, device)
        for index, layer in enumerate(self.layers):
            if index in self.skip_layers:
                continue
            if checkpoint_blocks and self.training:
                x = checkpoint(
                    layer, x, rope, mod, mask, cu_seqlens, max_seqlens,
                    use_reentrant=False,
                )
            else:
                x = layer(x, rope, mod, mask, cu_seqlens, max_seqlens)
        outputs, offset = [], 0
        for span, latent in zip(spans, latents):
            height, width = latent.shape[-2:]
            part = x[:, offset : offset + height * width]
            part = self.final_norm_video(part.to(self.final_norm_video.weight.dtype))
            part = part * (1 + head[:, 1, None]) + head[:, 0, None]
            part = self.final_linear_video(part.to(self.final_linear_video.weight.dtype))
            outputs.append(part.transpose(1, 2).reshape(1, -1, height, width))
            offset += span
        return outputs

    def forward_padded(self, latents, timestep, contexts, checkpoint_blocks=False):
        """传统 [B, L] 批次，在 **token 维** 做 padding。

        每张 latent 先各自 flatten 成 token 序列（长度 = 各自的 h/16 × w/16），再在序列
        轴上补齐到 ``L_max``。pad 出来的空位只有 ``L_max − L_i``，而不会像按画布 padding
        那样把每张图撑成「最宽 × 最高」。

        LLM 用右侧 padding 免 mask 是因为因果注意力；这里注意力是双向的，真实 token 会
        看到尾部的 pad token，所以仍然需要 key-padding mask —— 这份 mask 税（走不了
        flash kernel）才是「没有 varlen」的真正代价。

        返回逐样本输出（已按各自真实尺寸 reshape），调用方不必再关心 padding。
        """
        dtype = self.video_embedder.weight.dtype
        device = latents[0].device
        batch = len(latents)
        channels = latents[0].shape[1]
        sizes = [(int(latent.shape[-2]), int(latent.shape[-1])) for latent in latents]

        pieces, ropes, spans = [], [], []
        for index, latent in enumerate(latents):
            height, width = sizes[index]
            n_img = height * width
            video = self.video_embedder(latent.flatten(2).transpose(1, 2).to(dtype))
            rope = self.rope(height, width, contexts[index].shape[1], device).to(dtype)
            text = self.text_embedder(contexts[index], rope[:, n_img:])
            pieces.append(torch.cat((video, text), 1))
            ropes.append(rope)
            spans.append(int(video.shape[1] + text.shape[1]))

        seq = max(spans)
        x = pieces[0].new_zeros(batch, seq, pieces[0].shape[-1])
        rope = ropes[0].new_zeros(batch, seq, ropes[0].shape[-1])
        token_mask = torch.zeros(batch, seq, dtype=torch.bool, device=device)
        for index in range(batch):
            x[index, : spans[index]] = pieces[index][0]
            rope[index, : spans[index]] = ropes[index][0]
            token_mask[index, : spans[index]] = True

        # 没有 padding 时不传 mask：SDPA 见到 mask 就退出 flash kernel。
        mask = None if bool(token_mask.all()) else token_mask[:, None, None, :]
        mod, head = self._time_mods(timestep, device)
        for index, layer in enumerate(self.layers):
            if index in self.skip_layers:
                continue
            if checkpoint_blocks and self.training:
                x = checkpoint(
                    layer, x, rope, mod, mask, None, None, use_reentrant=False
                )
            else:
                x = layer(x, rope, mod, mask)

        outputs = []
        for index, (height, width) in enumerate(sizes):
            n_img = height * width
            part = x[index : index + 1, :n_img]
            part = self.final_norm_video(part.to(self.final_norm_video.weight.dtype))
            part = part * (1 + head[:, 1, None]) + head[:, 0, None]
            part = self.final_linear_video(part.to(self.final_linear_video.weight.dtype))
            outputs.append(part.transpose(1, 2).reshape(1, channels, height, width))
        return outputs

    def _run_block(self, layer, index, x, rope, mod):
        """One block with the configured swap/checkpoint strategy."""
        if index in self.skip_layers:
            return x
        if (self.block_swap is not None and self.training
                and self.block_swap.is_swapped(index)):
            # offloaded blocks are always recomputed (the weights are gone by backward)
            return self.block_swap(layer, index, x, rope, mod)
        if not (self.gradient_checkpointing and self.training):
            return layer(x, rope, mod)
        return checkpoint(layer, x, rope, mod, use_reentrant=False)

    def _forward_batched(self, hidden_states, times, context, height, width, mod, head):
        """One forward for the whole batch; requires equal text lengths.

        With no padding the joint attention needs no mask, so this is the same maths
        as the per-sample loop -- just B times fewer kernel launches.
        """
        batch = hidden_states.shape[0]
        n_img = height * width
        dtype = self.video_embedder.weight.dtype
        video = self.video_embedder(hidden_states.flatten(2).transpose(1, 2).to(dtype))
        rope = self.rope(height, width, context.shape[1], hidden_states.device).to(dtype)
        text = self.text_embedder(context, rope[:, n_img:])
        x = torch.cat((video, text), 1)
        for layer_index, layer in enumerate(self.layers):
            x = self._run_block(layer, layer_index, x, rope, mod)
        x = self.final_norm_video(x[:, :n_img].to(self.final_norm_video.weight.dtype))
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(batch, -1, height, width)

    def forward(self, x, timestep, context, attention_mask=None, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        hidden_states, encoder_hidden_states = x, context
        encoder_attention_mask = attention_mask
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(
            batch, 6, -1
        )
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        if self.batch_mode and batch > 1 and bool((mask.sum(-1) == mask.sum(-1)[0]).all()):
            return self._forward_batched(hidden_states, times, encoder_hidden_states, height, width, mod, head)

        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            # Match the reference activation dtype under fp32 weights + autocast.
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width :])
            x = torch.cat((video, text), 1)
            transformer_options["total_blocks"] = len(self.layers)
            transformer_options["block_type"] = "single"
            for layer_index, layer in enumerate(self.layers):
                transformer_options["block_index"] = layer_index
                if ("single_block", layer_index) in blocks_replace:

                    def block_wrap(args):
                        return {"img": layer(args["img"], args["pe"], args["vec"])}

                    out = blocks_replace[("single_block", layer_index)](
                        {
                            "img": x,
                            "pe": rope,
                            "vec": mod[index : index + 1],
                            "transformer_options": transformer_options,
                        },
                        {"original_block": block_wrap},
                    )
                    x = out["img"]
                else:
                    x = self._run_block(layer, layer_index, x, rope, mod[index : index + 1])
            x = self.final_norm_video(x[:, : height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return sample
