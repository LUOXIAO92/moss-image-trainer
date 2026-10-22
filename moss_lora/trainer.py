"""LoRA training loop.

Forward/backward goes through the upstream transformer; noise, loss, optimiser,
EMA, checkpointing and previews are ours. The flow-matching target is the report's:
v = noise - x0 at t = 1000 * sigma, with a token-normalised MSE.
"""

from __future__ import annotations

import json
import math
import hashlib
import os
import time
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from .config import Config
from .sigma import SigmaSampler
from .weights import LatentCodec, load_transformer, load_vae

LORA_TARGETS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]


@dataclass
class TrainState:
    step: int = 0
    losses: list = field(default_factory=list)
    evals: list = field(default_factory=list)
    lr: float = 0.0
    peak_gib: float = 0.0


class Trainer:
    def __init__(self, config: Config, log=print):
        self.config = config
        self.log = log
        self.device = "cuda"
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            str(self.config.model.dtype).lower(), torch.bfloat16
        )
        self.state = TrainState()
        self.cache_dir = config.cache.dir
        with open(os.path.join(self.cache_dir, "index.json"), encoding="utf-8") as f:
            self.index = json.load(f)
        self.items = self.index["items"]
        # bucket by image size only: (size, text_len) gives 77 shapes for 41 sizes at
        # batch 1, and equal-length grouping is only needed when batching anyway
        self.buckets: dict[tuple, list] = {}
        self.eval_items: list = []
        for item in self.items:
            self.buckets.setdefault((item["width"], item["height"]), []).append(item)
        self._split_eval()
        self.out_dir = os.path.join(config.output_dir, config.name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.generator = torch.Generator("cpu").manual_seed(config.train.seed)

    # -- setup -----------------------------------------------------------
    def _configure_precision(self):
        if self.dtype == torch.float16 and bool(getattr(self.config.vram, "fp16_accumulate", False)):
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            self.log("precision: fp16 with fp16 accumulation")

    def load(self):
        self._configure_precision()
        self.log(f"loading transformer {self.config.model.dit}")
        self.model = load_transformer(self.config.model.dit, dtype=self.dtype, device=self.device)
        self.model.train()
        self._attach_lora()
        self._maybe_quantize()
        self._set_checkpointing(bool(self.config.vram.gradient_checkpointing))
        self._setup_block_swap()
        self._maybe_compile()
        self._set_batch_mode(True)
        self._resolve_attention()
        params = [p for p in self.model.parameters() if p.requires_grad]
        n = sum(p.numel() for p in params)
        self.log(f"lora: {n/1e6:.1f}M trainable params (rank {self.config.lora.rank}, alpha {self.config.lora.alpha})")
        self.optimizer = self._make_optimizer(params)
        # fp16 needs loss scaling; bf16 does not
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
        self.sigma = SigmaSampler(self.config.train.sigma, generator=self.generator)
        self.ema = None
        decay = self.config.train.ema
        if decay > 0:
            self.ema = {k: v.detach().clone().float() for k, v in self.model.named_parameters() if v.requires_grad}
        return self

    def _set_checkpointing(self, enabled: bool):
        """Toggle per-block gradient checkpointing (added to the upstream forward)."""
        model = self.model
        base = getattr(model, "base_model", None)
        if base is not None:
            model = getattr(base, "model", base)
        for module in [model] + [m for m in model.modules() if type(m).__name__ == "MOSSImageTransformer"]:
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enabled
        self.log(f"gradient checkpointing: {'on' if enabled else 'off'}")

    def _maybe_quantize(self):
        scope = str(getattr(self.config.vram, "quantize_dit", "none") or "none").lower()
        if scope == "none":
            return
        from .quant import quantize_nf4

        model = getattr(getattr(self.model, "base_model", self.model), "model", self.model)
        replaced = quantize_nf4(model, scope, dtype=self.dtype)
        torch.cuda.empty_cache()
        self.log(f"quantize: {replaced} frozen Linears -> nf4 ({scope})")

    def _maybe_compile(self):
        if not bool(getattr(self.config.vram, "compile", False)):
            return
        model = getattr(getattr(self.model, "base_model", self.model), "model", self.model)
        try:
            model.forward = torch.compile(model.forward, dynamic=True, fullgraph=False)
            self.log("compile: torch.compile on the transformer forward")
        except Exception as exc:  # pragma: no cover - depends on the install
            self.log(f"compile: failed ({type(exc).__name__}: {exc}), continuing eager")

    def _split_eval(self):
        """Deterministic holdout: hash decides, so every run evaluates the same images."""
        share = float(getattr(self.config.train, "eval_holdout", 0.0) or 0.0)
        if share <= 0:
            return
        keep = {}
        for key, items in self.buckets.items():
            train, held = [], []
            for item in items:
                digest = hashlib.sha1(os.path.abspath(item["image"]).encode()).hexdigest()
                (held if int(digest[:8], 16) % 1000 < share * 1000 else train).append(item)
            if train:
                keep[key] = train
            self.eval_items.extend(held)
        self.buckets = keep

    def _append_metrics(self, row: dict):
        """One JSON line per logged step / eval, read by the web dashboard."""
        import json

        path = os.path.join(self.out_dir, "metrics.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _gpu_util(self) -> int | None:
        """GPU utilisation in %, sampled at most once every 2 s.

        nvidia-smi costs tens of milliseconds and the venv has no pynvml, so a
        per-step call would show up in the step time.
        """
        import subprocess
        import time

        now = time.monotonic()
        cached = getattr(self, "_gpu_util_cache", None)
        if cached and now - cached[0] < 2.0:
            return cached[1]
        util = None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip().splitlines()
            if out:
                util = int(float(out[0].strip()))
        except Exception:
            util = None
        self._gpu_util_cache = (now, util)
        return util

    def _eval_loss(self) -> float:
        cfg = self.config.train
        if not self.eval_items:
            return float("nan")
        subset = self.eval_items[: max(1, int(cfg.eval_batches))]
        generator = torch.Generator("cpu").manual_seed(1234)  # same noise every eval
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for item in subset:
                x0, embeds, mask = self._load_batch([item], item["width"], item["height"])
                sigma = self.sigma.sample(1, self.device, generator=generator)
                noise = torch.randn(x0.shape, generator=generator, device="cpu").to(self.device)
                sigma_b = sigma.view(-1, 1, 1, 1)
                xt = ((1 - sigma_b) * x0.float() + sigma_b * noise).to(self.dtype)
                prediction = self.model(xt, sigma * 1000, embeds, mask)
                total += float(torch.nn.functional.mse_loss(prediction.float(), noise - x0.float()))
        self.model.train(was_training)
        return total / len(subset)

    def _resolve_attention(self) -> str:
        """按 train.attention 决定打包注意力后端。

        auto（默认）：装得上 FlashAttention-2 就用 varlen，否则回退 SDPA 的稠密 mask；
        flash / sdpa 是显式指定，显式要 flash 但没有可用内核时告警并回退。
        """
        from .model import flash_attn_available

        want = str(getattr(self.config.train, "attention", "auto") or "auto").lower()
        available = flash_attn_available() and self.device == "cuda"
        if want == "flash" and not available:
            self.log("attention: 要求 flash，但环境里没有可用的 FlashAttention-2 → 回退 SDPA")
            backend = "sdpa"
        elif want == "sdpa":
            backend = "sdpa"
        else:
            backend = "auto" if available else "sdpa"
        if hasattr(self.model, "attention_backend"):
            self.model.attention_backend = backend
        self.log(
            "attention: "
            + ("flash-attn varlen" if backend != "sdpa" else "sdpa（稠密 block-diagonal mask）")
        )
        return backend

    def _set_batch_mode(self, enabled: bool):
        """Vectorised forward for equal-length batches (see _forward_batched)."""
        model = getattr(getattr(self.model, "base_model", self.model), "model", self.model)
        model.batch_mode = enabled

    def _setup_block_swap(self):
        count = int(self.config.vram.blocks_to_swap or 0)
        if count <= 0:
            return
        from .offload import BlockSwap

        model = getattr(getattr(self.model, "base_model", self.model), "model", self.model)
        if not getattr(model, "gradient_checkpointing", False):
            raise RuntimeError("blocks_to_swap needs gradient checkpointing")
        self._swap = BlockSwap(model.layers, count, device=self.device)
        model.block_swap = self._swap
        freed = sum(p.numel() * p.element_size() for i in self._swap.params for p in self._swap.params[i])
        self.log(f"block swap: {count} blocks on CPU ({freed/2**30:.1f} GiB offloaded)")

    def _attach_lora(self):
        from peft import LoraConfig, get_peft_model

        targets = list(self.config.lora.targets or LORA_TARGETS)
        # full-match on the block projections only: a bare suffix list also catches
        # text_embedder.blocks.*.ffn_down, which is not what "DiT LoRA" should mean
        layer_expr = "\\d+"
        layers = [int(x) for x in (self.config.lora.layers or [])]
        if layers:
            # A UI range selection becomes an explicit alternation; PEFT matches
            # this regex against the module name, not against a substring.
            layer_expr = "(?:" + "|".join(str(x) for x in sorted(set(layers))) + ")"
        pattern = "layers\\." + layer_expr + "\\.proj\\.(" + "|".join(targets) + ")$"
        if self.config.lora.include_text_embedder:
            pattern = (pattern[:-1] + "|text_embedder\\.blocks\\.\\d+\\.(qkv|proj|ffn_up|ffn_gate|ffn_down)$")
        config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            target_modules=pattern,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            init_lora_weights=True,
        )
        self.model = get_peft_model(self.model, config)
        if self.config.lora.init == "zero":
            for name, param in self.model.named_parameters():
                if "lora_B" in name:
                    torch.nn.init.zeros_(param)

    def _make_optimizer(self, params):
        cfg = self.config.train
        if cfg.optimizer == "adamw8bit" or self.config.vram.optimizer_in_8bit:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
        if cfg.optimizer == "muon":
            try:
                from muon import SingleDeviceMuon as Muon  # official package, single-GPU variant

                return Muon(params, lr=cfg.lr, momentum=0.95, weight_decay=cfg.weight_decay)
            except ImportError:
                from .muon import Muon

                return Muon(params, lr=cfg.lr, momentum=0.95, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adafactor":
            from transformers import Adafactor

            return Adafactor(params, lr=cfg.lr, scale_parameter=False, relative_step=False, weight_decay=cfg.weight_decay)
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # -- data ------------------------------------------------------------
    def _bucket_indices(self):
        return list(self.buckets.items())

    def _sample_batch(self, rng: torch.Generator):
        buckets = self._bucket_indices()
        # a group that cannot fill the batch defeats the vectorised forward
        full = [b for b in buckets if len(b[1]) >= self.config.train.batch_size]
        buckets = full or buckets
        # kohya convention: a "<repeat>_<name>" directory means each image is sampled
        # that many times, so both the bucket and the item draw are weighted by it
        def weight(item):
            try:
                return max(1, int(item.get("repeats", 1) or 1))
            except (TypeError, ValueError):
                return 1

        counts = torch.tensor([sum(weight(i) for i in items) for _, items in buckets], dtype=torch.float32)
        choice = torch.multinomial(counts, 1, generator=rng).item()
        (width, height), items = buckets[choice]
        batch_size = min(self.config.train.batch_size, len(items))
        if batch_size > 1:
            by_length: dict[int, list] = {}
            for item in items:
                by_length.setdefault(item.get("text_len", 0), []).append(item)
            ready = [group for group in by_length.values() if len(group) >= batch_size]
            if ready:
                items = max(ready, key=len)
                batch_size = min(batch_size, len(items))
        weights = torch.tensor([weight(i) for i in items], dtype=torch.float32)
        picks = torch.multinomial(weights, batch_size, replacement=False, generator=rng).tolist()
        return [items[i] for i in picks], width, height

    def _load_batch(self, entries, width, height):
        latents, embeds, masks = [], [], []
        for entry in entries:
            path = os.path.join(self.cache_dir, entry["file"])
            if not os.path.exists(path):
                raise SystemExit(
                    f"缓存文件不存在：{path}\n"
                    "索引和实际文件对不上；到「缓存」页重新生成一次缓存。"
                )
            data = load_file(path, device="cpu")
            missing = [key for key in ("latent", "text_embeds", "text_mask") if key not in data]
            if missing:
                raise SystemExit(
                    f"缓存里缺 {', '.join(missing)}：{path}\n"
                    "训练需要文本嵌入；先在「缓存」页点「一键生成缓存（latent + 文本）」，"
                    "或跑 python -m moss_lora.cli encode-text --config <yaml>。"
                )
            latents.append(data["latent"])
            embeds.append(data["text_embeds"])
            masks.append(data["text_mask"])
        length = max(e.shape[0] for e in embeds)
        padded = embeds[0].new_zeros(len(embeds), length, embeds[0].shape[-1])
        mask = torch.zeros(len(embeds), length, dtype=torch.bool)
        for i, (e, m) in enumerate(zip(embeds, masks)):
            padded[i, : e.shape[0]] = e
            mask[i, : e.shape[0]] = m[: e.shape[0]]
        x0 = torch.stack(latents).to(self.device, self.dtype)
        return x0, padded.to(self.device, self.dtype), mask.to(self.device)

    def _sample_random(self, rng: torch.Generator, count: int) -> list[dict]:
        """均匀抽样：不按尺寸分组，供不同注意力后端的横向对比使用。"""
        pool = self.items
        count = max(1, min(int(count), len(pool)))
        weights = torch.tensor(
            [max(1, int(i.get("repeats", 1) or 1)) for i in pool], dtype=torch.float32
        )
        picks = torch.multinomial(weights, count, replacement=False, generator=rng).tolist()
        return [pool[k] for k in picks]

    def _sample_by_tokens(self, rng: torch.Generator, budget: int) -> list[dict]:
        """按 token 预算装箱：随机顺序累加，装到超过预算为止（至少 1 张）。"""
        pool = self.items
        order = torch.randperm(len(pool), generator=rng).tolist()
        picks, total = [], 0
        for idx in order:
            item = pool[idx]
            tokens = (int(item["width"]) // 16) * (int(item["height"]) // 16) + int(
                item.get("text_len", 0) or 0
            )
            if picks and total + tokens > budget:
                break
            picks.append(item)
            total += tokens
        return picks or [pool[order[0]]]

    def _load_packed(self, entries):
        """每样本保留自己的尺寸和真实文本长度（不 padding、不裁剪）。"""
        latents, contexts = [], []
        for entry in entries:
            data = load_file(os.path.join(self.cache_dir, entry["file"]), device="cpu")
            latents.append(data["latent"].unsqueeze(0).to(self.device, self.dtype))
            mask = data["text_mask"]
            keep = int(mask.sum())
            contexts.append(
                data["text_embeds"][:keep].unsqueeze(0).to(self.device, self.dtype)
            )
        return latents, contexts

    # -- training --------------------------------------------------------
    def warmup_steps(self) -> int:
        """``train.warmup`` is either a step count or a fraction of the run."""
        cfg = self.config.train
        value = float(cfg.warmup or 0)
        if 0 < value < 1:
            return max(1, int(round(value * max(cfg.steps, 1))))
        return max(0, int(round(value)))

    def _lr_at(self, step: int) -> float:
        cfg = self.config.train
        total = max(cfg.steps, 1)
        warmup = self.warmup_steps()
        if step < warmup:
            return cfg.lr * (step + 1) / max(warmup, 1)
        if cfg.lr_scheduler == "cosine":
            progress = (step - warmup) / max(total - warmup, 1)
            return cfg.lr * (0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))
        return cfg.lr

    def _train_step(self, step: int, rng) -> float:
        cfg = self.config.train
        self.optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        packed = cfg.batch_mode in ("packed", "padded")
        for _ in range(cfg.grad_accum):
            if packed:
                # bin size 的单位始终是 token：留空（0）时按面积上限换算出一个
                # 默认预算（约 3 张满尺寸图的 token + 1k 文本余量），而不是按张数取样。
                budget = int(getattr(cfg, "bin_tokens", 0) or 0) or (
                    3 * (int(self.config.dataset.max_area or 1048576) // 256) + 1024
                )
                entries = self._sample_by_tokens(rng, budget)
                latents, contexts = self._load_packed(entries)
                # 打包/填充两条路都必须共用一个 timestep（共享 modulation）
                sigma = self.sigma.sample(1, self.device).reshape(1)
                noise = [torch.randn_like(latent, dtype=torch.float32) for latent in latents]
                xt = [
                    ((1 - sigma) * latent.float() + sigma * eps).to(self.dtype)
                    for latent, eps in zip(latents, noise)
                ]
                target = [eps - latent.float() for latent, eps in zip(latents, noise)]
                forward = (
                    self.model.forward_packed
                    if cfg.batch_mode == "packed"
                    else self.model.forward_padded
                )
                prediction = forward(
                    xt,
                    float(sigma.item()) * 1000,
                    contexts,
                    checkpoint_blocks=bool(getattr(self.model, "gradient_checkpointing", False)),
                )
                loss = sum(
                    F.mse_loss(pred.float(), tgt) for pred, tgt in zip(prediction, target)
                ) / len(prediction)
                self.scaler.scale(loss / cfg.grad_accum).backward()
                step_loss += float(loss.detach()) / cfg.grad_accum
                continue
            entries, width, height = self._sample_batch(rng)
            x0, embeds, mask = self._load_batch(entries, width, height)
            sigma = self.sigma.sample(x0.shape[0], self.device)
            noise = torch.randn(x0.shape, generator=rng, device="cpu", dtype=torch.float32).to(self.device)
            sigma_b = sigma.view(-1, 1, 1, 1)
            xt = ((1 - sigma_b) * x0.float() + sigma_b * noise).to(self.dtype)
            target = noise - x0.float()
            prediction = self.model(xt, sigma * 1000, embeds, mask)
            loss = F.mse_loss(prediction.float(), target)
            self.scaler.scale(loss / cfg.grad_accum).backward()
            step_loss += float(loss.detach()) / cfg.grad_accum
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], cfg.max_grad_norm
        )
        lr = self._lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self._update_ema()
        self.state.step = step + 1
        self.state.lr = lr
        self.state.losses.append(step_loss)
        return step_loss

    def _degrade_memory(self) -> bool:
        """OOM fallback: move more blocks to CPU and retry the step."""
        current = int(self.config.vram.blocks_to_swap or 0)
        if current >= 40:
            return False
        nxt = min(40, current + 8 if current else 8)
        self.log(f"out of memory -> blocks_to_swap {current} -> {nxt}, retrying")
        self.config.vram.blocks_to_swap = nxt
        model = getattr(getattr(self.model, "base_model", self.model), "model", self.model)
        from .offload import BlockSwap

        model.block_swap = BlockSwap(model.layers, nxt, device=self.device)
        self._swap = model.block_swap
        torch.cuda.empty_cache()
        return True

    def _report_allocator(self):
        """报告本次训练实际使用的显存回收机制。

        两种机制都只是策略不同，不影响训练正确性；没启用时给出两条可直接照抄的配置。
        """
        from .settings import alloc_conf, load_settings

        def name(conf: str) -> str:
            if conf.startswith("expandable_segments"):
                return "expandable_segments + 阈值回收"
            if conf.startswith("backend:cudaMallocAsync"):
                return "cudaMallocAsync"
            return conf or "默认"

        have = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        try:
            runtime = load_settings()["runtime"]
            label = runtime["allocator"]
            want = alloc_conf(label, runtime["gc_threshold"])
        except Exception:
            label, want = "unknown", ""
        if not have:
            self.log("allocator: 未设置额外显存回收机制，长训时可能显存紧张")
            self.log(
                "  可选 1（推荐，settings 里 allocator: expandable）: "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.90"
            )
            self.log(
                "  可选 2（settings 里 allocator: async）: "
                "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
            )
            return
        if have == want:
            self.log(f"allocator: 已启动 GC（{name(have)}）")
        else:
            self.log(
                f"allocator: 与 settings 不一致（settings={label}，实际={name(have)}），"
                f"已启动 GC"
            )

    def train(self, resume: str = ""):
        cfg = self.config.train
        self._report_allocator()
        start_step = self._load_checkpoint(resume) if resume else 0
        rng = self.generator
        torch.cuda.reset_peak_memory_stats()
        running = 0.0
        self.model.train()
        self._write_heartbeat(start_step, "start")
        for step in range(start_step, cfg.steps):
            # perf_counter is monotonic: a system clock adjustment (NTP/WSL drift)
            # must not turn a step time into 0 or a negative number.
            started = time.perf_counter()
            try:
                step_loss = self._train_step(step, rng)
            except torch.cuda.OutOfMemoryError:
                if not self._degrade_memory():
                    raise
                step_loss = self._train_step(step, rng)
            running += step_loss
            if (step + 1) % cfg.log_every == 0:
                peak = torch.cuda.max_memory_allocated() / 2**30
                self.state.peak_gib = peak
                elapsed = time.perf_counter() - started
                self.log(
                    f"step {step + 1}/{cfg.steps}  loss {step_loss:.4f}  avg {running / cfg.log_every:.4f}  "
                    f"lr {self.state.lr:.2e}  {elapsed:.2f}s  peak {peak:.1f} GiB"
                )
                self._append_metrics({
                    "step": step + 1, "total": cfg.steps, "train": step_loss,
                    "avg": running / cfg.log_every, "lr": self.state.lr,
                    "sec": round(elapsed, 3), "peak": round(peak, 2),
                    "gpu": self._gpu_util(),
                })
                running = 0.0
            if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
                loss_eval = self._eval_loss()
                self.state.evals.append((step + 1, loss_eval))
                self.log(f"step {step + 1}/{cfg.steps}  train {step_loss:.4f}  eval {loss_eval:.4f}")
                self._append_metrics({"step": step + 1, "total": cfg.steps, "eval": round(loss_eval, 5)})
            if (step + 1) % 20 == 0:
                self._write_heartbeat(step + 1, "running")
            if cfg.save_every and (step + 1) % cfg.save_every == 0:
                self.save_checkpoint(step + 1)
            if cfg.sample_every and (step + 1) % cfg.sample_every == 0:
                try:
                    self.preview(step + 1)
                except Exception as exc:  # a broken preview must not kill the run
                    self.log(f"preview failed at step {step + 1}: {type(exc).__name__}: {exc}")
        self.save_checkpoint(cfg.steps, final=True)
        self.export()
        self.make_contact_sheet()
        self._write_heartbeat(cfg.steps, "done")
        return self.state

    def _write_heartbeat(self, step: int, status: str):
        """Let the dashboard see a CLI-started run without the web job system."""
        try:
            path = os.path.join(self.out_dir, "training.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "step": int(step),
                           "total": int(self.config.train.steps), "status": status,
                           "time": time.time()}, handle)
        except OSError:
            pass

    # -- state -----------------------------------------------------------
    def _update_ema(self):
        if not self.ema:
            return
        decay = self.config.train.ema
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.ema:
                    self.ema[name].mul_(decay).add_(param.detach().float(), alpha=1 - decay)

    def _trainable(self):
        """(names, params) in a stable order shared by save and load."""
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(names) != len(params):
            raise RuntimeError("parameter order mismatch; cannot map optimizer state")
        return names, params

    def save_checkpoint(self, step: int, final: bool = False):
        tag = "final" if final else f"step{step:06d}"
        path = os.path.join(self.ckpt_dir, f"{tag}.safetensors")
        names, params = self._trainable()
        from .lora_file import kohya_key

        state = {}
        modules = {}
        for name, param in zip(names, params):
            if ".lora_" not in name:
                continue
            module, _, tail = name.partition(".lora_")
            modules.setdefault(module.replace("base_model.model.", ""), {})[tail.split(".")[0]] = param.detach().cpu()
        for module, pair in modules.items():
            if "A" not in pair or "B" not in pair:
                continue
            key = kohya_key(module)
            state[f"{key}.lora_down.weight"] = pair["A"]
            state[f"{key}.lora_up.weight"] = pair["B"]
            state[f"{key}.alpha"] = torch.tensor(float(self.config.lora.alpha))
        # the file is a ready-to-ship LoRA: step/config live in metadata, not tensors
        meta = {"step": str(step), "config": json.dumps(_config_digest(self.config))}
        # checkpoints stay weight-only: a 3x file per step fills the disk for no reason
        save_file(state, path, metadata=meta)
        self.log(f"checkpoint {path}")

        mode = str(getattr(self.config.train, "save_state", "last") or "off").lower()
        if mode in ("1", "true", "yes"):
            mode = "last"
        if mode in ("0", "false", "no"):
            mode = "off"
        if mode != "off":
            # one rolling resume file per run: optimizer + RNG only, never the weights
            resume = {"step": torch.tensor(step)}
            for index, param in enumerate(params):
                moment = self.optimizer.state.get(param)
                if not moment:
                    continue
                for key, value in moment.items():
                    if torch.is_tensor(value):
                        resume[f"optim.{index}.{key}"] = value.detach().cpu()
                    elif isinstance(value, (int, float)):
                        resume[f"optim.{index}.{key}"] = torch.tensor(float(value))
            resume["rng.cpu"] = self.generator.get_state()
            if torch.cuda.is_available():
                for index, rng in enumerate(torch.cuda.get_rng_state_all()):
                    resume[f"rng.cuda.{index}"] = rng
            if mode == "all":
                # one state file per checkpoint: resume from any step, 3x disk
                resume_path = os.path.join(self.ckpt_dir, f"{tag}.state.safetensors")
            else:
                resume_path = os.path.join(self.ckpt_dir, "state.safetensors")
            save_file(resume, resume_path, metadata={**meta, "has_state": "1"})
            self.log(f"training state {resume_path}")

    def _load_checkpoint(self, path: str) -> int:
        from safetensors import safe_open

        data = load_file(path)
        step = 0
        if "step" in data:
            step = int(data.pop("step").item())
        else:
            try:
                with safe_open(path, "pt") as handle:
                    step = int((handle.metadata() or {}).get("step", 0))
            except Exception:
                step = 0
        own = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        loaded = {k: v for k, v in data.items()
                  if not k.startswith(("ema.", "optim.", "rng."))}
        loaded = self._denormalise_keys(loaded)
        missing = [k for k in own if k not in loaded]
        if missing:
            raise RuntimeError(f"checkpoint missing {len(missing)} params, e.g. {missing[:3]}")
        with torch.no_grad():
            for name, param in own.items():
                param.copy_(loaded[name].to(param.device, param.dtype))

        if self.ema and any(k.startswith("ema.") for k in data):
            for key, value in data.items():
                if key.startswith("ema."):
                    self.ema[key[4:]] = value.float()

        restored = 0
        _, params = self._trainable()
        for key, value in data.items():
            if not key.startswith("optim."):
                continue
            _, index, name = key.split(".", 2)
            moment = self.optimizer.state.setdefault(params[int(index)], {})
            moment[name] = value.to(params[int(index)].device)
            restored += 1
        if "rng.cpu" in data:
            self.generator.set_state(data["rng.cpu"].to(torch.uint8))
        for key, value in data.items():
            if key.startswith("rng.cuda."):
                torch.cuda.set_rng_state(value.to(torch.uint8), int(key.rsplit(".", 1)[1]))
        if not restored:
            # weight-only checkpoints pair with one rolling state.safetensors
            folder = os.path.dirname(os.path.abspath(path))
            tag = os.path.splitext(os.path.basename(path))[0]
            sibling = os.path.join(folder, f"{tag}.state.safetensors")
            if not os.path.isfile(sibling):
                sibling = os.path.join(folder, "state.safetensors")
            if os.path.isfile(sibling):
                from safetensors import safe_open as _safe_open

                extra = load_file(sibling)
                if "step" in extra:
                    state_step = int(extra.pop("step").item())
                else:
                    try:
                        with _safe_open(sibling, "pt") as _h:
                            state_step = int((_h.metadata() or {}).get("step", -1))
                    except Exception:
                        state_step = -1
                if state_step == step:
                    for key, value in extra.items():
                        if not key.startswith("optim."):
                            continue
                        _, index, name = key.split(".", 2)
                        moment = self.optimizer.state.setdefault(params[int(index)], {})
                        moment[name] = value.to(params[int(index)].device)
                        restored += 1
                    if "rng.cpu" in extra:
                        self.generator.set_state(extra["rng.cpu"].to(torch.uint8))
        if restored:
            self.log(f"resumed optimizer state ({restored} tensors)")
        else:
            self.log("checkpoint has no optimizer state; resuming weights only")
        self.log(f"resumed from {path} at step {step}")
        return step

    def _denormalise_keys(self, loaded: dict) -> dict:
        """Map kohya ``lora_unet_*`` keys back onto the PEFT parameter names."""
        if not any(k.startswith("lora_unet_") for k in loaded):
            return loaded
        from .lora_file import kohya_key

        mapping = {}
        for name, _param in self.model.named_parameters():
            if ".lora_" not in name:
                continue
            module, _, tail = name.partition(".lora_")
            module = module.replace("base_model.model.", "")
            kind = tail.split(".")[0]
            suffix = "lora_down.weight" if kind == "A" else "lora_up.weight"
            mapping[f"{kohya_key(module)}.{suffix}"] = name
        out = {}
        for key, value in loaded.items():
            if key.endswith(".alpha"):
                continue  # metadata, not a parameter
            out[mapping.get(key, key)] = value
        return out

    # -- export ----------------------------------------------------------
    @staticmethod
    def _scalar(value, default):
        if isinstance(value, dict):
            return float(next(iter(value.values()), default))
        return float(value) if value is not None else float(default)

    def lora_tensors(self):
        """module path -> (A, B, lora_alpha) from the PEFT-wrapped model."""
        alphas = {}
        parts = {}
        for name, param in self.model.named_parameters():
            if ".lora_" not in name:
                continue
            module, _, tail = name.partition(".lora_")
            module = module.replace("base_model.model.", "")
            kind = tail.split(".")[0]
            parts.setdefault(module, {})[kind] = param.detach()
        for module, adapter in self.model.named_modules():
            module = module.replace("base_model.model.", "")
            if hasattr(adapter, "lora_alpha") and module in parts:
                alphas[module] = self._scalar(getattr(adapter, "lora_alpha", None), self.config.lora.alpha)
        out = {}
        for module, tensors in parts.items():
            if "A" in tensors and "B" in tensors:
                out[module] = (tensors["A"], tensors["B"], alphas.get(module, self.config.lora.alpha))
        return out

    def export(self, step: int | None = None, formats=None, suffix: str = ""):
        """Write the LoRA in the two key layouts ComfyUI / diffusers expect.

        kohya:     ``lora_unet_<module>.lora_down/lora_up`` + alpha (what ComfyUI loads)
        diffusers: ``transformer.<module>.lora_A/lora_B`` (PEFT naming)
        """
        formats = list(formats or getattr(self.config.export, "formats", []) or
                       ["kohya"])
        modules = self.lora_tensors()
        if not modules:
            self.log("nothing to export")
            return {}
        step = step or self.state.step
        diffusers = {}
        kohya = {}
        alpha = self.config.lora.alpha
        for module, (a, b, module_alpha) in modules.items():
            diffusers[f"transformer.{module}.lora_A.weight"] = a.contiguous()
            diffusers[f"transformer.{module}.lora_B.weight"] = b.contiguous()
            # Civitai / kohya / A1111-Neo naming: the only scheme every loader maps
            # without a model-specific branch (ComfyUI turns it into
            # lora_unet_<module with underscores> -> diffusion_model.<module>.weight)
            key = "lora_unet_" + module.replace(".", "_")
            kohya[f"{key}.lora_down.weight"] = a.contiguous()
            kohya[f"{key}.lora_up.weight"] = b.contiguous()
            kohya[f"{key}.alpha"] = torch.tensor(float(module_alpha))
        written = {}
        # one format -> plain "<name>.safetensors"; several -> "<name>_<format>"
        tag = (lambda fmt: "") if len(formats) == 1 else (lambda fmt: f"_{fmt}")
        if "kohya" in formats:
            path = os.path.join(self.out_dir, f"{self.config.name}{suffix}{tag('kohya')}.safetensors")
            save_file(kohya, path, metadata={"step": str(step), "alpha": str(alpha)})
            written["kohya"] = path
            self.log(f"exported {path} ({len(kohya)} tensors)")
        if "diffusers" in formats:
            path = os.path.join(self.out_dir, f"{self.config.name}{suffix}{tag('diffusers')}.safetensors")
            save_file(diffusers, path, metadata={"step": str(step), "alpha": str(alpha)})
            written["diffusers"] = path
            self.log(f"exported {path} ({len(diffusers)} tensors)")
        return written

    # -- preview ---------------------------------------------------------
    def preview(self, step: int):
        from .sampling import sample

        path = os.path.join(self.cache_dir, "preview.safetensors")
        if not os.path.exists(path):
            return None
        from PIL import Image

        data = load_file(path)
        if getattr(self, "_codec", None) is None:
            self._codec = LatentCodec(load_vae(self.config.model.vae, dtype=torch.float32, device=self.device))
        codec = self._codec
        was_training = self.model.training
        self.model.eval()
        embeds = data["embeds"].to(self.device, self.dtype)
        images = sample(
            self.model, codec, embeds, data["mask"].to(self.device),
            data["negative_embeds"].to(self.device, self.dtype), data["negative_mask"].to(self.device),
            height=self.config.preview.height, width=self.config.preview.width,
            steps=self.config.preview.steps, guidance=self.config.preview.guidance,
            shift=self.config.preview.shift,
            uncond_skip_layers=tuple(int(x) for x in (self.config.preview.uncond_skip_layers or ())),
            generator=torch.Generator("cpu").manual_seed(self.config.preview.seed),
            dtype=self.dtype,
        )
        out_dir = os.path.join(self.out_dir, "samples")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for index, image in enumerate(images):
            pixels = ((image.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            target = os.path.join(out_dir, f"step{step:06d}_{index}.png")
            Image.fromarray(pixels).save(target)
            paths.append(target)
        self.model.train(was_training)
        self.log(f"preview -> {paths}")
        return paths


    def make_contact_sheet(self):
        """One row per preview step, one column per prompt: the training trajectory."""
        from PIL import Image

        sample_dir = os.path.join(self.out_dir, "samples")
        if not os.path.isdir(sample_dir):
            return None
        files = sorted(f for f in os.listdir(sample_dir) if f.startswith("step") and f.endswith(".png"))
        if not files:
            return None
        steps = sorted({int(f[4:10]) for f in files})
        columns = len([f for f in files if f.startswith(f"step{steps[0]:06d}")])
        frames = [Image.open(os.path.join(sample_dir, f)).convert("RGB")
                  for s in steps for f in sorted(x for x in files if x.startswith(f"step{s:06d}"))]
        if not frames:
            return None
        width, height = frames[0].size
        sheet = Image.new("RGB", (width * columns, height * len(steps)), "black")
        for index, frame in enumerate(frames):
            sheet.paste(frame, ((index % columns) * width, (index // columns) * height))
        path = os.path.join(sample_dir, "trajectory.png")
        sheet.save(path)
        self.log(f"trajectory -> {path}")
        return path


def _config_digest(config: Config) -> dict:
    return {
        "rank": config.lora.rank,
        "alpha": config.lora.alpha,
        "targets": config.lora.targets,
        "layers": list(config.lora.layers or []),
        "dataset": config.dataset.path,
    }
