# MOSS-Image LoRA Trainer Design

**English** · [简体中文](DESIGN.zh-CN.md)

This document specifies the architectural boundaries, core mechanisms, and engineering trade-offs of `moss-image-trainer`. It reuses the model architecture and weight mappings from the official [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) implementation; this repository implements only dataset processing, two-stage feature caching, the Flow Matching training loop, sample previews, VLM caption generation, and a buildless WebUI.

## 1. Project Scope and Boundaries

This repository provides a single-node, single-GPU LoRA fine-tuning toolchain for MOSS-Image-SFT and MOSS-Image-Base, moving the 9B text encoder and FLUX.2 VAE out of the training process via offline pre-caching.

The upstream repository publishes the inference pipeline and ComfyUI nodes without fine-tuning code. MOSS-Image consists of a 5B single-stream Diffusion Transformer (DiT, 42 layers), a frozen Qwen3.5-9B text encoder, and a 32-channel FLUX.2 VAE. Keeping all three components in VRAM during training exceeds consumer GPU capacity. The trainer splits the pipeline into discrete subcommands: it first writes image latents and text features to disk, loads only the 5B DiT in an isolated process for backpropagation, and writes `lora_unet_*` weight files recognized natively by ComfyUI. Every feature runs either from the terminal via [cli.py](<../moss_lora/cli.py>) or by workspace through the WebUI served by [server.py](<../moss_lora/web/server.py>).

This repository enforces three engineering boundaries:
- It trains only low-rank adapters on the DiT (`layers.<0..41>.proj.<q|k|v|o|ffn_up_gate|ffn_down>` and optionally `text_embedder`); it does not implement full-parameter fine-tuning or fine-tune Qwen3.5-9B or the FLUX.2 VAE.
- It targets single-node, single-GPU environments and local filesystems without distributed communication (DDP / FSDP / DeepSpeed), multi-tenant authentication, or remote object storage.
- It runs without external ComfyUI processes or host conda environments, executes all tasks through `.venv/bin/python` at the project root, and never fetches model weights over the network in the background.

This trade-off reflects that character and style fine-tuning is bottlenecked by VRAM capacity rather than cross-node bandwidth. Dropping distributed and full-fine-tuning abstractions lets the training loop optimize directly around single-GPU VRAM tiers (NF4 quantization, CPU block swapping, and variable-length sequence packing), at the cost of not scaling across multiple GPUs to shorten long runs and not rewriting text-tower representations for rare concept injection.

## 2. Overall Architecture

The system operates as a unidirectional data flow—"caption generation → latent caching → text feature caching → training and sampling → export"—while the WebUI acts as a stateless scheduling shell that invokes the CLI via subprocesses.

Inputs, outputs, and entry points for all stages converge inside the workspace directory `workspaces/<WORKSPACE>/`:

```text
workspaces/<WORKSPACE>/
├── train_cache/                      # Training set cache directory (default cache.dir)
│   ├── <uid>.safetensors             # Per-sample cache (latent, text_embeds, text_mask)
│   ├── preview.safetensors           # Preview positive and negative prompt feature bundle
│   └── index.json                    # Cache index (schema: 2)
├── reg_cache/                        # Regularization cache directory (default cache.reg_dir)
│   └── <uid>.safetensors             # Regularization sample cache
├── baseline/                         # LoRA-free baseline comparison samples (baseline_N.png)
├── logs/                             # Job execution logs and config snapshots
│   ├── <job_id>.yaml                 # Actual config snapshot captured at job launch
│   └── <run_id|cache-*|caption-*>.log
├── runs/<run_id>/                    # One run: its config and everything it produced
│   ├── config.yaml                   # This run's complete config; frozen while it trains
│   ├── metrics.jsonl                 # Per-step training loss, eval loss, LR, and VRAM records
│   ├── training.json                 # CLI training process heartbeat (pid / step / time)
│   ├── checkpoints/
│   │   ├── stepXXXXXX.safetensors    # Periodic checkpoints (kohya keys, LoRA weights only)
│   │   ├── final.safetensors         # Final checkpoint
│   │   └── state.safetensors         # Rolling optimizer momentum and RNG state
│   ├── samples/
│   │   ├── stepXXXXXX_N.png          # Periodic preview images and matching .png.json metadata
│   │   ├── seeds.json                # Sampling seeds and prompt summary
│   │   └── trajectory.png            # Full-training preview contact sheet grid
│   └── <name>.safetensors            # Default exported final LoRA file
└── exports/                          # Batch exports and zip archives
```

Process boundaries enforce strict physical isolation. The FastAPI main process ([server.py](<../moss_lora/web/server.py>)) never creates a CUDA context and queries GPU status by invoking `nvidia-smi` in a subprocess (see [system.py](<../moss_lora/web/services/system.py>)), preventing the web server from holding hundreds of megabytes of resident VRAM. When a user launches caching, captioning, training, or export from the UI, [launcher.py](<../moss_lora/web/services/launcher.py>) writes a snapshot of the current configuration to `logs/<job_id>.yaml` and spawns an isolated `python -m moss_lora.cli <subcommand>` subprocess via `subprocess.Popen`. When the subprocess exits, it immediately returns all VRAM and host memory to the operating system.

Subprocess isolation eliminates cross-stage VRAM leaks and CUDA context corruption while guaranteeing identical behavior between the CLI and WebUI, at the cost of paying Python interpreter startup and disk weight-loading latency on every job launch rather than supporting sub-second interactive inference.

## 3. Model Reuse Strategy

Model architecture, timestep scheduling, and weight dequantization reuse the upstream reference implementation, while a thin adapter layer normalizes on-disk weight formats and injects training hooks.

### 3.1 DiT Architecture Adaptation and 4D RoPE

[model.py](<../moss_lora/model.py>) is ported directly from the official `MOSSImageTransformer` (42 single-stream blocks, hidden dimension 3072, 24 heads, head dimension 128, FFN dimension 8192, SiTU-GLU activation, shared timestep modulation `shared_block_modulation`, and 4D axial RoPE) with five training adaptations:
1. Attention backend dispatch: Replaces the upstream attention dispatcher with `optimized_attention`. When the caller passes `cu_seqlens` and `max_seqlens`, it calls FlashAttention-2 `flash_attn_varlen_func` directly (requiring a concatenated sequence with `B=1`); otherwise it calls `F.scaled_dot_product_attention` with `attn_mask`.
2. Block-level checkpointing and swap hooks: Splits `_Block` forward into `attn` and `ffn` halves and wires `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` and `block_swap` hooks into `MOSSImageTransformer._run_block`.
3. Multiple batching paths: Adds `_forward_batched` (vectorized forward for equal-length sequences), `forward_packed` (concatenated forward for variable-length sequences), and `forward_padded` (token-dimension padded forward).
4. 4D axial positional encoding: `_RoPE` splits the 128-dim head dimension into four 32-dim axes `(r, f, h, w)` with frequency base `base=10000.0` always computed in `float32` on CPU. Sequence order is fixed as image tokens followed by text tokens (`torch.cat((image, text), 0)`): image tokens set reference axis $r=0$, temporal axis $f$ to the valid text length $L$, and spatial axes $(h, w)$ centered by subtracting $(H-1)/2$ and $(W-1)/2$ (accepting an explicit `center` argument to prevent phase drift under canvas padding); text tokens use coordinates $(0, 0 \dots L-1, 0, 0)$.
5. Skip Layer Guidance (SLG): Adds a `skip_layers` tuple to bypass specified blocks (layer 22 by default) when evaluating the unconditional branch during sampling.

### 3.2 Weight Format Conversion and ConvRot Dequantization

The weight loading layer ([weights.py](<../moss_lora/weights.py>), [convrot.py](<../moss_lora/convrot.py>), and [hf_text.py](<../moss_lora/hf_text.py>)) reads three local weight layouts without depending on ComfyUI:
- DiT fused projection splitting: Official and ComfyUI single-file checkpoints fuse attention projections into `layers.<i>.proj.qkv.weight` (shape `[9216, 3072]`). At load time, `split_fused_qkv` splits dimension 0 via `chunk(3, dim=0)` into separate `q.weight`, `k.weight`, and `v.weight` tensors and pops `rope.bands` so `_RoPE` rebuilds them dynamically (`text_embedder.blocks.*.qkv` remains fused).
- FLUX.2 VAE key mapping and latent convention: `legacy_vae_to_diffusers` maps legacy single-file VAE keys (such as `.mid.attn_1.`, 4D `1x1` convolution weights, `nin_shortcut`, and reversed `decoder.up.<3-i>` blocks) onto `diffusers.AutoencoderKLFlux2`. `LatentCodec` reproduces the exact upstream pipeline order:
  1. Encode: Pass the image through `vae.encode` and take the distribution mode `dist.mode()` to obtain `[B, 32, H/8, W/8]`; apply $2\times 2$ space-to-depth rearrangement via `pack_latents` to `[B, 128, H/16, W/16]` (total downsample stride 16); normalize with `vae.bn` `running_mean` and `running_var` (`BN_EPS = 1e-4`) as $(z - \mu) / \sqrt{\sigma^2 + \epsilon}$.
  2. Decode: Multiply by the standard deviation and add back the mean, restore 32 channels via `unpack_latents`, and pass through `vae.decode`.
- Qwen3.5-9B text tower and ConvRot dequantization: `hf_text.load_text_model` instantiates the text-only `Qwen3_5TextModel` directly on the target device (skipping slow random initialization on CPU), strips `model.language_model.`, `language_model.`, `text_model.`, or `model.` prefixes via `convrot.normalize_key`, and drops `visual.`, `vision.`, `mtp.`, `lm_head.`, `audio.`, and `projector.`. When `model.text_encoder` points to a ConvRot int8 single-file checkpoint, `convrot.dequantize` reads `.weight_scale` and the `.comfy_quant` JSON marker (a metadata key name in the packaged format) and constructs a block-normalized Hadamard matrix:

$$H_4 = \begin{bmatrix} 1 & 1 & 1 & -1 \\ 1 & 1 & -1 & 1 \\ 1 & -1 & 1 & 1 \\ -1 & 1 & 1 & 1 \end{bmatrix}, \quad H = \frac{1}{\sqrt{G}} \bigotimes H_4$$

  With block size $G$ taken from `convrot_groupsize` (default 256), it reconstructs `bfloat16` weights in memory via $W = (q \cdot s) \operatorname{blockdiag}(H)$, matching `comfy_kitchen.dequantize_int8_convrot_weight_dtype` bit-for-bit. Tokenizer and text-tower configuration files are bundled under `moss_lora/assets/` so single-file weights run without an external `config.json`.

### 3.3 Scheduler and Noise Sampling

Inference and training timestep schedules live in [scheduler.py](<../moss_lora/scheduler.py>) and [sigma.py](<../moss_lora/sigma.py>):
- `MOSSImageScheduler` subclasses `FlowMatchEulerDiscreteScheduler`, generates a shifted grid $\sigma' = \frac{s \cdot \sigma}{1 + (s - 1)\sigma}$ using inference `shift` (default `5.0`), snaps each point onto the 1000-step discrete training grid (`training_shift=1.0`) via `torch.searchsorted` with deduplication (`torch.unique_consecutive`), and resets the base class `_shift` to `1.0` to prevent double shifting.
- `SigmaSampler` samples $\sigma \in (0, 1]$ during training according to `train.sigma.mode`: the default `logit_normal` mode uses `mu` (`-1.09`), `sigma` (`0.95`), upper truncation `cutoff` (`0.95`), and `uniform_mix` (`0.01`) to precompute a probability table over grid points $j \in \{1 \dots 1000\}$ ($x = a \cdot j / 1000$) from the cumulative distribution function $F(x) = 1 - \Phi\left(\frac{\log((1-x)/x) - \mu}{s}\right)$, while also supporting `uniform` and `shifted` modes.

Reusing upstream modules directly and confining format conversion to the loading boundary guarantees mathematical equivalence between training forwards and official inference, at the cost of splitting `qkv` and dequantizing the `int8` text tower at load time rather than injecting a single joint LoRA matrix into fused `qkv`.

## 4. Data Layer

The data layer manages multi-style annotations through non-intrusive sidecar JSON files (`schema: 2`) alongside per-image sizing, directory repeat multipliers, and regularization datasets.

### 4.1 Scanning and Caption Priority

`scan` in [data.py](<../moss_lora/data.py>) recursively traverses `dataset.path` for `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files. Each image resolves its training caption through a four-level priority chain (stopping at the first hit):
1. `dataset.metadata`: When `metadata.jsonl` is configured, reads the `caption` / `prompt` / `text` field matching `file_name` / `image` / `path` (`caption_source="metadata"`).
2. Sidecar JSON: Reads the field selected by `dataset.moss_caption_source` from `<stem>_moss-image-trainer.json` (`caption_source="moss"`).
3. Matching text file: Searches for a same-stem caption file in `dataset.caption_ext` order (default `[".txt", ".caption"]`, `caption_source="file"`).
4. Default fallback text: When all sources above are empty and a default caption is supplied (`dataset.reg_caption` for regularization sets), uses the default text (`caption_source="default"`). `CacheBuilder._caption` then prepends `dataset.caption_prefix` and appends `dataset.caption_suffix` before encoding.

### 4.2 Sidecar v2 Schema and Dual-Style Generation

The caption generation module [captioner.py](<../moss_lora/captioner.py>) calls any OpenAI-compatible local vision endpoint over HTTP and writes results to `<stem>_moss-image-trainer.json` next to each image. Service probing starts with a 3-second TCP connection check (`_tcp_ok`) to catch blackholed ports that drop SYN packets, queries `/models` to verify readiness, and probes `/status` or `/props` to detect whether the backend supports `json_schema` constrained decoding (`--structured auto|on|off`). For models with thinking mode enabled, `last_json_object` matches balanced braces backward from the end of the response to extract the JSON payload and stores preceding reasoning text in `_analysis`.

Sidecar files use a fixed `schema: 2` layout holding two independent style dictionaries:

```json
{
  "schema": 2,
  "tool": "moss-image-trainer",
  "version": 2,
  "created": "2026-10-20T12:00:00+0900",
  "role": "train",
  "tag_caption_style": {
    "caption": "<trigger>, 1girl, solo, standing in a sunlit courtyard, medium shot",
    "kept_tags": ["1girl", "solo"],
    "scene": "standing in a sunlit courtyard, medium shot",
    "reference": "1girl, solo",
    "trigger": "<trigger>",
    "analysis": "Medium shot from eye level...",
    "watermark": [],
    "borders": ["none"],
    "generated": { "model": "...", "temperature": 1.0, "top_p": 0.95, "max_tokens": 16384, "thinking": true }
  },
  "official_style": {
    "caption_long": "<trigger>. Detailed composition...",
    "caption_medium": "<trigger>. One-paragraph summary...",
    "caption_short": "<trigger>. One-sentence summary.",
    "trigger": "<trigger>",
    "analysis": "...",
    "watermark": [{ "box_2d": [920, 800, 980, 990], "label": "signature" }],
    "borders": ["none"],
    "generated": { "model": "...", "temperature": 1.0, "top_p": 0.95, "max_tokens": 16384, "thinking": true }
  }
}
```

Generation and validation rules for the two styles work as follows:
- `tag_caption_style` (CLI `--styles tag_caption`): Designed for character fine-tuning that preserves trigger words and Danbooru tags. `reference_tags` reads the matching `.txt` file first and falls back to `kept_tags` from an existing sidecar (skipping the image if neither exists). `clean_tags` strips tags matching the trigger word, deduplicates the rest, and fills the `{tags}` placeholder in `TAGS_PROMPT`, instructing the VLM to output a 2–4 sentence `analysis` (at least 10 words) followed by a `scene` of at most 25 words covering only pose, framing, camera angle, background, and lighting. `validate_scene` rejects scenes exceeding 30 words or containing identity terms such as `hair`, `blonde`, `brunette`, `skin color`, `breast`, or `<color> eyes` (exempting identity filtering and omitting the trigger word when `role="reg"`). `compose_caption` joins the trigger word, `kept_tags`, and `scene` with commas into `caption`.
- `official_style` (CLI `--styles official`): Follows the official recaptioner specification (`NATURAL_PROMPT`), ignoring existing `.txt` files and instructing the VLM to produce `caption.long` (preserving non-watermark in-image text and line breaks verbatim), `caption.medium`, and `caption.short` (validated at minimum lengths of 8, 8, and 3 words), prepending `<trigger>. <body>` when a trigger is configured. Both styles validate `borders` (allowing only `left|right|up|down` or `["none"]`) and `watermark` (`0..1000` normalized coordinates `[y_min, x_min, y_max, x_max]`, fixing inverted coordinates and dropping zero-area boxes). `merge_sidecar` overwrites only the active style key using atomic `.tmp` + `os.replace` writes, while `migrate_sidecar` (`--migrate`) upgrades legacy flat sidecars to `schema: 2` losslessly.

### 4.3 Per-image Sizing, Repeats, and Regularization

Image sizing and sampling weights are computed in [data.py](<../moss_lora/data.py>) and [cache.py](<../moss_lora/cache.py>):
- Per-image sizing: `assign_sizes` gives every image its own dimensions. `target_size` preserves the aspect ratio, caps the pixel area at `dataset.max_area` (default `1048576`, with no upper bound), and aligns both sides down to `ALIGN = 16`; it never upscales an image that already fits. There is no fixed aspect-ratio set: self-attention runs over tokens, so a batch does not need a shared canvas, and the token budget decides how many images share a step.
- Repeats: When a subdirectory name matches `^(\d+)[_-](.+)$` (such as `3_concept`), the base repeat count takes the integer prefix and multiplies it by the global `dataset.repeats`.
- Regularization: When `dataset.reg_path` is set, `CacheBuilder` scans that directory and marks entries with `reg=True`, multiplying any subdirectory integer prefix by `dataset.reg_repeat` to prevent trigger concepts from drifting into generic class priors.

Storing generated captions in sidecar JSON files and selecting the active field via `dataset.moss_caption_source` (`tag_caption|official_long|official_medium|official_short`) lets a single image folder feed both tag-based and natural-language training runs without duplicating images, at the cost of doubling file count in the dataset directory and requiring pre-existing `.txt` tag files for `tag_caption`.

## 5. Cache Layer

The cache layer splits VAE latent encoding and Qwen3.5-9B text encoding into two sequential stages, hashes sample paths to disk filenames, and tracks staleness through `index.json` (`schema: 2`) and safetensors metadata headers.

Each sample filename is derived by `item_uid`: `hashlib.sha1(os.path.abspath(item.image).encode()).hexdigest()[:16] + ".safetensors"`. Training samples write to `cache.dir` (`train_cache/` under the workspace by default), while regularization samples write to `cache.reg_dir` (defaulting via `reg_cache_dir` to a sibling `reg_cache/` directory when unset), physically separating the two sets during cleanup and inspection.

The two encoding pipelines are driven by [cache.py](<../moss_lora/cache.py>) and [encode.py](<../moss_lora/encode.py>):
1. Latent encoding (`CacheBuilder.run_latents`): Loads the FLUX.2 VAE in `float32` and groups items by target dimensions; a CPU `ThreadPoolExecutor` (default 6 workers) prefetches images, resizes them with bicubic interpolation to each item's target size, and normalizes pixels to $[-1, 1]$ in overlap with GPU encoding. Batch size comes from `cache.vae_batch` when set, otherwise `latent_batch` in [memory.py](<../moss_lora/memory.py>) computes it as 80% of free VRAM divided by the per-image footprint estimate. The only ceiling is the number of images waiting, so the batch follows the card rather than an assumption about the host: a value that exceeds the estimate is logged as an OOM warning instead of being silently clamped. Output latents are cast to `float16` and saved under the `latent` key in `<uid>.safetensors`. An existing sample file is skipped unless `cache.overwrite` is set; the WebUI and `cache --overwrite` expose the same switch, and the chained text stage only re-encodes when text encoding was also selected.
2. Text feature encoding (`run_encode_text`): Uses `HfTextEncoder` in [hf_text.py](<../moss_lora/hf_text.py>) to wrap each caption in the fixed chat template `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n` (up to `MAX_TOKENS = 4096` in that module, `PAD_ID = 248044`). The loader supports in-memory weight compression via `--quantize none|fp8|nf4_ffn|nf4`. Pending captions are sorted by character length to minimize intra-batch padding, with batch size from `cache.text_batch` when set, otherwise chosen dynamically by `memory.text_batch` (capped at `TEXT_BATCH_CAP = 64`); batches apply right-side causal padding, extract `last_hidden_state` and boolean `attention_mask`, and write them into `text_embeds` and `text_mask` inside the same `<uid>.safetensors`. Because Qwen3.5 uses causal attention, right-side padding never contaminates preceding valid token states. When `preview.prompts` is configured, this stage also encodes positive and negative preview prompts (with an empty negative prompt still encoded into 10 chat-template tokens) into `train_cache/preview.safetensors`.

On every write, both `CacheBuilder._write` and `run_encode_text` store `image`, `caption` (the full training caption with prefix and suffix), `width`, `height`, and `repeats` inside the safetensors `metadata` header, then write `train_cache/index.json`:

```json
{
  "schema": 2,
  "cache_dir": "<WORKSPACE>/train_cache",
  "reg_dir": "<WORKSPACE>/reg_cache",
  "caption_source": "tag_caption",
  "batch": { "latent": 4, "text": 64 },
  "items": [
    {
      "file": "<uid>.safetensors",
      "image": "<DATASET_DIR>/001.png",
      "caption": "...",
      "width": 1024,
      "height": 1024,
      "repeats": 1,
      "reg": false,
      "text_len": 42
    }
  ]
}
```

For regularization entries, `file` stores a path relative to `train_cache` (`../reg_cache/<uid>.safetensors`), allowing the trainer to resolve every sample uniformly with `os.path.join(cache_dir, entry["file"])`.

Batch size belongs in that record because it changes the numbers for text. A caption encoded alone and the same caption inside a batch of 64 reach the GPU as differently shaped matmuls, land on different kernels, and come back a few `1e-4` apart. Latents do not move with the batch: `LatentCodec` turns cuDNN TF32 off, so a VAE encode is bit-identical at batch 1 and batch 6, and faster for it (167 ms per image against 268). `batch.latent` comes from `CacheBuilder.run_latents`, `batch.text` from `run_encode_text`; running one phase alone keeps the other phase's value instead of erasing it. An index written before this field existed has no `batch`, which is the honest answer for a cache nobody recorded.

Cache invalidation and incremental re-encoding follow three rules:
- Incremental latent check: `CacheBuilder._pending("latent")` checks whether `<uid>.safetensors` exists and contains a `latent` tensor, skipping the sample when present and `cache.overwrite: false`.
- Incremental and stale text check: `run_encode_text` reads `metadata["caption"]` from each cached safetensors header and compares it against the target caption computed from the current config. Switching `dataset.moss_caption_source`, editing `caption_prefix` / `caption_suffix`, or updating a sidecar JSON triggers selective re-encoding of only the mismatched entries while preserving the existing `latent` tensor without re-running the VAE.
- Pre-training readiness check: `_cache_summary` in [cache_cmd.py](<../moss_lora/web/services/cache_cmd.py>) requires `index.json` to exist with `schema == 2` and `text_len > 0` on every item; otherwise `ready` evaluates to `false` and both the WebUI and CLI refuse to start training.

Combining `latent` and `text_embeds` inside a single `<uid>.safetensors` file reduces each training step to one file open per sample and enables caption-only incremental updates via header metadata, at the cost that a change to `dataset.max_area` invalidates every cached latent, since `_pending("latent")` checks key presence rather than latent shape; the index carries a `sizing` fingerprint (`mode`, `area`, `align`), so the builder re-encodes latents automatically and a trainer refuses a cache whose fingerprint differs from the current config.

## 6. Training Layer

The training layer implements the standard Flow Matching velocity objective and combines PEFT LoRA, NF4 frozen-layer quantization, CPU block swapping, and three batched attention modes through VRAM presets.

### 6.1 Training Objective and Batching Modes

The loss function in [trainer.py](<../moss_lora/trainer.py>) matches the official technical report: given clean latents $x_0$, standard normal noise $\epsilon \sim \mathcal{N}(0, I)$, and sampled $\sigma \in (0, 1]$, it forms the linear interpolation $x_t = (1 - \sigma) x_0 + \sigma \epsilon$, feeds scaled timestep $t = 1000\sigma$ and text features into the DiT to predict velocity $v_\theta(x_t, 1000\sigma, c)$, and computes the mean squared error against target velocity $v = \epsilon - x_0$ over image tokens via `F.mse_loss(prediction.float(), noise - x0.float())` without SNR weighting.

Because the upstream DiT slices text sequences to each sample's valid length, `train.batch_mode` offers three batching paths (paired with `train.attention: auto|flash|sdpa`):
Every micro-batch is assembled by `_sample_by_tokens`: it draws samples in random order until the token budget (`train.bin_tokens`, or $\text{max\_area} / 256 + 1024$, one full-size image plus text when `0`) is spent, weighting each sample by its `repeats`. Samples keep their own resolutions and text lengths; `train.batch_mode` picks how they are fused, paired with `train.attention: auto|flash|sdpa`:
- `auto` (default): resolves to `packed` when FlashAttention-2 can split sequences, otherwise `padded`.
- `packed`: `forward_packed` concatenates the samples into one sequence and shares a single scalar timestep modulation. With FlashAttention-2 it builds `cu_seqlens` and calls `flash_attn_varlen_func` with zero padding and zero wasted FLOPs; without it, the samples are separated by a dense block-diagonal boolean mask on SDPA.
- `padded`: `forward_padded` flattens each sample into a 1D token sequence, pads along the sequence dimension to the batch maximum $L_{\max}$, and passes a `token_mask` to SDPA, avoiding 2D canvas padding waste.

### 6.2 LoRA Injection, Optimizers, and Scheduling

`Trainer._attach_lora` injects adapters via `peft.get_peft_model` using the regex `^layers\.<layer_expr>\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)$` (controlled by `lora.targets` and `lora.layers`) so inner layers of `text_embedder` are never matched accidentally; `text_embedder\.blocks\.\d+\.(qkv|proj|ffn_up|ffn_gate|ffn_down)$` is appended only when `lora.include_text_embedder: true`. Weights use Kaiming initialization by default, or zero-initialize `lora_B` when `lora.init: zero`.

`train.optimizer` supports four configurations:
- `adamw` (default): `torch.optim.AdamW` with fixed $\beta=(0.9, 0.95)$.
- `adamw8bit` (or `vram.optimizer_in_8bit: true`): Uses `bitsandbytes.optim.AdamW8bit` to compress first and second moments.
- `muon`: Imports upstream `muon.SingleDeviceMuon` when installed and falls back to the built-in [muon.py](<../moss_lora/muon.py>). Built-in `Muon` maintains a single momentum buffer ($\mu=0.95$, Nesterov momentum) for 2D LoRA matrices, orthogonalizes updates with 5 Newton-Schulz iterations (coefficients $3.4445, -4.7750, 2.0315$), and scales steps by $0.2 \sqrt{\max(m, n)}$ to match AdamW learning-rate magnitudes while halving optimizer state memory.
- `adafactor`: `transformers.Adafactor` (`scale_parameter=False, relative_step=False`).

Learning rate updates per step via `_lr_at`: `train.warmup` values in `(0, 1)` are interpreted as a fraction of total steps, while values $\ge 1$ specify an exact step count, followed by `train.lr_scheduler` (`constant` or `cosine`) decay. After accumulating `grad_accum` backward passes, each step clips gradients at `train.max_grad_norm` (default `1.0`) and updates the optional FP32 exponential moving average dictionary (`train.ema`).

### 6.3 Precision, Quantization, Block Streaming, and OOM Self-Healing

VRAM controls are centralized in `VRAM_PRESETS` in [config.py](<../moss_lora/config.py>), [quant.py](<../moss_lora/quant.py>), [memory.py](<../moss_lora/memory.py>), and [offload.py](<../moss_lora/offload.py>):

| Preset | `gradient_checkpointing` | `quantize_dit` | `blocks_to_swap` | `optimizer_in_8bit` | `compile` | `preview.vae_dtype` |
| --- | --- | --- | --- | --- | --- | --- |
| `32` / `24` | `true` | `none` | `0` | `false` | `true` | `fp32` |
| `16` / `12` | `true` | `ffn` | `0` | `false` | `true` | `fp32` |
| `8` / `6` | `true` | `all` | `42` | `true` | `false` | `fp16` |

`apply_preset` writes every value in that row into the config when a human picks a tier, and nothing reads the tier afterwards. The run reads the fields, so `quantize_dit: none` in the file means an unquantized run no matter which preset was applied earlier, and a config with no preset key at all is a complete run. This replaced a resolution step that filled in sentinels (`gradient_checkpointing: auto`, `blocks_to_swap: -1`) from a tier chosen by the detected card: it made the YAML, the form and the run disagree, and the "automatic" part was a capacity lookup, not an optimization. Two retired sentinels are now folded on load by `Config.normalize` -- `auto` to `true`, `-1` to `0` -- so old configs still start. Every count in the config counts blocks, so it ranges 0-42: block 42 does not exist, blocks are numbered 0-41.

Underlying mechanisms operate as follows:
- Precision and accumulators: `model.dtype` defaults to `bfloat16`; setting `float16` enables `GradScaler` automatically, and pairing it with `vram.fp16_accumulate: true` sets `torch.backends.cuda.matmul.allow_fp16_accumulation = True` for higher half-precision Tensor Core throughput on consumer GPUs.
- Weight-only DiT quantization: `vram.quantize_dit` selects the scope — `ffn` replaces the two FFN projections, `all` replaces QKVO plus FFN in every block — and `vram.quantize_dtype` selects the storage format. `nf4` swaps each module for `bitsandbytes.nn.Linear4bit` (`quant_type="nf4"`, `compress_statistics=True`); `fp8` swaps it for [FP8Linear](<../moss_lora/quant.py>), which stores the weight as `float8_e4m3fn` with one scale per output row and widens it to the compute dtype for the matmul. That weight is a frozen `Parameter` rather than a buffer, because block streaming walks `parameters()` to decide what may leave the card: as a buffer it stayed resident and made `blocks_to_swap` a silent no-op for fp8 runs. Both leave the adapter in `bfloat16` and pass the gradient through to the input; only frozen modules — those without `lora_` children — are replaced, so trainable weights are never rounded. The model is built on the meta device and filled one block at a time (`moss_lora/weights.py`, `LOAD_MODE = "streamed"`): each block is read from the checkpoint, packed on the card, and either kept there or moved to host memory if it belongs to the streamed tail. The bf16 checkpoint therefore never exists as a second copy of host memory, and the random initialisation the constructor would run never happens — the full path (`LOAD_MODE = "full"`) reads everything and fills the model in one go, and exists only to compare the two.
- Conditioning half: `vram.quantize_conditioning` extends either format to the 16 frozen Linear layers outside `layers.*` — the text projection and its two blocks, both modulations, the time embedding, and the final projection. It is `none` by default and inert unless `quantize_dit` is set, because a per-block modulation is multiplied into every block and its rounding error compounds instead of averaging out.
- Block streaming: `vram.blocks_to_swap = N` keeps the leading `42 - N` blocks resident and streams the trailing `N`, up to the whole stack at `42`. `preview.blocks_to_swap` overrides the residency while sampling and restores the training value afterwards; `-1` follows training. The WebUI presents the complement — how many of the 42 blocks stay resident, so 0-42 — because that is the number a user reasons about, and the stored key is unchanged. Streaming requires `gradient_checkpointing`. Frozen base weights of a streamed block live in pinned host memory while its LoRA weights stay on the card; `_SwapCheckpoint` copies the base weights in before the block's forward and backward recomputation and releases them afterwards, so one streamed block's base weights occupy the card at a time. A one-block prefetch window overlaps the copy of the next block with the current block's compute, and the copies run on their own stream with `record_stream` so the caching allocator cannot recycle a buffer while a copy into it is still in flight. Released GPU buffers are pooled rather than freed: every block has identical shapes, so one set is handed to the next upload. What streaming saves on the card depends on the allocator in `runtime.allocator`: `expandable_segments` returns the whole-block segments the offload frees, while `backend:cudaMallocAsync` pools them per stream and hands nothing back, so a streamed run allocates much less but occupies the same card. The trainer prints a warning when it sees the second case, because the feature's whole point is the card footprint. Allocating and freeing a block's worth of storage on every load -- 84 transients per step -- never coalesces, and the reserved footprint climbs step after step until the card is full, which defeats the memory the streaming was meant to return.
- Dynamic OOM degradation: When the training loop catches `torch.cuda.OutOfMemoryError`, `_degrade_memory` increases `blocks_to_swap` by 8 blocks up to the number of transformer blocks in the model, clears the CUDA cache, and retries the step immediately. The trainer therefore degrades towards streaming instead of dying, and a run started with too little headroom recovers on its own.
- Compilation: `vram.compile: true` wraps `model.forward` with `torch.compile(..., dynamic=True, fullgraph=False)` and falls back to eager mode if compilation fails. It is refused outright when any block streams: the swap path records a CUDA event per copy, which Dynamo cannot trace, and the failure only surfaces at the first eval rather than at startup. A streamed run gains nothing from compilation anyway.
- Telemetry and estimate: `_memory_stats` samples allocated and reserved VRAM plus the device name into every metrics row and log line, so `metrics.jsonl` records the peak each run actually reached. `memory.estimate_vram` answers the same question before a run exists: it walks the config, sums the frozen weights under the chosen quantization, the checkpointed activations, the LoRA and optimizer state, the preview path, and a fixed CUDA-context allowance, and returns the training and preview peaks separately with the total. `POST /api/estimate` exposes it to the dashboard, which shows the predicted peak next to the measured one so drift between the two stays visible instead of silent.

### 6.4 Validation, Sampling, Checkpoints, and Resumption

- Deterministic holdout validation: When `train.eval_holdout > 0` (default `0.05`), `_split_eval` hashes each image's absolute path with SHA1 and takes the first 8 hex digits modulo 1000 to partition the validation set. Hash-based splitting keeps the validation set invariant across runs and resumptions; every `train.eval_every` steps, the trainer evaluates `train.eval_batches` batches with fixed seed `1234` and appends the validation loss to `metrics.jsonl`.
- Periodic sampling and trajectory sheet: Every `train.sample_every` steps, `sample` in [sampling.py](<../moss_lora/sampling.py>) generates preview images, and at the end of training `make_contact_sheet` stitches them into `samples/trajectory.png` with one row per checkpoint step and one column per prompt. Sampling exceptions are caught and logged without interrupting the training loop. When `vram.compile` is on, `_preview_forward` swaps the model's uncompiled forward in for the sampling call and restores the compiled one afterwards: a preview runs a batch size and a token shape the training loop never traced, so the first compiled preview would spend minutes in graph capture with nothing in the log, which a user reads as a hang. The preview batch (`preview.batch`, `0` = every prompt in one pass) still controls the sampling peak.
- Separated checkpoints and optimizer state: Every `train.save_every` steps, `save_checkpoint` writes LoRA-only weights using `lora_unet_*` keys to `checkpoints/stepXXXXXX.safetensors` (and `final.safetensors` at completion) with step count and config summary in the `metadata` header. Optimizer momentum and CPU/CUDA RNG states are written separately according to `train.save_state` (`off|last|all`, default `last`) to a rolling `checkpoints/state.safetensors` or step-tagged `<tag>.state.safetensors`.
- Resumption and queuing: When `--resume <ckpt>` is passed, `_load_checkpoint` maps `lora_unet_*` keys back to PEFT parameter names via `_denormalise_keys`, locates the matching `.state.safetensors` or `state.safetensors` in the same directory, verifies `state_step == step`, and restores optimizer momentum and RNG states. `run_pipeline` in [queue.py](<../moss_lora/queue.py>) scans a workspace for runs whose `metrics.jsonl` has not reached target steps, builds any missing caches, and resumes each run from its own `runs/<run>/config.yaml` sequentially.

Separating weights from optimizer state shrinks each checkpoint by two-thirds and makes every saved checkpoint directly loadable for inference, at the cost that under the default `save_state: last`, only the latest checkpoint retains matching optimizer momentum—resuming from an earlier historical step restores LoRA weights while reinitializing optimizer momentum from scratch.

## 7. Export

The export layer uses ComfyUI-native kohya `lora_unet_*` keys as the internal checkpoint format and provides sub-second model-free format conversion and batch packaging without loading the 5B model.

Both periodic checkpointing and final export convert PEFT module paths `base_model.model.layers.<L>.proj.<M>` into the standard community kohya key layout:

```text
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_down.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_up.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.alpha
```

ComfyUI's built-in `Load LoRA` node automatically maps the `lora_unet_` prefix and underscore path onto `diffusion_model.layers.<L>.proj.<M>.weight` (even when the underlying ComfyUI model uses fused `qkv`, its loader slices separate `q`/`k`/`v` LoRA weights into the corresponding channels). When the user requests `diffusers` format (`export.formats: [diffusers]`), keys are written as `transformer.layers.<L>.proj.<M>.lora_A.weight` and `.lora_B.weight`. Exporting a single format writes `<name>.safetensors`; exporting both formats simultaneously appends `_kohya` and `_diffusers` suffixes automatically.

Offline and batch exports are implemented in [lora_file.py](<../moss_lora/lora_file.py>) and [export_batch.py](<../moss_lora/export_batch.py>):
- Model-free conversion: `lora_file.convert` opens the checkpoint `.safetensors` on CPU, parses layer indices and projection names with `^lora_unet_layers_(\d+)_proj_(q|k|v|o|ffn_up_gate|ffn_down)\.lora_(down|up)\.weight$`, renames the keys, and writes the output directly to disk without instantiating the 5B DiT. When the target format is only `kohya`, `cli.py export` and `export_batch` copy the checkpoint verbatim via `shutil.copyfile`.
- Step-range batch export: `export_batch.parse_range` parses step ranges (`250-2000:250`), comma-separated lists (`250,500,1000`), single steps, `final` (represented internally by sentinel `FINAL = 0`), or an empty string (exporting all available steps). Passing `--state` copies `state.safetensors` alongside the weights, and passing `--zip` packages the files with `ZIP_STORED` (bundling without recompressing floating-point tensors).

Storing checkpoints directly as `lora_unet_*` makes every intermediate checkpoint a ready-to-use LoRA and reduces export to a file copy or dictionary key rename, at the cost that `_kohya_module` parses underscore paths against an explicit vocabulary—any module outside `layers.<n>.proj.<module>` and `text_embedder.blocks.<n>.<module>` is skipped unless both regex tables are updated together.

## 8. WebUI Backend

The WebUI backend follows a three-tier FastAPI architecture—"routers → services → shared state and job registry"—combining a global single-job lock, SSE event streaming, and disk fallback recovery across server restarts.

The codebase is partitioned into five responsibilities:
- Application assembly ([server.py](<../moss_lora/web/server.py>) and compatibility shim [app.py](<../moss_lora/web/app.py>)): `create_app` registers 11 `APIRouter` modules in fixed order ([i18n.py](<../moss_lora/web/routers/i18n.py>), [system.py](<../moss_lora/web/routers/system.py>), [settings.py](<../moss_lora/web/routers/settings.py>), [workspaces.py](<../moss_lora/web/routers/workspaces.py>), [dataset.py](<../moss_lora/web/routers/dataset.py>), [config.py](<../moss_lora/web/routers/config.py>), [jobs.py](<../moss_lora/web/routers/jobs.py>), [export.py](<../moss_lora/web/routers/export.py>), [files.py](<../moss_lora/web/routers/files.py>), [validate.py](<../moss_lora/web/routers/validate.py>), [samples.py](<../moss_lora/web/routers/samples.py>)), calls `_flatten_included_routes` to flatten sub-routers into the top-level route table so literal paths (such as `/api/i18n/template` and `/api/i18n/current`) always precede wildcard routes (`/api/i18n/{code}`), and mounts `moss_lora/web/static` at `/`.
- Shared state ([deps.py](<../moss_lora/web/deps.py>)): At startup, it checks `sys.prefix/pyvenv.cfg` and exits immediately if not running inside a virtual environment. The module holds `threading.RLock`-protected global snapshots `SETTINGS`, `RESOLVED`, and `FIRST_RUN`, plus the memory-only plaintext vision API key `_LITERAL_API_KEY`. All text file writes go through `_write_text` using PID- and random-suffixed atomic replacement.
- Job model and lifecycle ([jobs.py](<../moss_lora/web/jobs.py>) and [launcher.py](<../moss_lora/web/services/launcher.py>)): The global singleton `JOBS` tracks all in-memory `Job` instances. `JOBS.active()` enforces a project-wide single-job mutex: at most one `train`, `cache`, or `caption` job can be `running` at any time, and concurrent launch requests receive `HTTP 409`. A background daemon thread runs `_run(job)`, executes `pre_commands` sequentially (such as pre-training LoRA-free `baseline` sampling, where failures emit a `warning` event without aborting training), and spawns the main command via `subprocess.Popen(..., start_new_session=True)`. `_stream_command` reads merged stdout lines in real time, writes raw lines to `logs/<log_name>.log`, parses step progress with `STEP_RE`, and appends events to `job.events` (capped at 5000 entries, trimming the oldest 1000 on overflow); upon normal training completion, `_write_preview_sidecars` writes `<filename>.json` and `seeds.json` for every sample image. When stopping a job, `_terminate` sends `SIGTERM` to the entire process group (`os.killpg`) so child processes spawned under `bash -c` exit cleanly.
- Real-time streaming and CLI job discovery ([routers/jobs.py](<../moss_lora/web/routers/jobs.py>) and [routers/workspaces.py](<../moss_lora/web/routers/workspaces.py>)): `GET /api/jobs/{job_id}/events?from_index=N` serves a `text/event-stream` SSE feed, polling new events every 0.3 seconds and emitting `: keep-alive` comment frames every 15 seconds of inactivity so the frontend can resume losslessly from its last received event index after a disconnect. Training runs started directly from the terminal CLI bypass `JOBS`: the trainer updates `runs/<run>/training.json` (recording `pid`, `step`, `total`, and `time`) every 20 steps, and `/api/workspaces/{name}/active` and `/stop` combine that heartbeat file with `/proc/<pid>/cmdline` inspection to detect and stop external CLI runs.
- Static and media serving ([services/dataset.py](<../moss_lora/web/services/dataset.py>) and [routers/files.py](<../moss_lora/web/routers/files.py>)): `/api/thumb` uses Pillow to generate WebP thumbnails keyed by the SHA1 of source path, modification time, byte size, and target width (default `384px`), storing sample thumbnails under `runs/<run>/.thumbs/` (cleaned up automatically when a run is deleted) and dataset thumbnails under workspace `.thumbs/` with `Cache-Control: no-cache`. `/api/files` serves read-only source images and `.safetensors` files matching an extension allowlist with `Content-Disposition: attachment; filename=...` so browsers preserve file extensions.

Pairing the in-memory event queue with on-disk `logs/*.log` and `metrics.jsonl` delivers sub-second updates over active SSE connections while reconstructing full loss curves and logs after browser refreshes or server restarts, at the cost that the in-memory `JOBS` registry does not persist across WebUI server restarts—subprocesses still running during a server restart detach from memory handles and rely on `/proc` and heartbeat files for takeover.

One error contract spans the backend, the CLI and the browser. [problems.py](<../moss_lora/problems.py>) holds `CODES`, one Chinese template per `err.<code>`, and the `Problem(code, params)` record that `Config.validate()` and the settings checker return; [errors.py](<../moss_lora/web/errors.py>) wraps it as `raise fail(status, code, **params)`, whose `detail` is `{code, params, message}`. The message is the rendered Chinese so logs, the CLI and any client that ignores the code still read a sentence, while `detail.code` names the key each language pack translates and the parameters travel unrendered. `POST /api/validate` returns `problems` and `problem_codes` in parallel, one entry per problem, and `POST /api/settings` returns `{field: detail}` so each message lands next to its own input. The frontend's `describeError()` and `describeDetail()` translate a code when the pack carries it and fall back to the server's `message` when it does not. [test_error_keys.py](<../tests/test_error_keys.py>) parses every `err.*` literal in the package and fails when a code has no manifest entry, no translation, a `{placeholder}` set that disagrees with Chinese, or a template that drifted from the zh pack.

## 9. WebUI Frontend

The frontend uses a buildless, zero-npm-dependency, zero-CDN native browser ES module architecture, managing nine functional tabs through runtime HTML fragment injection and a mutually exclusive dual-mode state machine.

### 9.1 Shell, Fragment Injection, and Boot Sequence

The entry file [index.html](<../moss_lora/web/static/index.html>) contains only the application shell: an inline SVG icon sprite, the top bar (brand mark, GPU monitor chip, global job status chip, language switcher, and 9 tab navigation links), the global `#activity` progress bar, the `<main id="main">` mount point, and 8 native `<dialog>` modals. Stylesheets are split into design tokens (`css/tokens.css`), base components, and page-specific CSS files using a system font stack and a dark theme.

The boot sequence (`boot` in [main.js](<../moss_lora/web/static/js/main.js>)) runs in strict order:
1. Calls `mountFragments(TABS)` in [fragments.js](<../moss_lora/web/static/js/core/fragments.js>) to concurrently `fetch` all nine HTML fragments—[dashboard.html](<../moss_lora/web/static/pages/dashboard.html>), [train.html](<../moss_lora/web/static/pages/train.html>), [dataset.html](<../moss_lora/web/static/pages/dataset.html>), [cache.html](<../moss_lora/web/static/pages/cache.html>), [samples.html](<../moss_lora/web/static/pages/samples.html>), [export.html](<../moss_lora/web/static/pages/export.html>), [workspace.html](<../moss_lora/web/static/pages/workspace.html>), [settings.html](<../moss_lora/web/static/pages/settings.html>), and [caption.html](<../moss_lora/web/static/pages/caption.html>)—parse each via `<template>`, verify its root `#tab-<name>` ID, and insert them in order before the first `<dialog>` inside `#main`.
2. Reorders dashboard panel DOM nodes, calls `bindEvents()` in [events.js](<../moss_lora/web/static/js/events.js>) to attach all event listeners once, and initializes Canvas charts.
3. Awaits `loadI18n()` to load the active dictionary and render the top-bar language selector before other data calls, preventing untranslated raw keys from flashing in the header.
4. Concurrently fetches `/api/settings`, `/api/workspaces`, `/api/system`, `/api/jobs`, and `/api/presets`, loads the active workspace configuration, runs `applyI18n(document)`, and routes to the tab indicated by `location.hash`; when the backend reports `first_run: true`, it opens the five-step onboarding dialog.
5. Attaches core state and action helpers onto `window.__mossUI` for headless browser regression tests.

Every file under `static/` is served by `UncachedStaticFiles` in [server.py](<../moss_lora/web/server.py>) with `Cache-Control: no-store` and without `ETag` or `Last-Modified`. The URLs carry no content hash, so a conditional request would answer `304` and keep the previous module graph alive beside the HTML just written; a module that gained an export in that window then fails to link and the page renders empty. A reload must fetch what is on disk.

### 9.2 Dual-Mode Form State Machine and i18n

- Mutually exclusive Form/YAML state machine ([form.js](<../moss_lora/web/static/js/config/form.js>), [model.js](<../moss_lora/web/static/js/config/model.js>), and [yaml.js](<../moss_lora/web/static/js/config/yaml.js>)): `state.mode` in global state ([state.js](<../moss_lora/web/static/js/core/state.js>)) takes only `form` or `yaml`. In Form mode, the 36 `FIELD_DEFS` controls and the layer-range selector read and mutate `state.config` in memory while the right-hand YAML preview pane stays a read-only projection serialized by the built-in indentation-subset serializer `dumpYaml`; switching to Advanced YAML mode disables all left-hand form controls and makes the YAML textarea the sole source of truth (parsed in real time by `parseYaml` with line and column syntax error reporting), eliminating bidirectional sync race conditions.
- Multi-language dictionaries and reload-free switching ([core/i18n.js](<../moss_lora/web/static/js/core/i18n.js>) and backend [i18n.py](<../moss_lora/i18n.py>)): Ships with four built-in dictionaries—Simplified Chinese (`zh`), English (`en`), Japanese (`ja`), and Korean (`ko`) under `static/i18n/<code>.json`—and supports exporting a template from the Settings tab and uploading custom or override dictionaries to `i18n/<code>.json` at the project root. Backend `i18n.merged` merges dictionaries in three layers ("user dictionary $\rightarrow$ built-in target dictionary $\rightarrow$ built-in Chinese base dictionary"), while frontend `t(key, vars)` falls back to the raw `key` string and interpolates `{name}` placeholders if still unmatched. `applyI18n` scans `data-i18n`, `data-i18n-title`, `data-i18n-placeholder`, and `data-i18n-aria` attributes across the DOM to redraw all static labels, chart legends, and status tooltips on language change without reloading the page.

### 9.3 Three Frontend Hard Conventions

1. No top-level cross-module side effects: Top-level scope in `static/js/**/*.js` may declare only constants, pure data structures, and functions; DOM queries (`$()`), network requests, and cross-module initialization calls at import time are forbidden. Every side effect must run inside `boot()` in `main.js` after `mountFragments` completes, preventing circular dependencies and null DOM references.
2. Page markup lives in `static/pages/`: HTML structure for all nine tabs must reside in `moss_lora/web/static/pages/<tab>.html`; inlining tab markup back into `index.html` or concatenating large static page skeletons inside JS is forbidden.
3. 700-line file limit: No frontend JS module or backend Python module may exceed 700 lines (the largest file is currently [events.js](<../moss_lora/web/static/js/events.js>) at 693 lines); new features must be split by responsibility into `core/`, `config/`, `components/`, or `pages/` submodules.

Using native buildless ES modules lets developers edit any JS, HTML, or CSS file and refresh the browser immediately without a Node.js bundler toolchain or stale build artifacts, at the cost of forgoing TypeScript static type checking and build-time tree shaking, and relying on a handwritten YAML subset parser ([yaml.js](<../moss_lora/web/static/js/config/yaml.js>)) that does not support multiline folded scalars or anchors (and drops YAML comments when switching back to Form mode).

## 10. Configuration System

The configuration system separates machine environment settings (`settings.yaml`) from experiment hyperparameters, catching invalid configurations before launch through VRAM presets and preflight validation endpoints. A training config lives in exactly one place: `runs/<run>/config.yaml`, a complete config rather than an overlay. A new run starts from `Config()` defaults or a copy of another run, and from then on depends on nothing else. Writes are refused while the run trains and allowed afterwards.

Responsibilities are divided across two configuration layers:
- Machine-level configuration `settings.yaml` (managed by [settings.py](<../moss_lora/settings.py>)): Located at the repository root, ignored by `.gitignore`, and copied automatically from [settings.example.yaml](<../settings.example.yaml>) by `ensure_settings_file` on first launch. It contains four top-level sections:
  - `paths.workspaces`: Root directory for workspaces (default `workspaces`, resolved relative to the repository root).
  - `runtime`: CUDA caching allocator policy `allocator` (`expandable|async|off`, default `expandable`) and garbage collection threshold `gc_threshold` (default `0.9`). `alloc_conf_for` injects `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.90` (or `backend:cudaMallocAsync`) for training jobs, but clears the variable for `encode-text` and `encode-prompt` to avoid GPU mapping failures on certain drivers or WSL2 when streaming 9B weights at once.
  - `caption`: OpenAI-compatible local vision service settings including `base_url`, `api_key`, `model`, `trigger`, `workers` (`1..32`, default `6`), `retries` (`1..10`, default `3`), `temperature` (default `1.0`), `top_p` (default `0.95`), `max_tokens` (default `16384`), `thinking` (default `true`), and custom templates `prompt_tags` / `prompt_natural` (saved as empty strings when identical to built-in templates). `api_key` follows strict persistence rules (see [caption.py](<../moss_lora/web/services/caption.py>) and [settings_svc.py](<../moss_lora/web/services/settings_svc.py>)): if the input matches an environment variable name `^[A-Z_][A-Z0-9_]{0,63}$` that is set in the environment, the variable name is saved to `settings.yaml`; if a literal secret key is entered and passes `/models` probing, it is stored only in process memory `deps._LITERAL_API_KEY` while `settings.yaml` records an empty string; captioning subprocesses always receive the key via subprocess environment variables, never via CLI arguments or log files.
  - `ui.language`: Active UI language code (matching `^[a-z][a-z0-9-]{1,7}$`, default `zh`).
- Each run's own `workspaces/<WORKSPACE>/runs/<run>/config.yaml` (managed by [config.py](<../moss_lora/config.py>)): Mapped to the nested `Config` dataclass containing `name`, `output_dir`, and eight subsections: `model` (`ModelConfig`), `dataset` (`DatasetConfig`), `cache` (`CacheConfig`), `lora` (`LoraConfig`), `train` (`TrainConfig`, embedding `sigma: SigmaConfig`), `vram` (`VramConfig`), `export` (`ExportConfig`), and `preview` (`PreviewConfig`). When loaded from `runs/<run>/config.yaml`, `Config.load` sets `_base_dir` to the workspace directory so `dataset.path`, `dataset.reg_path`, `cache.dir` (defaulting to `train_cache` when empty), `cache.reg_dir`, and `output_dir` (defaulting to `runs` when empty) all resolve relative to the workspace; when saving from the WebUI, `_portable_workspace_config` rewrites absolute cache and output paths pointing inside the workspace back to relative names so the entire workspace folder can be moved or backed up cleanly. `${VAR}` environment expressions expand only when the variable is non-empty and remain literal when unset so validation catches missing variables instead of silently resolving to root paths `/...`.

Validation and preset planning span the frontend, backend, and CLI:
- Static and runtime validation: `Config.validate()` verifies required paths (`dataset.path`, `model.dit`, and `model.vae` must exist; `model.text_encoder` may be empty for cache-only training, or must point to a directory with `config.json` or a `.safetensors` file), enum values (`dataset.moss_caption_source`, `train.attention`, `train.batch_mode`), and numeric bounds (`lora.rank > 0`, `lora.layers` and `preview.uncond_skip_layers` within `0..41`, and `preview.width` / `preview.height` positive multiples of 16). Web endpoint `POST /api/validate` ([validate.py](<../moss_lora/web/routers/validate.py>)) layers cache readiness checks on top and returns `next_tab` and `focus_field` so the UI jumps directly to the first invalid control.
- VRAM presets and CLI planning: `apply_preset(config, tier)` writes the tier's bundle -- `gradient_checkpointing`, `blocks_to_swap`, `optimizer_in_8bit`, `quantize_dit`, `quantize_dtype`, `quantize_conditioning`, `compile`, and the preview VAE dtype -- into the config, and only when a caller passes a tier (the WebUI selector, `--preset`, the `plan` helper). The trainer never resolves a tier, so `detect_vram_gb()` is not on the training path. The CLI subcommand `python -m moss_lora.cli plan --vram <GB> --config workspaces/<WORKSPACE>/runs/<RUN>/config.yaml --write workspaces/<WORKSPACE>/runs/<RUN>/config.yaml` subtracts sampling headroom and selects the fastest precision, NF4 quantization, and block-swap combination from the built-in `MEMORY_TABLE` benchmark table that fits within the VRAM budget, writing the result back to the config file.

Separating the two configuration tiers and resolving paths relative to the workspace keeps workspace configs free of host-specific data while excluding `settings.yaml` from version control protects local directory layouts and API keys, at the cost that every run in a workspace shares a single `train_cache/`—a run whose `dataset.max_area` differs invalidates the cached latents (the builder re-encodes them), while one that changes the caption source requires re-running the text-encoding pass.

## 11. Testing and Verification

The repository includes GPU-free unit tests, an optional encoder numerical consistency check, a zero-dependency CDP frontend regression harness, and mandatory documentation sanitization scanning.

Automated test scripts reside under `tests/`:
1. Sidecar read/write and migration unit tests ([test_sidecar.py](<../tests/test_sidecar.py>)): Verifies `schema: 2` single-style writes, lossless dual-style merges, single-style overwrites, `reference_tags` fallback priority between `.txt` files and saved `kept_tags`, and automatic migration and validation from legacy flat JSON files to `tag_caption_style` / `official_style` in temporary directories.
2. Multi-language dictionary completeness unit tests ([test_i18n.py](<../tests/test_i18n.py>)): Uses the built-in Chinese dictionary `zh.json` as the baseline and asserts that `zh`, `en`, `ja`, and `ko` share identical key sets, contain no empty translations, and preserve the exact same `{placeholder}` token sets in every string.
3. Vision service API key rule unit tests ([test_caption_key.py](<../tests/test_caption_key.py>)): Exercises all five branches of `_plan_api_key` and `resolve_api_key` offline: persisting environment variable names to disk and resolving their runtime values, keeping plaintext keys solely in process memory `deps._LITERAL_API_KEY` without writing them to disk, clearing keys on empty input, reading non-identifier strings from legacy files via `legacy` fallback, and returning `missing` when a stored variable name is unset in the environment so the variable name itself is never sent as a bearer token.
4. Text encoder right-padding isolation test ([test_text_padding.py](<../tests/test_text_padding.py>)): Run with `MOSS_TE=<MODELS_DIR>/text_encoders/Qwen3.5-9B`, it compares embeddings of a short prompt encoded alone (`batch_size=1`) against the same prompt batched with a much longer prompt under right-side padding (`batch_size=2`), asserting cosine similarity above `0.999` to verify that causal masking prevents right-padding leakage.
5. Static asset cache header contract ([test_static_cache.py](<../tests/test_static_cache.py>)): Builds the application over an ASGI transport and asserts that `/`, ES modules, and page fragments answer `200` with `Cache-Control: no-store` even when the request carries `If-None-Match` or a future `If-Modified-Since`, that no validator headers come back, that API routes still match ahead of the `/` static mount, and that a missing module stays `404`.

6. Quantization and memory accounting unit tests ([test_quant_linear.py](<../tests/test_quant_linear.py>), [test_quant_fp8.py](<../tests/test_quant_fp8.py>), [test_quant_conditioning.py](<../tests/test_quant_conditioning.py>)): The fp8 check asserts that a replaced Linear reproduces its bf16 output within the storage error and still passes a gradient to its input; the conditioning check builds a toy model with the same 16-Linear layout and asserts that only those layers change format while blocks, `video_embedder`, and anything carrying an adapter stay `bfloat16`.
7. VRAM estimate calibration ([test_vram_estimate.py](<../tests/test_vram_estimate.py>)): Replays five recorded runs through `estimate_vram` and requires the predicted peak to land within 10% of the measured one, which pins the per-block, conditioning, activation, and optimizer coefficients against regression. Every row errs high, and the streamed row errs highest: it charges the lookahead and pool blocks whether or not they are in use, and an estimate that overshoots only makes a user reserve more headroom, while one that undershoots lets a run die at step one.
8. Cache overwrite contract ([test_cache_overwrite.py](<../tests/test_cache_overwrite.py>)): Asserts that an existing sample file is skipped with `cache.overwrite` off, re-encoded with it on, and that the chained text stage stays a no-op unless text encoding was selected too, all against a temporary cache directory without a GPU.
9. Preflight handler regression ([test_validate_router.py](<../tests/test_validate_router.py>)): Drives `POST /api/validate` over an ASGI transport with a valid YAML and a broken one, guarding the local-variable shadowing bug that turned every valid preflight into an `UnboundLocalError`.
10. Error-key completeness ([test_error_keys.py](<../tests/test_error_keys.py>)): Parses every `problem()` call site, compares the codes against the error manifest, and requires each of them to carry a Chinese key in `zh.json` whose `{placeholder}` set matches the code's parameters, so a new failure mode cannot ship with an untranslated message.
11. Streaming correctness ([test_offload_stream.py](<../tests/test_offload_stream.py>)): Runs a four-block toy transformer through `BlockSwap` in training and in eval mode and asserts that streamed and resident weights produce the same output, that the host copy is released after use, that the prefetch path never hands back a buffer the allocator has already reclaimed, and that `set_count` moves residency in both directions without changing a sampled value.
12. Estimate endpoint isolation ([test_estimate_router.py](<../tests/test_estimate_router.py>)): Calls `POST /api/estimate` with a valid YAML while `detect_vram_gb` is patched to raise, proving the handler reads the config's own block count and never asks the driver, which would create a CUDA context in the web process; it also checks the returned train, preview, and total figures add up and that a retired `preset: auto` config still loads.

Frontend regression testing and router-split equivalence are enforced by [smoke.mjs](<../tests/ui/smoke.mjs>) and route flattening:
- Backend split equivalence: [server.py](<../moss_lora/web/server.py>) flattens all 11 sub-routers in place into `app.router.routes` via `_flatten_included_routes`, preserving the exact route matching order of the original monolithic `app.py` while keeping [app.py](<../moss_lora/web/app.py>) as a backwards-compatible re-export shim.
- Frontend CDP smoke regression: [smoke.mjs](<../tests/ui/smoke.mjs>) connects directly to the Chromium DevTools Protocol (CDP) over native Node.js ESM and WebSocket without Playwright or Puppeteer. At a fixed `1440×1000` viewport, the script cycles through `zh`, `en`, `ja`, and `ko` via `POST /api/i18n/current`, visits all 9 tabs, waits until the DOM structural signature stays stable for 700ms, injects `browserMetrics`, and asserts four invariants:
  1. Runtime errors: `Runtime.exceptionThrown`, console `error/warning/assert`, and non-favicon `Network.loadingFailed` counts must all equal zero.
  2. Horizontal overflow: `scrollWidth - clientWidth` on both `document.documentElement` and the active `.tabpane` must equal `0`.
  3. Unintended text clipping: Walking up from every text node to any overflowing `overflow: hidden|clip` ancestor and excluding elements with explicit `text-overflow: ellipsis`, `-webkit-line-clamp`, or a `title` tooltip, `clippedText` must equal zero.
  4. Empty translations and raw keys: Every visible `[data-i18n*]` target must be non-empty, and non-code elements (excluding `pre`, `code`, and `.mono`) must not display dotted raw keys such as `nav.dashboard`.

Public documentation sanitization is enforced by [test_docs_sanitized.py](<../tests/test_docs_sanitized.py>). The script scans `README*.md`, `AGENTS.md`, `settings.example.yaml`, and all text files under `docs/` and `skills/` against two regex rule sets: `HARD` rules block local absolute and mount paths and credential shapes (exiting non-zero on any match); `SOFT` rules flag measured VRAM numbers, step latency or throughput figures, temperatures, and GPU model names (emitting warnings). Host-specific terms are supplied by an untracked local list that the script loads when present. Run this script after every documentation change and confirm `0 hard hit(s), 0 warning(s)`.

## 12. Known Limitations and Future Directions

The current architecture has explicit boundaries around multi-sample batching, low-VRAM numerical precision, and adapter coverage, with future work focused on variable-length timestep modulation and additional conditioning modalities.

Four primary limitations apply to the current version:
1. Token-budget granularity: a micro-batch is filled until the budget is spent, so its image count varies with the aspect ratios and text lengths of the samples drawn. The budget bounds compute rather than image count, which makes step cost predictable but `batch_size` meaningless as a knob; `train.grad_accum` and `train.bin_tokens` are the levers.
2. Shared timestep per micro-batch: fusing several samples into one forward pass forces them to share a single scalar timestep $\sigma$ (and thus one `shared_block_modulation`). This raises per-step gradient variance slightly compared with one $\sigma$ per sample, and relies on step count or gradient accumulation to smooth out.
3. Precision and visual quality trade-offs in low-VRAM tiers: Enabling `vram.fp16_accumulate: true` under `model.dtype: float16` lowers matrix-multiply accumulation precision and is more prone to gradient overflow at high learning rates or long runs than `bfloat16`; NF4 base-model quantization (`vram.quantize_dit: ffn|all`) introduces static quantization noise that slightly degrades fine brushstroke fidelity and dense typography compared to a full `bfloat16` base.
4. Adapter target scope: The trainer does not fine-tune the Qwen3.5-9B text encoder or the FLUX.2 VAE and restricts low-rank adapters to the DiT and optional `text_embedder`; the offline converter [lora_file.py](<../moss_lora/lora_file.py>) covers only those two key families and requires vocabulary extensions for custom modules.

Future iterations can expand along two directions:

- Per-sample timestep modulation in packed mode: Update the modulation broadcast logic in `_Block.attn` and `_Block.ffn` to expand each sample's `[1, 6, D]` modulation vector across token segments via `torch.repeat_interleave` over `cu_seqlens`, removing the shared-timestep constraint in `packed` mode.
- Reference-image and layout-controlled fine-tuning: Wire up the first axis of MOSS-Image's 4D RoPE (reference index `r`) and bounding-box layout JSON supervision to support image-editing (`Edit`) variants and precise text-layout LoRA training.
