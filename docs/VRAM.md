# VRAM Guide

**English** · [简体中文](VRAM.zh-CN.md)

Training loads the 5B DiT and nothing else. The text encoder and the VAE ran during caching, so the card holds the DiT, the activations of one step, the LoRA optimizer state, and the CUDA context. A preview adds sampling activations and the VAE decoder on top, and that is where a run that trained comfortably can still run out of memory.

This guide gives the measured cost of each choice and a setting for each card class. All figures come from one machine; the last section explains how far they travel.

## 1. What decides VRAM here

Four terms add up, and each has its own control.

| Term | What sets it |
| --- | --- |
| Frozen DiT weights | `vram.quantize_dit`, `vram.quantize_dtype`, `vram.blocks_to_swap` |
| Activations | the joint sequence length, fixed by `dataset.max_area` |
| Optimizer state | the number of trainable parameters and `vram.optimizer_in_8bit` |
| CUDA context | the driver, about 1.5 GB |

The DiT weights are 9.5 GB in bf16, 2.5 GB at nf4, and 4.7 GB at fp8. Those figures cover the 42 blocks. The conditioning half (text projection, both modulations, time embedding, final projection) adds 0.6 GB and stays bf16 unless you ask for otherwise.

Activations follow the sequence the model sees. MOSS-Image is a single-stream DiT: image tokens and text tokens meet in one self-attention, so the term grows with their sum, not with the number of images alone. A 1024x1024 image produces about 4096 tokens (pixels divided by 256); a caption contributes tens of tokens. `dataset.max_area` caps the cached latent size, which fixes the image-token count before training starts. Raising it raises the activation term on every step; quantization and streaming leave that term alone.

The optimizer keeps state for the trainable parameters only, which are the LoRA adapters. AdamW holds two fp32 moments per parameter, eight bytes per parameter in total; the 8-bit optimizer keeps each moment in eight bits. At the default rank of 32, the saving is about 0.2 GB, at a cost of two to three percent of wall time. The optimizer is the smallest of the four terms.

The CUDA context is fixed. The driver charges about 1.5 GB before PyTorch allocates a single tensor, and every run pays it.

## 2. Quantization

`vram.quantize_dit` chooses how much of the frozen DiT to replace: `none` keeps bf16, `ffn` converts the two FFN projections in every block, and `all` converts QKVO and FFN. `vram.quantize_dtype` chooses the storage: `nf4` packs four bits per weight plus per-group scales, and `fp8` stores one byte plus one scale per output row. The adapter stays in the compute dtype; only frozen weights are rounded.

| `quantize_dit` + `quantize_dtype` | All 42 blocks resident | All 42 blocks streamed |
| --- | --- | --- |
| none (bf16) | 14.8 GB | 6.3 GB |
| ffn + nf4 | 10.2 GB | 6.2 GB |
| ffn + fp8 | 12.1 GB | 7.0 GB |
| all + nf4 | 8.2 GB | 6.1 GB |
| all + fp8 | 10.9 GB | 6.4 GB |

Measured at 1024x1024, batch 1, no previews, bf16 weights, `compile: false`, allocator `expandable`. Figures are driver peaks.

Compared with bf16, the resident column drops by about 5 GB with `ffn` + nf4 and about 7 GB with `all` + nf4. On the 32 GB reference card none of these settings made a step faster. The differences stayed within noise; quantization bought memory, not speed.

`nf4` is the smaller format and carries more quality risk. `fp8` costs more memory at the same scope and stays closer to bf16. When memory allows, prefer fp8. Between the 4-bit scopes, `ffn` rounds fewer matrices and is the safer choice; `all` trades more quality for another 2 GB. Leave `vram.quantize_conditioning` at `none`: it is off by default and unverified, and its rounding error feeds every block.

## 3. Block streaming

`vram.blocks_to_swap: N` keeps the leading 42 - N blocks on the card and leaves the trailing N in host memory. The WebUI asks for the complement, the resident count, because that is the number a user reasons about; the stored key still counts the offloaded blocks. `42` streams the whole stack; `0` keeps it all resident. Streaming needs gradient checkpointing, which every preset enables.

Streaming moves weights; it does not reduce the work. The offloaded blocks end up in pinned host memory, and the pinned copy equals the offloaded bytes in steady state.

| Offloaded weights | Pinned host memory |
| --- | --- |
| bf16 | 8.9 GB |
| ffn + nf4 | 4.4 GB |
| all + fp8 | 4.4 GB |
| all + nf4 | 2.2 GB |

Host memory peaks higher than the steady pinned copy: a quantized run builds the model on the host before it uploads. On a machine with 16 GB of system memory, host memory is the binding constraint on a streamed run, not the card: bf16 pins 8.9 GB, and a quantized run needs 10-11 GB just to load.

The copies cost time in proportion to the bytes moved. bf16 fully streamed was about 20% slower per step than bf16 resident. `all` + nf4 fully streamed was within 2% of resident, because it moves a quarter of the bytes. The lookahead overlaps each copy with the block's compute, so the penalty depends on bytes and PCIe width, not on the number of blocks.

## 4. The allocator decides whether streaming saves card memory

`runtime.allocator` lives in `settings.yaml`, not in the run config. `expandable` sets `expandable_segments:True` and lets PyTorch return the whole-block segments that streaming frees. `async` selects `backend:cudaMallocAsync`, which pools those segments per stream and hands nothing back; a streamed run then allocates less and occupies the same card. `off` leaves PyTorch's default allocator in place.

| `runtime.allocator` | Peak with all 42 blocks streamed, `all` + nf4 |
| --- | --- |
| `expandable` | 6.1 GB |
| `async` (`cudaMallocAsync`) | 8.1 GB |

Step time was the same in both runs. Under `async` the streamed figure matched the resident figure (about 8 GB), so streaming bought nothing on the card.

Use `runtime.allocator: expandable`. It is the default. The trainer prints a warning when a streamed run starts under `cudaMallocAsync`, because the feature cannot do its job there.

## 5. Preview images

A preview is the peak of the run, not the training step. Sampling adds its own activations, and the VAE decoder runs at the end of the pass; a run that trains at about 6 GB can peak at about 9 GB when it draws a picture.

| Preview size | `preview.vae_dtype` | Driver peak | Time per image |
| --- | --- | --- | --- |
| 1024x1024 | fp32 | 9.3 GB | 1.0 |
| 1024x1024 | fp16 | 7.8 GB | 0.65 |
| 768x768 | fp32 | 6.9 GB | 0.64 |
| 768x768 | fp16 | 6.0 GB | 0.65 |

Measured with `all` + nf4 fully streamed, an 8-step distilled (DMD) LoRA, one image per pass (batch 1). Time is relative to the 1024x1024 fp32 case.

Two settings move the peak. `preview.vae_dtype: fp16` halves the decode's memory and produces a visually identical image: the largest per-pixel difference measured was 1/255, the rounding of one 8-bit channel step. Smaller `preview.width` and `preview.height` cut both the sampler and the decode. At 1024, fp16 also cut the time per image by about a third; at 768 it saved memory but not time.

`preview.batch: 1` is the default in this guide: one image per pass. Batching several prompts multiplies the sampler peak.

Step count dominates preview time. The same preview at 50 steps without the LoRA took about eight times as long as eight steps with it.

## 6. Choosing a tier

`vram.preset` fills the settings for a card class: quantization, streaming, gradient checkpointing, `torch.compile`, and the optimizer; `auto` picks the tier from the detected card. The training page shows each resolved value and whether it came from the preset or from you. Every tier below keeps `preview.batch: 1`.

### 24 GB and up

`vram.preset: 24` (or `32`). bf16, every block resident, `compile: true`. A bf16 run peaks around 15 GB. Draw previews at 1024x1024 with fp32; fp16 leaves more room.

### 16 GB

`vram.preset: 16`. `ffn` + nf4, every block resident, `compile: true`; a run peaks around 10 GB. Draw previews at 1024x1024 with `preview.vae_dtype: fp16`, or at 768x768 with fp32.

### 12 GB

`vram.preset: 12`. The same `ffn` + nf4 resident settings; a run peaks around 10 GB. Keep previews at 768x768 with fp16 to leave headroom. If you need 1024x1024, raise `vram.blocks_to_swap` until the preview fits; streaming applies during previews too, and the copies cost time.

### 8 GB

`vram.preset: 8`. `all` + nf4, all 42 blocks streamed, the 8-bit optimizer, `compile: false`; a run peaks around 6 GB. Preview at 768x768 with fp16, which measured about 6 GB. 1024x1024 at fp16 measured about 8 GB, which leaves too little room on an 8 GB card. Streaming pins about 2 GB of host memory.

### 6 GB

`vram.preset: 6`. The same settings as 8 GB, with a peak around 6 GB against a 6 GB card. Lower `dataset.max_area` to 589824 (768x768), which nearly halves the activation term. Rebuild the latent cache and close other GPU users first. Preview at 768x768 with fp16, or 640x640 if that still runs out.

## 7. What these numbers are worth

Training figures come from 101-step runs on one 32 GB card with a 1007 GB disk, a bf16 base model, 1024x1024 images, and batch 1. Preview figures come from a single pass with the 8-step LoRA described above. The values are rounded: memory peaks keep one decimal in the tables, and the prose rounds them to whole numbers.

A slower card takes longer per step. Narrower PCIe makes the streaming copies cost more, so the 20% penalty for bf16 streaming grows. A machine with 16 GB of system memory is limited by pinned host memory before it is limited by the card.

The absolute figures belong to that one machine; the ordering of the choices transfers. bf16 costs the most and streams the worst, nf4 costs the least, `all` saves more than `ffn`, fp8 sits between, and `expandable` returns memory that `cudaMallocAsync` keeps. Choose a direction from the tables, then measure your own run; `metrics.jsonl` records the peak each run reaches.
