# VRAM Guide

**English** · [简体中文](VRAM.zh-CN.md)

Training loads the 5B DiT and nothing else. The text encoder and the VAE ran during caching, so the card holds the DiT, the activations of one step, the LoRA optimizer state, and the CUDA context. A preview adds sampling activations and the VAE decoder on top, and that is where a run that trained comfortably can still run out of memory.

This guide gives the measured cost of each choice and a setting for each card class. All figures come from one machine; the last section explains what they are worth.

## 1. What decides VRAM here

Four terms add up, and each has its own control.

| Term | What sets it |
| --- | --- |
| Frozen DiT weights | `vram.quantize_dit`, `vram.quantize_dtype`, `vram.blocks_to_swap` |
| Activations | the joint sequence length, fixed by `dataset.max_area`, and `vram.offload_activations` |
| Optimizer state | the number of trainable parameters and `vram.optimizer_in_8bit` |
| CUDA context | the driver, about 1.5 GB of CUDA-visible memory, part of it in shared system memory |

The DiT weights are 9.5 GB in bf16, 2.5 GB at nf4, and 4.7 GB at fp8. Those figures cover the 42 blocks. The conditioning half (text projection, both modulation layers, time embedding, final projection) adds 0.6 GB and stays bf16 unless you ask for otherwise.

Activations follow the sequence the model sees. MOSS-Image is a single-stream DiT: image tokens and text tokens meet in one self-attention, so the term grows with their sum, not with the number of images alone. A 1024x1024 image produces about 4096 tokens (pixels divided by 256); a caption contributes tens of tokens. `dataset.max_area` caps the cached latent size, which fixes the image-token count before training starts. Raising it raises the activation term on every step; quantization and streaming leave that term alone.

The activation term itself has two parts. Gradient checkpointing throws away the tensors inside each block and keeps only the block's input, so it can recompute the block in backward: that is 42 boundaries of `tokens x 3072 x 2` bytes, about 1 GB at 1024x1024. The other part is what one block holds while it computes — its projections and attention buffers — and recomputation cannot remove it. `vram.offload_activations: true` moves the first part to pinned host memory and leaves the second on the card; section 4 covers it.

The optimizer keeps state for the trainable parameters only, which are the LoRA adapters. AdamW holds two fp32 moments per parameter, eight bytes per parameter in total; the 8-bit optimizer keeps each moment in eight bits. At the default rank of 32, the saving is about 0.2 GB, at a cost of two to three percent of wall time. The optimizer is the smallest of the four terms.

The CUDA context is fixed. The CUDA runtime charges about 1.5 GB before PyTorch allocates a single tensor, and every run pays it. On Windows the display driver backs part of that context with shared system memory: `nvidia-smi` and Task Manager count roughly 0.5 GB of it as dedicated VRAM, and the rest never takes card memory.

## 2. Quantization

`vram.quantize_dit` chooses how much of the frozen DiT to replace: `none` keeps bf16, `ffn` converts the two FFN projections in every block, and `all` converts QKVO and FFN. `vram.quantize_dtype` chooses the storage: `nf4` packs four bits per weight plus per-group scales, and `fp8` stores one byte plus one scale per output row. The adapter stays in the compute dtype; only frozen weights are rounded.

| `quantize_dit` + `quantize_dtype` | All 42 blocks resident | All 42 blocks streamed | Streamed + activation offload |
| --- | --- | --- | --- |
| none (bf16) | 13.2 / 14.3 GB | 4.9 / 6.1 GB | 4.0 / 5.1 GB |
| ffn + nf4 | 9.0 / 10.1 GB | 4.7 / 5.8 GB | 3.7 / 4.9 GB |
| ffn + fp8 | 10.6 / 11.7 GB | 4.9 / 6.0 GB | 3.9 / 5.1 GB |
| all + nf4 | 6.6 / 7.8 GB | 4.6 / 5.7 GB | 3.6 / 4.7 GB |
| all + fp8 | 9.0 / 10.1 GB | 4.9 / 6.0 GB | 3.9 / 5.0 GB |

Measured at 1024x1024, batch 1, no previews, bf16 weights, `compile: false`, allocator `expandable`. Each cell is dedicated VRAM first — what `nvidia-smi` and Task Manager show, sampled every 100 ms during the run — then the CUDA pool counter (`cudaMemGetInfo`, the `device` field in `metrics.jsonl`). On Windows and WSL the CUDA counter reads about 1.1 GB higher, because the driver backs part of the CUDA context with shared system memory; on Linux the two counters agree. Resident runs drop by about 1 GB with the offload too: `all` + nf4 goes from 6.6 to 5.7 GB dedicated.

Compared with bf16, the resident peak drops by about 4 GB with `ffn` + nf4 and about 6.5 GB with `all` + nf4. On the 32 GB reference card none of these settings made a step faster. The differences stayed within noise; quantization bought memory, not speed.

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

Host memory peaks higher than the steady pinned copy only while loading, and the loader keeps that peak small. The model is built block by block: each block is read, packed to its storage format, and then either kept on the card or moved to host memory. The bf16 weights of the whole model are never resident at once, so the load peak is the packed tail plus one block in flight, not a full bf16 copy.

An unquantized run is the exception, because its blocks are bf16 wherever they live: on a card that cannot hold all 42, the streamed tail is 8.9 GB of host memory, and loading is the only moment the card sees the whole model.

The copies cost time in proportion to the bytes moved. bf16 fully streamed was about 20% slower per step than bf16 resident. `all` + nf4 fully streamed was within 2% of resident, because it moves a quarter of the bytes. The lookahead overlaps each copy with the block's compute, so the penalty depends on bytes and PCIe width, not on the number of blocks.

## 4. Activation offload

`vram.offload_activations: true` moves the checkpoint boundaries of section 1 to pinned host memory. Each block's input leaves the card when the block finishes; the backward pass reads one boundary back per block and frees it again, so a step holds one boundary instead of 42. The setting needs gradient checkpointing, which every preset enables: without checkpointing the whole graph is retained, and the offload would pin tens of gigabytes of host memory instead of a bounded set. The trainer switches `torch.compile` off while it is on, because the hooks that move the tensors cannot be captured in a compiled graph.

The saving is the boundary term, about 1 GB at 1024x1024: `all` + nf4 fully streamed goes from 4.6 to 3.6 GB dedicated, and bf16 fully streamed from 4.9 to 4.0 GB. Combining it with the 8-bit optimizer measured the lowest training peak in this guide:

| Setting (1024x1024, batch 1) | Dedicated / CUDA pool |
| --- | --- |
| `all` + nf4, 42 blocks streamed | 4.6 GB / 5.7 GB |
| + activation offload | 3.6 GB / 4.7 GB |
| + 8-bit optimizer | 4.2 GB / 5.3 GB |
| + activation offload and 8-bit optimizer | 3.2 GB / 4.4 GB |

That 3.2 GB — 4.4 GB on the CUDA counter — is close to the floor for this configuration. About 1 GB is the three streamed blocks, the LoRA weights, and their optimizer state; about 0.9 GB is the block-local activation that recomputation cannot remove; the remaining 1.3 GB is the part of the CUDA context and the cuBLAS and FlashAttention workspaces that stay in dedicated memory, plus allocator slack. Lowering `dataset.max_area` shrinks the middle term — at 768x768 the block-local activation is roughly halved — but the cache has to be rebuilt for the new size first.

The copies cost time. The offload adds about 2 GB of PCIe traffic per step (1 GB out, 1 GB back). A quantized streamed run hides most of it and pays about a tenth more per step; bf16 streaming already moves 11 GB of weights through PCIe per step, so the extra traffic shows up and cost closer to a third. It also pins about 1 GB of host memory while it runs.

## 5. The allocator decides whether streaming saves card memory

`runtime.allocator` lives in `settings.yaml`, not in the run config. `expandable` sets `expandable_segments:True` and lets PyTorch return the whole-block segments that streaming frees. `async` selects `backend:cudaMallocAsync`, which pools those segments per stream and hands nothing back; a streamed run then allocates less and occupies the same card. `off` leaves PyTorch's default allocator in place.

| `runtime.allocator` | Peak with all 42 blocks streamed, `all` + nf4 |
| --- | --- |
| `expandable` | 6.1 GB |
| `async` (`cudaMallocAsync`) | 8.1 GB |

Step time was the same in both runs. Under `async` the streamed figure matched the resident figure (about 8 GB), so streaming bought nothing on the card.

Use `runtime.allocator: expandable`. It is the default. The trainer prints a warning when a streamed run starts under `cudaMallocAsync`, because the feature cannot do its job there.

## 6. Preview images

A preview is the peak of the run, not the training step. Sampling adds its own activations, and the VAE decoder runs at the end of the pass; a run that trains at about 6 GB can peak at about 9 GB when it draws a picture.

| Preview size | `preview.vae_dtype` | Peak (CUDA) | Time per image |
| --- | --- | --- | --- |
| 1024x1024 | fp32 | 9.3 GB | 1.0 |
| 1024x1024 | fp16 | 7.8 GB | 0.65 |
| 768x768 | fp32 | 6.9 GB | 0.64 |
| 768x768 | fp16 | 6.0 GB | 0.65 |

Measured with `all` + nf4 fully streamed, an 8-step distilled (DMD) LoRA, one image per pass (batch 1). Time is relative to the 1024x1024 fp32 case. These rows were recorded with the CUDA pool counter; subtract about 1.1 GB for the dedicated figure Task Manager shows.

Two settings move the peak. `preview.vae_dtype: fp16` halves the decode's memory and produces a visually identical image: the largest per-pixel difference measured was 1/255, the rounding of one 8-bit channel step. Smaller `preview.width` and `preview.height` cut both the sampler and the decode. At 1024, fp16 also cut the time per image by about a third; at 768 it saved memory but not time.

`preview.batch` defaults to 1: one image per pass. Batching several prompts multiplies the sampler peak.

Step count dominates preview time. The same preview at 50 steps without the LoRA took about eight times as long as eight steps with it.

## 7. Choosing a tier

A preset is a starting point, not a decision. Picking one — the selector on the training page, or `--preset` on the CLI — writes that tier's values into the config: activation offload, quantization scope and format, streaming, gradient checkpointing, `torch.compile`, the 8-bit optimizer, and the preview VAE dtype. Nothing reads the tier afterwards; the run reads the fields, and every field stays editable. A new run starts at the defaults with no preset applied, and nothing is derived from the installed card. Every tier below keeps `preview.batch: 1`.

### 24 GB and up

Preset `24` (or `32`). bf16, every block resident, `compile: true`. A bf16 run peaks around 15 GB. Draw previews at 1024x1024 with fp32; fp16 leaves more room.

### 16 GB

Preset `16`. bf16 base weights with `blocks_to_swap: 6`: 36 blocks stay resident and the trailing 6 stream from host memory. `compile: false`, because a streamed run cannot use torch.compile. Gradient checkpointing stays on. Quantizing would only cost quality here, so the weights stay in bf16 and the copies buy the margin instead. Draw previews at 1024x1024 with the fp16 VAE the preset pins; an fp32 decode leaves almost no margin.

### 12 GB

Preset `12`. bf16 base weights with `blocks_to_swap: 22`: 20 blocks stay resident and the trailing 22 stream from host memory. `compile: false`, because a streamed run cannot use torch.compile. Gradient checkpointing stays on, which streaming requires. Keeping the weights in bf16 rather than nf4 costs step time in the copies, not precision; these cards are expected to have enough host memory for the streamed tail. Previews use the fp16 VAE at 1024x1024: an fp32 decode leaves no usable margin. Lower the preview resolution or give the preview its own residency when another process shares the card.

### 8 GB

Preset `8`. `all` + nf4, all 42 blocks streamed, the 8-bit optimizer, `compile: false`; a run peaks around 6 GB. Set the preview to 768x768, where the fp16 VAE the preset already chose measured about 6 GB. 1024x1024 at fp16 measured about 8 GB, which leaves too little room on an 8 GB card, and at fp32 about 9 GB. Streaming pins about 2 GB of host memory.

### 6 GB

Preset `6`. The same settings as 8 GB, with a peak around 6 GB against a 6 GB card. Lower `dataset.max_area` to 589824 (768x768), which nearly halves the activation term. Rebuild the latent cache and close other GPU users first. Preview at 768x768 with fp16, or 640x640 if that still runs out. Turning on `vram.offload_activations` brings the training step to about 4.4 GB, which leaves the preview as the only thing that has to fit.

## 8. What these numbers are worth

Training figures come from three-step probe runs on one 32 GB card: 1024x1024 images, batch 1, no previews, `compile: false`, allocator `expandable`, and the peak is reached by the second step. Dedicated figures are `nvidia-smi` samples taken every 100 ms during the run; the CUDA pool figures come from the trainer's own counter. Preview figures come from one pass per row with the 8-step LoRA described above. The values are rounded: memory peaks keep one decimal in the tables, and the prose rounds them to whole numbers.

A slower card takes longer per step. Narrower PCIe makes the streaming copies cost more, so the 20% penalty for bf16 streaming grows, and so does the activation offload's share. A machine with 16 GB of system memory is limited by pinned host memory before it is limited by the card.

The absolute figures belong to that one machine; the ordering of the choices transfers. bf16 costs the most and streams the worst, nf4 costs the least, `all` saves more than `ffn`, fp8 sits between, the activation offload takes about 1 GB off every row, and `expandable` returns memory that `cudaMallocAsync` keeps. Choose a direction from the tables, then measure your own run; `metrics.jsonl` records the peak each run reaches.
