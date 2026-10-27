# MOSS-Image LoRA Trainer

**[English](README.md)** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

Local LoRA trainer for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) (5B single-stream DiT, frozen Qwen3.5-9B text encoder, and FLUX.2 VAE), with a PyTorch training stack, a FastAPI backend, and a zero-build native ES module frontend.

This repository reuses the official model architecture and weight mapping from `comfyui/moss-image/model.py` and `loaders.py`, and adds the missing fine-tuning components: dataset scanning and sizing, latent and text embedding caching, Flow Matching training, sample previews, caption generation, and a WebUI.

The trainer splits the workflow into offline caching and online training so the 5B DiT and 9B text encoder never sit in VRAM at the same time. Pre-training steps write latents and text embeddings to disk; training loads only the 5B DiT and fits 1024² fine-tuning on a single consumer GPU. Output `.safetensors` checkpoints use `lora_unet_*` keys by default and load directly into ComfyUI without requiring a ComfyUI runtime for training or export.

## Features

- **Two-stage caching**: Encodes image latents with the FLUX.2 VAE and text embeddings with a local Qwen3.5-9B, then writes both to disk. Text batches use right-side causal padding, and the training forward pass slices each sequence by its attention mask so padding tokens never pollute valid text embeddings.
- **Flow Matching training and VRAM presets**: Trains with truncated logit-normal $\sigma$ sampling and per-token normalized velocity MSE loss. Six presets (`32`, `24`, `16`, `12`, `8`, `6` GB) combine gradient checkpointing, weight-only quantization (`quantize_dit: none|ffn|all` in `nf4` or `fp8`, plus the optional conditioning half), block streaming from host memory, `torch.compile`, and `fp16` accumulation across `packed` (FlashAttention-2 varlen) and `padded` (token-dimension padding) batch modes. A preset writes values into the config; the run reads the values. The dashboard shows a predicted VRAM peak beside the measured one before and during a run.
- **Regularization and holdout validation**: Loads regularization images via `dataset.reg_path` to limit concept drift and splits a deterministic validation set via `train.eval_holdout` for periodic loss checks.
- **Warm starts from an existing adapter**: `lora.init_from` points at a `.safetensors` LoRA (kohya, PEFT, or ComfyUI naming) and fills the fresh adapter from it before training begins. The file's rank must equal `lora.rank`; a different alpha is folded into `lora_B` so the loaded model computes what the file computed. The WebUI inspects the file at start, rejects a rank mismatch, and asks before rescaling an alpha.
- **Dual-style captioning**: Queries any OpenAI-compatible vision endpoint to generate `tag_caption` (keeps existing tags and appends a scene sentence) and `official` captions (long, medium, and short descriptions), stored in a companion sidecar JSON.
- **Zero-build WebUI**: Runs native ES modules across nine tabs (Dashboard, Training settings, Dataset, Cache, Samples, Export, Workspace, Settings, and Caption). Source edits take effect on browser refresh. The dashboard has one training action: it resumes the selected run when a checkpoint exists, and a **Retrain** checkbox switches it to a fresh start (with a confirmation) instead.
- **Native ComfyUI format**: Saves checkpoints and exports with `lora_unet_*` keys by default, with optional `diffusers` conversion and step-range batch zipping.
- **Multilingual UI**: Ships with Simplified Chinese, English, Japanese, and Korean dictionaries, hot-swappable without page reloads, plus template export and custom language pack overrides.

## Requirements

- **OS and Python**: Linux, WSL2, or Windows 10/11 with PowerShell 5.1+, and Python 3.11+.
- **GPU and drivers**: NVIDIA GPU with `bfloat16` or `float16` support and CUDA drivers matching PyTorch 2.9+.
- **VRAM settings, with presets as starting points**: every VRAM field is a plain value in the run's config, and the trainer reads those values as written. The preset selector in the WebUI (`32`, `24`, `16`, `12`, `8`, `6`) and `--preset` on the CLI fill in one tier's values once — gradient checkpointing, resident blocks, activation offload, quantization scope and format, `torch.compile`, the 8-bit optimizer, and the preview VAE dtype — and then get out of the way. A new run starts at the defaults with no preset applied, nothing is derived from the card, and the trainer raises `blocks_to_swap` by itself only when a step runs out of memory.
- **Local weights**: MOSS-Image DiT, Qwen3.5-9B text encoder (HuggingFace directory, or a ConvRot int8-quantized single-file `.safetensors`), and FLUX.2 VAE.

To choose settings for a specific card, read the [VRAM guide](<docs/VRAM.md>). It gives the measured cost of quantization and block streaming, explains the allocator setting, and lists a preview size for each card class.

## Installation

The repository isolates dependencies in `.venv` at the project root. Both the CLI and WebUI background workers invoke the venv interpreter (`.venv/bin/python`, or `.venv\Scripts\python.exe` on Windows) directly, without relying on external conda or ComfyUI environments.

[setup.sh](<setup.sh>) searches for `python3.14` down to `python3.11`, creates `.venv`, installs [requirements.txt](<requirements.txt>), and runs [setup_fa2.py](<setup_fa2.py>) to install a prebuilt FlashAttention-2 wheel matching your Python, PyTorch, CUDA, and GPU architecture (or warns and falls back to SDPA if no wheel matches). [run.sh](<run.sh>) checks whether the target port is free, runs `./setup.sh` if `.venv` is missing, and starts the WebUI (`127.0.0.1:7861` by default).

On native Windows, [setup.ps1](<setup.ps1>) does the same job: it finds Python through the `py` launcher, creates `.venv`, installs CUDA PyTorch from the index that matches the installed NVIDIA driver (`cu126` for 527+, `cu128` for 570+, `cu130` for 580+), then installs [requirements.txt](<requirements.txt>). [run.ps1](<run.ps1>) calls it when `.venv` is missing. Prebuilt FlashAttention-2 wheels are published for Linux only, so Windows trains with `padded` batches and SDPA unless `-WithFlashAttention` finds a wheel.

```bash
# Create .venv and install dependencies (override via PYTHON or TORCH_INDEX_URL)
./setup.sh
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./setup.sh --force

# Start or stop the WebUI on a specific port
./run.sh --port 7869
./run.sh --port 7869 --stop
```

```powershell
# Windows: same install, with the CUDA index picked from the installed driver
.\setup.ps1
.\setup.ps1 -Force -TorchIndex cu130
.\setup.ps1 -DryRun

# Start the WebUI (creates .venv on first run)
.\run.ps1 --port 7869
```

If Windows blocks local scripts, run `powershell -ExecutionPolicy Bypass -File .\setup.ps1`.

## Quick Start

In the WebUI, run `./run.sh --port 7869` and open `http://127.0.0.1:7869`. Create a workspace in the **Workspace** tab, create a run in **Training settings**, fill in `<MODELS_DIR>` and `<DATASET_DIR>`, then run latent caching, text embedding caching, and training in order.

From the CLI, run the three stages against one run's config:

```bash
WS=workspaces/<WORKSPACE>
RUN=<run>
CONFIG="$WS/runs/$RUN/config.yaml"

# 1. Encode image latents to disk (writes to $WS/train_cache)
.venv/bin/python -m moss_lora.cli cache --config "$CONFIG"

# 2. Encode text embeddings to disk (local Qwen3.5-9B)
.venv/bin/python -m moss_lora.cli encode-text --config "$CONFIG"

# 3. Run LoRA training (writes to $WS/runs/<name>)
.venv/bin/python -m moss_lora.cli train --config "$CONFIG"
```

The dataset scanner recursively finds `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` images and pairs each image with its sidecar JSON, matching `.txt`/`.caption` file, or `metadata.jsonl` entry. Subdirectories prefixed like `3_concept` repeat their images 3 times per epoch.

Common helper subcommands:

```bash
# Inspect dataset statistics and caption coverage
.venv/bin/python -m moss_lora.cli info --config "$CONFIG"

# Plan dtype, quantization, and block swap for a VRAM budget (GB) and write back
.venv/bin/python -m moss_lora.cli plan --vram 12 --config "$CONFIG" --write "$CONFIG"

# Generate LoRA-off baseline images or render previews for a specific step checkpoint
.venv/bin/python -m moss_lora.cli baseline --config "$CONFIG"
.venv/bin/python -m moss_lora.cli preview --config "$CONFIG" --step 500

# Encode custom prompts and sample with a specific checkpoint
.venv/bin/python -m moss_lora.cli encode-prompt --config "$CONFIG" \
    --out "$WS/custom_prompts.safetensors" --prompt "a portrait in soft daylight"
.venv/bin/python -m moss_lora.cli sample --config "$CONFIG" \
    --checkpoint "$WS/runs/<name>/checkpoints/<ckpt>.safetensors" \
    --embeds "$WS/custom_prompts.safetensors" --out "$WS/samples"

# Sequentially resume unfinished training runs in a workspace
.venv/bin/python -m moss_lora.cli queue --workspace "$WS"

# Start a run that survives an SSH disconnect, then read it the way the dashboard does
.venv/bin/python -m moss_lora.cli train --config "$CONFIG" --detach
.venv/bin/python -m moss_lora.cli status
.venv/bin/python -m moss_lora.cli status --workspace my-lora --live --json
```

`status` reads the workspace files directly: no server, no cached state. The dashboard, `GET /api/jobs`, and this command all project the same heartbeat, so a run started in a terminal shows up in the browser and a run started in the browser shows up in the terminal. [docs/RUNTIME-CONTRACT.md](<docs/RUNTIME-CONTRACT.md>) defines the files and the test that keeps the three in step.

## Configuration

Configuration splits into machine-level `settings.yaml` and per-run configs.

`settings.yaml` holds local machine settings (`.gitignore` excludes it, and the first launch copies it from [settings.example.yaml](<settings.example.yaml>)). It configures the workspaces root `paths.workspaces`, CUDA allocator settings `runtime` (`allocator: expandable|async|off` and `gc_threshold`), UI language `ui.language`, and the caption service `caption` (`base_url`, environment variable name `api_key`, model name, concurrency, and prompt templates). Plaintext API keys entered in the Settings tab stay in process memory and never touch disk.

Each run carries a complete config at `workspaces/<WORKSPACE>/runs/<RUN>/config.yaml`; it defines models, datasets, caches, LoRA rank and alpha, training hyperparameters, VRAM settings, and preview prompts. Relative paths (such as `cache.dir` and `output_dir`) resolve against the workspace directory, so moving a workspace directory keeps its paths valid. The file is frozen while its run trains; editing it afterwards is allowed. Create a run in the WebUI to get one, or copy an existing run with the run dialog:

```yaml
name: my-lora
model:
  dit: <MODELS_DIR>/diffusion_models/moss_image_dit.safetensors
  text_encoder: <MODELS_DIR>/text_encoders/Qwen3.5-9B
  vae: <MODELS_DIR>/vae/flux2_vae.safetensors
  dtype: bfloat16
dataset:
  path: <DATASET_DIR>
  max_area: 1048576
  moss_caption_source: tag_caption
lora:
  rank: 32
  alpha: 32
  targets: [q, k, v, o, ffn_up_gate, ffn_down]
train:
  steps: 1000
  seed: 42
  batch_size: 1
  grad_accum: 1
  lr: 0.0001
  optimizer: adamw
  save_every: 250
  sample_every: 250
vram:
  gradient_checkpointing: true
  offload_activations: false
  blocks_to_swap: 0
  quantize_dit: none
  compile: true
export:
  formats: [kohya]
```

`train.seed` is the only seed a run needs: it drives batch sampling, sigma, the per-step flow-matching noise, caption dropout, and adapter initialisation. Checkpoints carry the RNG state forward, so a resumed run continues the same trajectory instead of replaying it. On one machine, two runs of the same config produce the same losses, because the FlashAttention-2 varlen backward runs in its deterministic mode, which costs about a tenth of the step time. `preview.seed` controls preview sampling and nothing in training.

## Captioning

The captioning module ([captioner.py](<moss_lora/captioner.py>)) calls an OpenAI-compatible vision model from either the WebUI **Caption** tab or the CLI:

```bash
.venv/bin/python -m moss_lora.captioner \
    --images <DATASET_DIR> \
    --styles tag_caption,official \
    --base-url http://127.0.0.1:8080/v1 \
    --api-key VLM_API_KEY
```

You can also pass the endpoint and credentials via `VLM_BASE_URL`, `VLM_API_KEY`, and `VLM_MODEL`. `--api-key` accepts either a literal key or the name of an environment variable holding the key (`cli caption` reads the corresponding fields from `settings.yaml`).

`--styles` accepts one or both styles:

- `tag_caption` (`tag_caption_style`): Reads `<stem>.txt`/`.caption` (falling back to `kept_tags` in an existing sidecar), keeps the original tags, asks the vision model for a natural-language scene description (pose, framing, angle, background, lighting) while stripping identity traits such as hair or eye color, and skips images without usable tags.
- `official` (`official_style`): Ignores existing tag files and follows the official recaptioner specification to output `caption_long`, `caption_medium`, and `caption_short`, along with watermark boxes and border directions.

Both styles share `<stem>_moss-image-trainer.json` (`schema: 2`) and leave the original `.txt` file untouched. Re-running a single style updates only its own block; the runner skips existing results by default, `--overwrite` forces regeneration, `--migrate` upgrades legacy flat sidecars, and `--validate-only` checks existing files:

```json
{
  "schema": 2,
  "role": "train",
  "tag_caption_style": {
    "caption": "<trigger>, 1girl, solo, standing in a sunlit courtyard, medium shot",
    "kept_tags": ["1girl", "solo"],
    "scene": "standing in a sunlit courtyard, medium shot",
    "watermark": [],
    "borders": ["none"],
    "generated": { "model": "...", "temperature": 1.0, "top_p": 0.95 }
  },
  "official_style": {
    "caption_long": "...",
    "caption_medium": "...",
    "caption_short": "...",
    "watermark": [],
    "borders": ["none"],
    "generated": { "model": "...", "temperature": 1.0, "top_p": 0.95 }
  }
}
```

`dataset.moss_caption_source` selects which field training reads (`tag_caption`, `official_long`, `official_medium`, or `official_short`). Setting `dataset.metadata` gives `metadata.jsonl` priority; if a sample lacks the chosen style, the loader falls back to `<stem>.txt`/`.caption` and `reg_caption`. Switching the caption source only requires re-running `encode-text`; the latent cache stays valid.

Batch size is a cache setting rather than an implicit guess: `cache.text_batch` and `cache.vae_batch` pin it (`0` sizes it from free VRAM), and the batch that built a cache is recorded in `train_cache/index.json` under `batch`. VAE latents do not depend on the batch, because `LatentCodec` runs convolutions in true fp32; text embeddings do, so pin the text batch when a cache must be reproducible.

`train.caption_dropout` is a real per-sample caption dropout rate in `[0, 1]`; it defaults to `0`. `encode-text` writes the unconditional prompt to `empty_text.safetensors` beside the cache. When the rate is greater than zero, training replaces that fraction of caption embeddings with this cached empty prompt; when it is zero, captions are never replaced. Enabling dropout on an older cache requires one `encode-text` pass to add the empty prompt.

## Interface Language

The WebUI includes built-in Simplified Chinese (`zh`), English (`en`), Japanese (`ja`), and Korean (`ko`) dictionaries (`moss_lora/web/static/i18n/<code>.json`) and switches languages from the top bar without reloading the page.

Under **Settings → Interface language**, export a template and import a custom language pack. Fill in `$code` (such as `ru` or `pt-br`), `$name`, and the translated strings (keeping placeholders like `{n}` and `{path}`), then upload the file to save it at `i18n/<code>.json` (`.gitignore` excludes this directory). A matching `$code` overrides the built-in dictionary until deleted; missing keys fall back in order: active language → Simplified Chinese → raw key.

## Exporting to ComfyUI

Training checkpoints (`runs/<name>/checkpoints/stepXXXXXX.safetensors`, `final.safetensors`) and the final export `runs/<name>/<name>.safetensors` use ComfyUI-compatible kohya keys. Optimizer and RNG state live separately in `state.safetensors` (or `<ckpt>.state.safetensors`).

Tensor keys start with `lora_unet_` and replace dots in the layer path with underscores, which ComfyUI's `Load LoRA` node maps onto `diffusion_model.layers.<layer>.proj.<module>.weight`:

```text
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_down.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_up.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.alpha
```

Drop the `.safetensors` file into `<COMFYUI_DIR>/models/loras/` and connect a `Load LoRA` node. You can convert checkpoints to `diffusers` format (`transformer.layers.*`) or batch-export a step range from the WebUI **Export** tab or the CLI (pure key remapping without loading the 5B model):

```bash
# Export a single step checkpoint in both kohya and diffusers formats
.venv/bin/python -m moss_lora.cli export \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --checkpoint 1000 \
    --formats kohya,diffusers \
    --out "workspaces/<WORKSPACE>/exports"

# Batch-export a step range and package as a zip archive
.venv/bin/python -m moss_lora.cli export-batch \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --steps 250-1000:250 \
    --formats kohya \
    --out "workspaces/<WORKSPACE>/exports" \
    --zip "workspaces/<WORKSPACE>/exports/batch.zip"
```

## Testing

Unit tests run without a GPU and cover sidecar read/write and migration, four-language dictionary parity, API key storage rules, and documentation sanitization. An optional test verifies right-padded batch text encoding against single-sample encoding on a local text encoder:

```bash
.venv/bin/python tests/test_sidecar.py
.venv/bin/python tests/test_i18n.py
.venv/bin/python tests/test_caption_key.py
.venv/bin/python tests/test_docs_sanitized.py

# Optional: verify right-padded batch encoding against single-item encoding
MOSS_TE=<MODELS_DIR>/text_encoders/Qwen3.5-9B .venv/bin/python tests/test_text_padding.py
```

The frontend regression harness [smoke.mjs](<tests/ui/smoke.mjs>) uses native Node.js ESM and the Chromium DevTools Protocol (zero external npm packages) to walk all 4 languages and 9 tabs at 1440×1000, checking for console errors, failed requests, horizontal overflow, clipped text, and untranslated raw keys:

```bash
.venv/bin/python -m moss_lora.cli web --port 7891 &
chromium --headless=new --remote-debugging-port=9401 \
    --user-data-dir=.tmp/chrome --no-sandbox --disable-gpu about:blank &
node tests/ui/smoke.mjs --base http://127.0.0.1:7891 --cdp http://127.0.0.1:9401 \
    --langs zh,en,ja,ko --out .tmp/ui.json
```

## Directory Structure

```text
moss-image-trainer/
├── setup.sh / setup.ps1 / setup_fa2.py   # Create .venv, install dependencies, and match FlashAttention-2 wheels
├── run.sh / run.ps1          # Check environment and launch the WebUI
├── settings.example.yaml     # Machine-level configuration template
├── moss_lora/
│   ├── cli.py                # CLI entry point (cache / encode-text / train / export / web, etc.)
│   ├── config.py             # Config dataclasses, path resolution, validation, and VRAM presets
│   ├── model.py              # Official MOSS-Image DiT adaptation (SDPA / FA2 and gradient checkpointing)
│   ├── weights.py / convrot.py        # DiT/VAE key mapping and ConvRot int8 dequantization
│   ├── data.py / cache.py    # Dataset scanning, per-image sizing, and image latent caching
│   ├── hf_text.py / encode.py # Local Qwen3.5-9B loader and text embedding caching
│   ├── captioner.py          # Caption generation worker and sidecar JSON management
│   ├── trainer.py            # Flow Matching training loop, sample previews, and checkpoint management
│   ├── quant.py / offload.py # NF4 frozen-layer quantization and CPU block swap
│   ├── lora_file.py / export_batch.py # Model-free LoRA format conversion and batch export
│   ├── i18n.py / settings.py # Language pack merging and machine-level settings I/O
│   └── web/                  # FastAPI routes/services and native ES module frontend
└── tests/                    # Python unit tests and CDP frontend regression harness
```

See [docs/DESIGN.md](<docs/DESIGN.md>) for module boundaries, data flow, and frontend-backend contracts.

## Known Limitations

- **Batch size and throughput**: A micro-batch is assembled by token budget, so images keep their own resolutions and `train.bin_tokens` (not `batch_size`) sets how much work one step does. Install FlashAttention-2 to switch the default from `padded` (token padding plus a key mask) to `packed` (varlen, no mask).
- **Low-VRAM precision trade-offs**: `model.dtype: float16` with `vram.fp16_accumulate: true` is more prone to numerical instability than `bfloat16` at higher learning rates or long training runs. Quantization (`quantize_dit: ffn|all`) targets the 8 GB and 6 GB tiers; the 16 GB and 12 GB tiers keep bf16 weights and stream the trailing blocks from host memory instead, which costs step time but not weight precision. `nf4` is the smallest format and `fp8` the steadier one, and quantizing the conditioning half on purpose compounds error into every block, which is why it stays off unless asked for. `vram.offload_activations: true` writes the gradient-checkpoint boundaries to pinned host memory and saves roughly 1 GB of card memory at 1024x1024 for about a tenth more step time; it requires gradient checkpointing and turns `torch.compile` off.
- **Training scope**: LoRA adapters attach only to the 5B DiT (`q`, `k`, `v`, `o`, `ffn_up_gate`, and `ffn_down` across 42 layers by default, plus optional `text_embedder`); the trainer freezes the Qwen3.5-9B text encoder and the VAE.

## License and Acknowledgements

- Model architecture ([model.py](<moss_lora/model.py>)), scheduler ([scheduler.py](<moss_lora/scheduler.py>)), and weight mapping adapt [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) (Apache-2.0 License). See [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), and [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) for base and repackaged weights.
- The bundled Qwen3.5 tokenizer and model config (`moss_lora/assets/`) use the Apache-2.0 License; see [QWEN_LICENSE.txt](<moss_lora/assets/QWEN_LICENSE.txt>).
