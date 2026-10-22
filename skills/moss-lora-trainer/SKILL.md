---
name: moss-lora-trainer
description: "Run end-to-end MOSS-Image LoRA fine-tuning in the moss-image-trainer repository: create workspaces, write workspace.yaml, generate captions, build latent and text feature caches, plan VRAM presets, run and monitor training, sample previews, and export weights. Load this skill when the task involves training, caching, captioning, sampling, or exporting a MOSS-Image LoRA."
when-to-use: "Use when the user asks to fine-tune a MOSS-Image LoRA, build training caches, generate dataset captions, troubleshoot training or caching errors, render preview samples, or export ComfyUI/diffusers weights."
---

# MOSS-Image LoRA Fine-Tuning Playbook

This playbook defines the standard procedure for an AI agent to run a complete MOSS-Image LoRA fine-tuning workflow from the repository root via the CLI. Every command binds to real arguments in `moss_lora/cli.py`. Execute steps in order; never skip steps or guess flag names.

## 1. Scope and Capabilities

This repository provides a single-node, single-GPU LoRA training stack and FastAPI WebUI for MOSS-Image (5B single-stream DiT, 42 Transformer layers, frozen Qwen3.5-9B text encoder, 32-channel FLUX.2 VAE). The trainer splits the pipeline into two physically isolated stages—offline caching and online training—running the VAE and Qwen3.5-9B in batches to persist features to disk before loading only the 5B DiT in a separate process for backpropagation, preventing all three models from occupying VRAM simultaneously.

**Supported capabilities:**
- Recursively scan image directories, assign resolutions across 7 fixed aspect-ratio buckets or native aspect ratios, and read captions from sidecar JSON, `.txt`/`.caption`, or `metadata.jsonl`.
- Call an OpenAI-compatible vision endpoint to generate dual-style sidecar JSON files (`schema: 2`) containing `tag_caption` (original tags plus scene prose) and `official` (long, medium, and short descriptions).
- Merge image latents (`latent`) and text features (`text_embeds`, `text_mask`) into `<uid>.safetensors` named by image path hash, and write `index.json` (`schema: 2`).
- Inject PEFT LoRA adapters (covering `layers.<0..41>.proj.<q|k|v|o|ffn_up_gate|ffn_down>` by default) and run Flow Matching velocity-field training, holdout validation, and periodic sampling.
- Produce `lora_unet_*` weight files natively recognized by ComfyUI, with zero-model-load instant conversion to `diffusers` format or batch packaging across step ranges.

**Out of scope:**
- Full-parameter fine-tuning, or fine-tuning the Qwen3.5-9B text encoder or FLUX.2 VAE.
- Multi-node or multi-GPU distributed training (no DDP, FSDP, or DeepSpeed).
- Automatic network downloads of base model weights, tokenizers, or third-party Python packages.
- Combining samples with different text lengths under the default `batch_mode: bucket`; to increase the effective batch size, raise `train.grad_accum` first or enable `batch_mode: packed`.

## 2. Path Placeholders and Execution Rules

All commands in this playbook use the following placeholders:
- `<MODELS_DIR>`: Local model root directory (containing DiT, Qwen3.5-9B, and FLUX.2 VAE weights).
- `<DATASET_DIR>`: Local training dataset directory (or regularization dataset directory `<REG_DATASET_DIR>`).
- `<WORKSPACE>`: Workspace name (maps to `workspaces/<WORKSPACE>/workspace.yaml`; letters, digits, and `._-` only).
- `<ckpt>`: Checkpoint filename or step number (e.g., `final.safetensors`, `step000500.safetensors`, or step `500`).
- `<VLM_ENDPOINT>`: OpenAI-compatible vision service URL in the form `http://<host>:<port>/v1`; omit this flag when `caption.base_url` is already configured in `settings.yaml`.

**Execution rule: Before running any command, replace `<MODELS_DIR>`, `<DATASET_DIR>`, `<WORKSPACE>`, and `<ckpt>` with real paths, or `export` environment variables of the same names in the current shell and pass them as variables. Never pass angle-bracket placeholders literally to a command!**

Run all commands from the repository root (the directory containing `moss_lora/` and `.venv/`), using `.venv/bin/python -m moss_lora.cli <subcommand>` as the fixed Python entry point.

## 3. Preflight Checks

Before creating a workspace, run the following check script from the repository root to verify that the environment, GPU, base weights, and dataset are ready:

```bash
set -euo pipefail

# 1. Confirm current directory is the repository root and virtualenv is ready
test -d moss_lora && test -x .venv/bin/python
.venv/bin/python -c "import torch, safetensors, peft, yaml; assert torch.cuda.is_available(), 'CUDA unavailable'"

# 2. Check GPU visibility and free VRAM
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 3. Bind real path variables (replace the right-hand side with actual paths before running)
export MODELS_DIR="<MODELS_DIR>"
export DATASET_DIR="<DATASET_DIR>"
export WORKSPACE="<WORKSPACE>"

# 4. Verify all three base model paths exist
test -e "$MODELS_DIR/diffusion_models/moss_image_dit.safetensors"
test -e "$MODELS_DIR/text_encoders/Qwen3.5-9B"
test -e "$MODELS_DIR/vae/flux2_vae.safetensors"

# 5. Scan dataset image count, caption coverage, and bucket distribution
.venv/bin/python -m moss_lora.cli info --dataset "$DATASET_DIR"
```

**Success criteria:**
- `nvidia-smi` lists at least one available GPU.
- `cli info` exits with code `0`, prints `images: N  with caption: M` with `N > 0`, and lists bucket counts such as `1024x1024` below. If `with caption: 0` and no default caption is configured, run Step 3 (caption generation) or add matching `.txt` files before caching.

## 4. End-to-End Fine-Tuning Workflow

The 9 steps below form a one-way pipeline. Never start Step 7 (training) before completing both Step 4 (latent caching) and Step 5 (text feature encoding).

### Step 1: Create Workspace Directories

- **Purpose**: Create an isolated workspace directory tree for configs, caches, logs, and training artifacts.
- **Command**:
  ```bash
  export WORKSPACE="<WORKSPACE>"
  mkdir -p "workspaces/$WORKSPACE"/{configs,train_cache,reg_cache,runs,logs,exports}
  ```
- **Success criteria**: `test -d "workspaces/$WORKSPACE/train_cache"` exits with code `0`.
- **Common errors and fixes**: Slashes or spaces in the workspace name fail validation; use only letters, digits, dots, underscores, and hyphens.

### Step 2: Write and Validate `workspace.yaml`

- **Purpose**: Write model paths, dataset paths, LoRA architecture, training hyperparameters, VRAM presets, and preview prompts, then run static path validation.
- **Command**:
  ```bash
  cat <<EOF > "workspaces/$WORKSPACE/workspace.yaml"
  name: $WORKSPACE
  output_dir: runs
  model:
    dit: $MODELS_DIR/diffusion_models/moss_image_dit.safetensors
    text_encoder: $MODELS_DIR/text_encoders/Qwen3.5-9B
    text_encoder_config: ""
    vae: $MODELS_DIR/vae/flux2_vae.safetensors
    dtype: bfloat16
  dataset:
    path: $DATASET_DIR
    caption_ext: [".txt", ".caption"]
    bucket_step: 16
    max_area: 1048576
    native_resize: false
    repeats: 1
    reg_path: ""
    reg_repeat: 1
    reg_caption: ""
    moss_caption_source: tag_caption
  cache:
    dir: train_cache
    reg_dir: reg_cache
    overwrite: false
  lora:
    rank: 32
    alpha: 32
    targets: [q, k, v, o, ffn_up_gate, ffn_down]
    layers: []
    include_text_embedder: false
    init: kaiming
  train:
    steps: 1000
    batch_size: 1
    grad_accum: 1
    batch_mode: bucket
    attention: auto
    bin_tokens: 0
    lr: 0.0001
    optimizer: adamw
    warmup: 50
    lr_scheduler: constant
    caption_dropout: 0.05
    seed: 42
    save_every: 250
    sample_every: 250
    eval_every: 100
    eval_holdout: 0.05
    save_state: last
  vram:
    preset: auto
    gradient_checkpointing: auto
    blocks_to_swap: -1
    optimizer_in_8bit: false
    quantize_dit: none
    fp16_accumulate: false
    compile: true
  export:
    formats: [kohya]
  preview:
    prompts:
      - "a portrait in soft daylight, high detail"
    negative_prompt: "low quality, blurry, deformed"
    width: 1024
    height: 1024
    steps: 50
    guidance: 3.0
    shift: 5.0
    uncond_skip_layers: [22]
    seed: 42
  EOF

  # Validate configuration
  .venv/bin/python -c "from moss_lora.config import Config; p = Config.load('workspaces/$WORKSPACE/workspace.yaml').validate(); assert not p, p; print('config valid')"
  ```
- **Success criteria**: The validation command prints `config valid` and exits with code `0`.
- **Common errors and fixes**:
  - `model.text_encoder directory has no config.json`: When `text_encoder` points to a single-file `.safetensors` checkpoint (such as ConvRot int8 quantized weights), the path must end with `.safetensors`; when it points to a directory, that directory must contain `config.json`.
  - `model.dit uses an unresolved variable`: The config contains an undefined `${VAR}` reference; replace it immediately with the resolved path.

### Step 3: Generate Dataset Captions (Optional)

Skip this step if the dataset already has matching `.txt` files or `schema: 2` sidecar JSON files.

- **Purpose**: Call an OpenAI-compatible vision model endpoint to generate `<stem>_moss-image-trainer.json` next to each image.
- **Command**:
  ```bash
  # Dry-run first to print the request prompt for the first image and verify parameters
  .venv/bin/python -m moss_lora.cli caption \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --styles tag_caption,official \
      --base-url "<VLM_ENDPOINT>" \
      --api-key VLM_API_KEY \
      --dry-run

  # Generate captions and validate the resulting sidecar JSON files
  .venv/bin/python -m moss_lora.cli caption \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --styles tag_caption,official \
      --base-url "<VLM_ENDPOINT>" \
      --api-key VLM_API_KEY \
      --workers 6 \
      --retries 3

  .venv/bin/python -m moss_lora.cli caption \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --styles tag_caption,official \
      --validate-only
  ```
- **Success criteria**: `--validate-only` prints `validated N sidecars, 0 invalid` and exits with code `0`.
- **Common errors and fixes**:
  - `skipped without reference: N`: The `tag_caption` style requires a matching `.txt`/`.caption` file next to each image as the source tag list. If no reference tags exist, pass `--styles official` and set `dataset.moss_caption_source` in `workspace.yaml` to `official_long`, `official_medium`, or `official_short`.
  - Legacy flat sidecar errors: Run `.venv/bin/python -m moss_lora.cli caption --config "workspaces/$WORKSPACE/workspace.yaml" --migrate` to upgrade sidecars to `schema: 2`.

### Step 4: Build Image Latent Cache (Latent Cache)

- **Purpose**: Load the FLUX.2 VAE to encode training images (and optional regularization images) into `float16` latents, writing `train_cache/<uid>.safetensors` (`reg_cache/` for regularization images) and generating `train_cache/index.json`.
- **Command**:
  ```bash
  .venv/bin/python -m moss_lora.cli cache \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --workers 6
  ```
- **Success criteria**: The command prints `index: .../train_cache/index.json (N items)` and exits with code `0`.
- **Common errors and fixes**:
  - Never pass `--text-only`: `cache --text-only` exits with an error immediately; always run `encode-text` in Step 5 for text encoding.
  - Tight VRAM: Append `--batch-size 1` (defaults to auto-scaling from free VRAM, capped at 4).

### Step 5: Build Text Features and Preview Cache (Text Encoding)

- **Purpose**: Load local Qwen3.5-9B to encode each image caption into `text_embeds` and `text_mask`, merge them into the same `<uid>.safetensors`, encode `preview.prompts` into `train_cache/preview.safetensors`, and update `text_len` in `index.json`.
- **Command**:
  ```bash
  .venv/bin/python -m moss_lora.cli encode-text \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --quantize none
  ```
- **Success criteria**:
  Run the following verification command to confirm `schema == 2`, `text_len > 0` for every item, and the presence of the preview file:
  ```bash
  .venv/bin/python -c "
  import json, os
  idx = json.load(open('workspaces/$WORKSPACE/train_cache/index.json'))
  assert idx.get('schema') == 2, 'invalid schema'
  items = idx.get('items', [])
  assert items and all(int(i.get('text_len', 0)) > 0 for i in items), 'unencoded text items found'
  assert os.path.isfile('workspaces/$WORKSPACE/train_cache/preview.safetensors'), 'missing preview.safetensors'
  print(f'cache ready: {len(items)} items')
  "
  ```
- **Common errors and fixes**:
  - OOM when loading the 9B encoder on 12 GB or smaller GPUs: Change `--quantize none` to `--quantize fp8` or `--quantize nf4` and append `--batch-size 4`.
  - Only `preview.prompts` changed: Run `encode-text --config "workspaces/$WORKSPACE/workspace.yaml" --preview-only` to refresh the preview bundle alone.

### Step 6: Plan VRAM Preset and Render LoRA-Free Baseline Images

- **Purpose**: Write the optimal quantization and block-swapping settings for the current GPU capacity, and render baseline comparison images with the base model before training.
- **Command**:
  ```bash
  # Plan parameters for the VRAM budget (--preview reserves sampling headroom and writes back to workspace.yaml)
  .venv/bin/python -m moss_lora.cli plan \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --preview \
      --write "workspaces/$WORKSPACE/workspace.yaml"

  # Render LoRA-free baseline samples (written to workspaces/<WORKSPACE>/baseline/baseline_*.png)
  .venv/bin/python -m moss_lora.cli baseline \
      --config "workspaces/$WORKSPACE/workspace.yaml"
  ```
- **Success criteria**: `workspaces/$WORKSPACE/baseline/baseline_0.png` exists and has a file size greater than 0.
- **Common errors and fixes**: `no preview embeddings` means Step 5 did not generate `preview.safetensors`; verify that `preview.prompts` in `workspace.yaml` is non-empty and re-run `encode-text --preview-only`.

### Step 7: Run Smoke Test and Launch Training

- **Purpose**: Run a 3-step smoke test first to verify forward pass, backward pass, holdout evaluation, sampling, and checkpoint saving end-to-end before launching the full training run.
- **Command**:
  ```bash
  # 1. Smoke preflight: fixed 3 steps covering forward/backward/eval/sample/save into <name>-smoke
  .venv/bin/python -m moss_lora.cli train \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --name "$WORKSPACE" --smoke

  # 2. Clean up smoke artifacts and launch full training (tee logs to workspaces/<WORKSPACE>/logs/)
  rm -rf "workspaces/$WORKSPACE/runs/${WORKSPACE}-smoke"
  .venv/bin/python -m moss_lora.cli train \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      2>&1 | tee "workspaces/$WORKSPACE/logs/${WORKSPACE}.log"
  ```
- **Success criteria**: The terminal logs `step 1/1000  loss ...` line by line and finishes with `checkpoint .../checkpoints/final.safetensors` and `exported .../<WORKSPACE>.safetensors (756 tensors)`.
- **Common errors and fixes**:
  - Resuming an interrupted run: Pass the latest checkpoint path via `.venv/bin/python -m moss_lora.cli train --config "workspaces/$WORKSPACE/workspace.yaml" --resume "workspaces/$WORKSPACE/runs/$WORKSPACE/checkpoints/<ckpt>.safetensors"`, or run `.venv/bin/python -m moss_lora.cli queue --workspace "workspaces/$WORKSPACE"` to scan and resume all unfinished jobs in the workspace automatically.

### Step 8: Monitor Training and Run On-Demand Sampling

- **Purpose**: Track training metrics and heartbeat status, and render extra preview images or custom-prompt samples for specific checkpoint steps.
- **Command**:
  ```bash
  RUN_DIR="workspaces/$WORKSPACE/runs/$WORKSPACE"

  # Inspect the latest 10 training/validation metric entries and the heartbeat file
  tail -n 10 "$RUN_DIR/metrics.jsonl"
  cat "$RUN_DIR/training.json"

  # Backfill preview images for a specific step (e.g., step 500) and redraw trajectory.png
  .venv/bin/python -m moss_lora.cli preview \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --step 500

  # Encode custom prompts and sample standalone images from a specific checkpoint
  .venv/bin/python -m moss_lora.cli encode-prompt \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --out "workspaces/$WORKSPACE/custom_prompts.safetensors" \
      --prompt "1girl, solo, upper body, looking at viewer, outdoor sunlight" \
      --negative "low quality, blurry"

  .venv/bin/python -m moss_lora.cli sample \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --checkpoint "$RUN_DIR/checkpoints/<ckpt>.safetensors" \
      --embeds "workspaces/$WORKSPACE/custom_prompts.safetensors" \
      --out "$RUN_DIR/custom_samples" \
      --steps 50 --width 1024 --height 1024
  ```
- **Success criteria**: `status` in `training.json` is `"running"` (in progress) or `"done"` (completed), and the sampling command prints the generated `.png` paths.

### Step 9: Export LoRA Weights

- **Purpose**: Export a specific checkpoint or step range into ComfyUI (`kohya`) or `diffusers` format.
- **Command**:
  ```bash
  # Export a single checkpoint (generating both kohya and diffusers formats)
  .venv/bin/python -m moss_lora.cli export \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --checkpoint final \
      --formats kohya,diffusers \
      --out "workspaces/$WORKSPACE/exports"

  # Batch-export a step range and package as a zip archive (pure key conversion without loading the 5B model)
  .venv/bin/python -m moss_lora.cli export-batch \
      --config "workspaces/$WORKSPACE/workspace.yaml" \
      --steps "250-1000:250" \
      --formats kohya \
      --out "workspaces/$WORKSPACE/exports" \
      --zip "workspaces/$WORKSPACE/exports/${WORKSPACE}_batch.zip"
  ```
- **Success criteria**: Matching `.safetensors` and `.zip` files appear under `workspaces/$WORKSPACE/exports/`.

### Appendix: One-Shot Pipeline Script

After replacing the three path variables at the top, save or pipe this script into bash to execute the complete loop:

```bash
set -euo pipefail

export MODELS_DIR="<MODELS_DIR>"
export DATASET_DIR="<DATASET_DIR>"
export WORKSPACE="<WORKSPACE>"
CFG="workspaces/$WORKSPACE/workspace.yaml"

mkdir -p "workspaces/$WORKSPACE"/{configs,train_cache,reg_cache,runs,logs,exports}
# (Ensure $CFG has been written according to Step 2)
.venv/bin/python -m moss_lora.cli cache --config "$CFG"
.venv/bin/python -m moss_lora.cli encode-text --config "$CFG"
.venv/bin/python -m moss_lora.cli baseline --config "$CFG"
.venv/bin/python -m moss_lora.cli train --config "$CFG"
.venv/bin/python -m moss_lora.cli export --config "$CFG" --checkpoint final --formats kohya --out "workspaces/$WORKSPACE/exports"
```

## 5. Parameter Decision Tables

### 5.1 VRAM Presets (`vram.preset` and `cli plan`)

| VRAM Capacity (GB) | `vram.preset` | `model.dtype` | `quantize_dit` | `blocks_to_swap` | `optimizer_in_8bit` | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **32 / 24 / 16** | `32` / `24` / `16` | `bfloat16` | `none` | `0` | `false` | Full-precision frozen backbone; highest throughput and best detail fidelity |
| **12** | `12` | `bfloat16` or `float16` | `ffn` | `0` | `false` | NF4 quantization on FFN linear layers only; balances precision and VRAM |
| **8** | `8` | `float16` (`fp16_accumulate: true`) | `all` | `0` | `false` | NF4 quantization on all frozen linear layers without CPU block swapping |
| **6** | `6` | `float16` (`fp16_accumulate: true`) | `all` | `20` | `true` | NF4 quantization combined with 20-layer CPU block swapping and 8-bit optimizer |

### 5.2 Core Training Hyperparameter Decisions

- **When to enable a regularization dataset (`dataset.reg_path`)**:
  - Enable when fine-tuning a specific character on a small dataset (e.g., 15–40 images) and preventing trigger-word bleed into generic class tokens (such as `"1girl"` without the trigger word still generating that character).
  - Set `dataset.reg_path: "<REG_DATASET_DIR>"` and `dataset.reg_caption: "1girl, solo"` (or place matching caption files inside the regularization directory). Regularization images must not contain the character trigger word.
- **How to choose `train.batch_mode`**:
  - `bucket` (default): Safest option; pair `batch_size: 1` with `grad_accum: 2..4` to scale the effective batch size. Because images in the same resolution bucket usually have different `text_len` values, raising `batch_size` alone under `bucket` mode inflates activation VRAM without unlocking vectorized batching.
  - `packed`: Enable when FlashAttention-2 is installed and you want to pack samples with different aspect ratios and text lengths for higher throughput. Packs samples up to `train.bin_tokens` (`0` computes the token budget automatically) with zero padding waste.
  - `padded`: Use when FlashAttention-2 is unavailable but multi-image batching within a token budget is desired; pads 1D sequences for SDPA execution.
- **How to set `lora.rank` and `lora.alpha`**:
  - Single-character appearance fine-tuning: `rank: 32, alpha: 32` (default) with learning rate `1e-4`.
  - Simple concepts or small datasets prone to overfitting: `rank: 16, alpha: 16`.
  - Complex art styles or multi-concept joint fine-tuning: `rank: 64, alpha: 64`, lower learning rate to `5e-5`, and set `lr_scheduler: cosine`.

## 6. CLI Subcommand and Argument Reference (Authoritative Match for `moss_lora/cli.py`)

| Subcommand | Required Arguments | Optional Arguments and Defaults |
| :--- | :--- | :--- |
| `info` | None (requires `--config` or `--dataset`) | `--config`, `--dataset`, `--preset` |
| `cache` | None (typically pass `--config`) | `--config`, `--dataset`, `--cache-dir`, `--preset`, `--latents-only`, `--text-only` (disabled; exits with error), `--batch-size` (`None`), `--workers` (`6`) |
| `encode-text` | `--config` | `--overwrite`, `--text-model` (`""`), `--quantize` (`none\|fp8\|nf4_ffn\|nf4`, default `none`), `--preview-only`, `--batch-size` (`None`) |
| `encode-prompt` | `--config`, `--out`, `--prompt` (repeatable) | `--negative` (`""`), `--text-model` (`""`) |
| `caption` | None (requires `--config` or `--images`) | `--config` (`""`), `--images` (`""`), `--out` (`""`), `--styles` (`""`, defaults to `tag_caption,official`), `--overwrite`, `--migrate`, `--role` (`train\|reg`), `--base-url` (`""`), `--api-key` (`""`), `--model` (`""`), `--trigger` (`None`), `--prompt-tags` (`""`), `--prompt-natural` (`""`), `--no-thinking`, `--temperature`, `--top-p`, `--max-tokens`, `--workers`, `--retries`, `--structured` (`""\|auto\|on\|off`), `--limit` (`0`), `--dry-run`, `--validate-only` |
| `plan` | None | `--vram` (auto-detected), `--config` (`runs/default.yaml`), `--write`, `--preview` |
| `baseline` | `--config` | `--out` (`""`, defaults to `<workspace>/baseline`) |
| `train` | None (typically pass `--config`) | `--config`, `--dataset`, `--cache-dir`, `--output-dir`, `--preset`, `--steps`, `--name`, `--rank`, `--alpha`, `--lr`, `--batch`, `--grad-accum`, `--save-every`, `--sample-every`, `--eval-every`, `--save-state` (`[last\|all]`), `--no-save-state`, `--grad-checkpointing`, `--resume` |
| `preview` | `--config` | `--checkpoint` (`""`), `--step` (`0`), `--samples-dir` (`""`), `--base` |
| `sample` | None (typically pass `--config`) | `--config`, `--checkpoint`, `--embeds` (defaults to `<cache.dir>/preview.safetensors`), `--out` (`output/samples`), `--base`, `--width`, `--height`, `--steps` |
| `export` | `--config` | `--checkpoint` (`""`, accepts path, step integer, or `final`), `--out` (`""`), `--formats` (`""`, default `kohya`), `--suffix` (`None`), `--state` |
| `export-batch` | `--config` | `--run-dir` (`""`), `--steps` (`""`, e.g. `250-1000:250`), `--formats` (`kohya`), `--out` (`""`), `--zip` (`""`), `--state` |
| `queue` | `--workspace` | None |
| `web` | None | `--host` (`127.0.0.1`), `--port` (`7861`) |

## 7. Troubleshooting

1. **CUDA Out of Memory (OOM) during training or caching**
   - **Symptom**: `torch.cuda.OutOfMemoryError`. The trainer includes automatic fallback (incrementing `blocks_to_swap` by 8 up to 40 and retrying); if it still crashes or fails during caching:
   - **Fix**:
     - Latent caching OOM: Run `cli cache --config ... --batch-size 1`.
     - Text encoding OOM: Run `cli encode-text --config ... --quantize nf4 --batch-size 2`.
     - Training OOM: Lower `vram.preset` by one tier in `workspace.yaml` (e.g., `16` → `12` or `8`), set `train.batch_size: 1`, or explicitly raise `vram.blocks_to_swap: 20` (`vram.gradient_checkpointing` must be `true` when `blocks_to_swap` is enabled).
2. **Cache schema mismatch or missing index**
   - **Symptom**: `缓存未就绪：缺 index.json` (cache not ready: missing `index.json`) or `缓存格式不对：不是当前版本的索引` (invalid cache format: outdated index schema).
   - **Fix**: Check whether `"schema"` in `workspaces/<WORKSPACE>/train_cache/index.json` equals `2`. If missing or outdated, re-run `cli cache --config ...` followed by `cli encode-text --config ...` (no `--overwrite` needed; incremental logic backfills missing entries and rewrites the v2 index).
3. **`KeyError: text_embeds` or `缓存里缺 text_embeds, text_mask` when starting training**
   - **Symptom**: `cli train` was launched right after `cli cache` without text encoding, or `dataset.reg_path` was changed without encoding captions for the regularization set.
   - **Fix**: Run `.venv/bin/python -m moss_lora.cli encode-text --config "workspaces/<WORKSPACE>/workspace.yaml"` immediately and wait for `index: ...` before launching training.
4. **`视觉服务连不上` (vision endpoint unreachable) during caption generation**
   - **Symptom**: `RuntimeError: ... 视觉服务连不上（http://...）`.
   - **Fix**: Test whether the OpenAI-compatible vision endpoint is alive with `curl -sS <base_url>/models`. Do not blindly raise `--retries`; once the service recovers and `VLM_API_KEY` is non-empty, re-run `cli caption` (already captioned images are skipped automatically).
5. **Lingering training processes or occupied GPU VRAM**
   - **Symptom**: Starting a training or WebUI job reports `409 已有任务运行中` (409 job already running), or `nvidia-smi` shows a lingering Python process.
   - **Fix**: Inspect `pid` in `workspaces/<WORKSPACE>/runs/<run>/training.json`, check `/proc/<pid>/cmdline` to confirm it belongs to the current workspace's `moss_lora.cli train`, and send `kill -INT <pid>` for a graceful stop. Never run indiscriminate `killall python`.

## 8. Acceptance Checklist

After fine-tuning completes, run through the checks below to verify the deliverables:

1. **Loss convergence check**: Inspect `workspaces/<WORKSPACE>/runs/<name>/metrics.jsonl`, verify there are no `NaN`/`Inf` values, and confirm that `avg` training loss and `eval` validation loss decrease and stabilize over the second half of the run compared to the initial steps.
2. **Sample and trajectory check**: Confirm `workspaces/<WORKSPACE>/runs/<name>/samples/trajectory.png` exists, and compare `workspaces/<WORKSPACE>/baseline/baseline_0.png` against the final-step samples to verify that target features emerge without color collapse or severe overfitting.
3. **Exported file and tensor count check**: Run the one-liner below to validate the exported `.safetensors` file (42 layers × 6 target modules × 3 tensors `lora_down`/`lora_up`/`alpha` = **756 tensors** by default, reduced proportionally when restricting `lora.layers` or `lora.targets`, or higher when `include_text_embedder` is enabled; `diffusers` format omits `alpha` and has **504 tensors**):
   ```bash
   .venv/bin/python -c "
   from safetensors.torch import load_file
   sd = load_file('workspaces/$WORKSPACE/runs/$WORKSPACE/$WORKSPACE.safetensors')
   down = [k for k in sd if k.startswith('lora_unet_') and k.endswith('.lora_down.weight')]
   assert len(down) == 252 and len(sd) == 756, f'unexpected tensor count: total={len(sd)}, down={len(down)}'
   print('LoRA verified: 756 tensors (252 modules) in ComfyUI kohya format')
   "
   ```
4. **ComfyUI loading compatibility**: Confirm that tensor keys match `lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.(lora_down.weight|lora_up.weight|alpha)` so the file can be copied directly into `<COMFYUI_DIR>/models/loras/` and loaded by the standard `Load LoRA` node.

## 9. Prohibitions

- **Do not touch other workspaces**: Read and write only inside the target `workspaces/<WORKSPACE>/` directory; never modify or delete other workspace directories under `workspaces/`.
- **Do not abuse `--overwrite` to wipe caches**: `cache`, `encode-text`, and `caption` all include incremental and metadata-staleness checks (when prompts change, simply re-run `encode-text`). Unless `max_area`, `bucket_step`, or `native_resize` changed and altered latent dimensions, never pass `--overwrite` or delete `train_cache/`.
- **Do not download from the network**: Never run `pip install`, `huggingface-cli download`, `git clone`, `wget`, or `curl` in scripts or the terminal to fetch external models, weights, or packages.
- **Do not disrupt a running WebUI or concurrent training job**: A single-GPU machine can run only one heavy workload at a time. Before starting a CLI job, run `pgrep -af "moss_lora"` to check for active WebUI or training processes, and never force-kill a running WebUI server.
