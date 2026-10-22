# AGENTS.md

Read these rules before modifying code or documentation in `moss-image-trainer`.

## Project Goal and Upstream References

Build a local LoRA trainer for MOSS-Image-SFT using a PyTorch training stack, a FastAPI backend, and a buildless ES module frontend. Reuse the upstream model architecture, weight loaders, and key mappings directly so this repository maintains only what upstream omits: dataset scanning and bucketing, latent and text feature caching, the Flow Matching training loop, sample previews, caption generation, and the WebUI.

Consult these public upstream sources:
- Official HuggingFace weights: https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
- Official GitHub repository: https://github.com/OpenMOSS/MOSS-Image
- ComfyUI repackaged weights: https://huggingface.co/Comfy-Org/MOSS-Image

Use these placeholders in all documentation and configuration examples to keep files portable across machines:
- `<MODELS_DIR>`: local model root directory containing `moss-image` weights under `diffusion_models`, `text_encoders`, and `vae`
- `<DATASET_DIR>`: local training or regularization dataset directory
- `<WORKSPACE>`: workspace name mapping to `workspaces/<WORKSPACE>/workspace.yaml`
- `<ckpt>`: checkpoint filename produced by training
- `<VLM_ENDPOINT>`: local vision-language endpoint URL for caption generation

## Skills and Documentation Entry Points

- Read `skills/moss-lora-trainer/SKILL.md` before running any fine-tuning task; it defines the step-by-step workflow from workspace creation, captioning, two-stage caching, and VRAM planning to training, monitoring, and export, along with completion criteria and recovery steps.
- Read `docs/DESIGN.md` before changing module boundaries or training internals to understand architectural trade-offs.
- Read `README.md` for user-facing CLI and WebUI usage.
- Run `.venv/bin/python tests/test_docs_sanitized.py` after any documentation edit to catch leaked machine-specific data.

## Action Confirmation

Diagnose the root cause and report your findings before touching any file when the user asks a question, reports an error, or discusses a design.

- Run read-only commands (file inspection, log checks, syntax checks, sanitization scans, unit tests) immediately to gather facts.
- List the target files and expected impact before modifying source code, rewriting configs, clearing caches, starting training runs, or launching batch captioning, and wait for explicit user approval.

## Git Rules

Keep all version control operations strictly local so unapproved code or private artifacts never leave the machine.

- Run `git init`, `git add`, `git commit`, `git diff`, `git log`, and `git status` locally as needed.
- Never run `git remote add`, `git push`, `git rebase`, configure submodules, or upload to GitHub without explicit user permission; do not even configure a remote until the user authorizes an upload.
- Follow the user's per-command instructions on whether to push, whether to rebase, and which branch to target once upload is approved.
- Track only the trainer core, agent skills, and public documentation:
  - `moss_lora/`: trainer package, CLI, and WebUI backend and frontend
  - `tests/`: Python unit tests and frontend regression fixtures
  - `docs/` and `skills/`: public architecture documentation, usage guides, and agent skill definitions
  - `README.md`, `settings.example.yaml`, `requirements.txt`, `setup.sh`, `setup_fa2.py`, `run.sh`, `run.ps1`, `.gitignore`
- Ignore all local, private, and generated paths via `.gitignore`, and never bypass ignore rules with `git add -f`:
  - `experiments/`: local benchmark records, profiling logs, and comparison charts
  - `workspaces/`, `output/`, `runs/`, `logs/`: local workspace configs, caches, checkpoints, and runtime logs
  - `.venv/`, `third_party/`: local virtual environments and cloned upstream reference repositories
  - `settings.yaml`, `i18n/`, `caption.sh`: local settings, custom dictionaries, and private scripts
  - `tests/.sanitize-local.txt`: machine-specific sanitization rules
  - `AGENTS.md` is tracked and published together with `docs/` and `skills/`
  - Private dataset directories, editor and agent caches (`.dsh/`, `.agents/`, `.run/`, `.tmp/`, `.vscode/`), and weight files (`*.safetensors`, `*.ckpt`, `*.pt`, `*.pth`, `*.bin`)

## Investigation and Code Reuse

Inspect the upstream source tree before writing code so you call existing implementations instead of duplicating them.

1. Read `third_party/MOSS-Image` (`README`, `USERGUIDE`, `inference_runtime.py`) and `third_party/MOSS-Image/comfyui/` (`README.md`, `model.py`, `loaders.py`, `hf.py`, `text_encoder.py`, `nodes.py`, `export.py`) before changing models, loaders, or inference logic. Run `grep -n "def \|class "` to inventory existing functions and classes first; `clip_state` and `load_clip` in `loaders.py` already load the text encoder, so skipping this check wastes work on redundant format converters.
2. Import upstream model definitions, weight loaders, key mappings, and dequantization routines directly, and write only the missing trainer modules (datasets, caches, training loop, sigma schedule, optimizer wrappers, WebUI). Stop immediately and re-check `third_party/MOSS-Image` if you catch yourself writing key mappings, dequantization, or weight format conversions by hand.
3. Never download models, weights, datasets, or packages from the network inside scripts, tests, or background jobs; list the package name and purpose and get explicit user approval before installing or fetching anything.

## Environments and Division of Labor

Isolate trainer dependencies inside the project virtual environment so host environments stay untouched.

- Install or update packages only inside `./.venv` (created via `./setup.sh`) or a dedicated new virtual environment; never install packages into existing host text-encoding or training environments.
- Run text encoding in the text-encoding environment or `./.venv` to load the official int8 `convrot` encoder or local Qwen3.5-9B and write feature tensors to disk:
  ```bash
  .venv/bin/python -m moss_lora.cli encode-text --config workspaces/<WORKSPACE>/workspace.yaml
  ```
- Run latent caching, LoRA training, periodic sampling, checkpoint export, and the WebUI server in `./.venv`:
  ```bash
  .venv/bin/python -m moss_lora.cli cache --config workspaces/<WORKSPACE>/workspace.yaml
  .venv/bin/python -m moss_lora.cli train --config workspaces/<WORKSPACE>/workspace.yaml
  .venv/bin/python -m moss_lora.cli web --port 7869
  ```

## Documentation and Sanitization

Keep public files (`README.md`, `AGENTS.md`, `settings.example.yaml`, `docs/`, `skills/`) free of host-specific traces so the repository stays portable.

- Strip all local absolute paths, mount paths, system usernames, local environment variable names, host conda environment names, private dataset directory names, workspace instance names, local service names, and trained character names.
- Omit measured hardware numbers (peak VRAM, step latency, throughput, temperature, power draw, GPU model, private dataset image count) from public docs; store benchmarks and charts under `experiments/` without citing them in public files.
- Write in the active voice with concrete nouns, cut filler modifiers and clichés, combine choppy fragments, and place emphatic words at the end of the sentence.
- Run the sanitization check after every documentation change and verify zero hard hits and zero warnings:
  ```bash
  .venv/bin/python tests/test_docs_sanitized.py
  ```

## Acceptance Criteria

Validate code changes and training runs against final inference quality in ComfyUI.

- Load base weights from `<MODELS_DIR>/{diffusion_models,text_encoders,vae}/moss-image` (or the paths configured in `workspace.yaml`) and read training images and captions from `<DATASET_DIR>`.
- Save checkpoints and exported files (`<ckpt>.safetensors`) using `lora_unet_*` keys by default, and store optimizer states separately in `state.safetensors`.
- Verify that the exported LoRA loads directly in the standard ComfyUI LoRA loader node and reproduces the target character features during inference sampling.
