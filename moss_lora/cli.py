"""Command line entry point: ``python -m moss_lora.cli <command>``."""

from __future__ import annotations

import argparse
import os
import sys

from .config import Config, apply_preset


def _cli_path(value: str) -> str:
    """Resolve a command line path against the current directory.

    YAML values resolve against the workspace; values typed on the command
    line mean what they mean in a shell. Without this, ``--cache-dir
    workspaces/x/train_cache`` resolves inside the workspace and looks for
    ``<ws>/workspaces/x/train_cache``.
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(value)))


def _load(args) -> Config:
    config = Config.load(args.config) if getattr(args, "config", None) else Config()
    if getattr(args, "dataset", None):
        config.dataset.path = _cli_path(args.dataset)
    if getattr(args, "cache_dir", None):
        config.cache.dir = _cli_path(args.cache_dir)
    if getattr(args, "output_dir", None):
        config.output_dir = _cli_path(args.output_dir)
    if getattr(args, "preset", None):
        config.vram.preset = args.preset
    apply_preset(config)
    return config.resolve_paths()


def cmd_cache(args):
    from .cache import CacheBuilder

    config = _load(args)
    problems = [p for p in config.validate() if not p.startswith("dataset.")]
    if problems:
        print("config problems:\n  " + "\n  ".join(problems))
        return 2
    builder = CacheBuilder(config)
    print(f"dataset: {len(builder.items)} images -> {builder.dir}")
    if args.text_only:
        builder.run_text()
        return 0
    builder.run_latents(batch_size=getattr(args, "cache_batch", None),
                        workers=getattr(args, "workers", 6) or 6)
    builder.write_index()
    print("\nnext: encode captions with the local Qwen3.5-9B encoder")
    print(f"  python -m moss_lora.cli encode-text --config {args.config or '<config.yaml>'}")
    return 0


def cmd_encode_text(args):
    from .encode import run_encode_text

    return run_encode_text(
        args.config,
        overwrite=args.overwrite,
        text_model=args.text_model,
        quantize=args.quantize,
        batch_size=args.batch_size,
        preview_only=getattr(args, "preview_only", False),
    )


def cmd_encode_prompt(args):
    from .encode import run_encode_prompt

    return run_encode_prompt(
        args.config,
        args.out,
        args.prompt,
        negative=args.negative,
        text_model=args.text_model,
    )


def cmd_caption(args):
    """Generate <name>_moss-image-trainer.json captions next to the dataset images."""
    import sys as _sys

    from . import captioner as worker
    from .config import Config
    from . import settings as runtime_settings

    config = Config.load(args.config).resolve_paths() if args.config else Config()
    images = args.images or os.path.abspath(os.path.expanduser(config.dataset.path))
    if not images:
        print("--images 或 dataset.path 必须有一个", file=_sys.stderr)
        return 2
    caption = runtime_settings.load_settings().get("caption") or {}
    styles = args.styles or "tag_caption,official"
    argv = ["--images", images, "--styles", styles, "--role", args.role,
            "--workers", str(args.workers if args.workers is not None else caption.get("workers", 6)),
            "--retries", str(args.retries if args.retries is not None else caption.get("retries", 3)),
            "--temperature", str(args.temperature if args.temperature is not None else caption.get("temperature", 1.0)),
            "--top-p", str(args.top_p if args.top_p is not None else caption.get("top_p", 0.95)),
            "--max-tokens", str(args.max_tokens if args.max_tokens is not None else caption.get("max_tokens", 16384))]
    if args.overwrite:
        argv.append("--overwrite")
    if args.migrate:
        argv.append("--migrate")
    # 地址、密钥、模型：命令行优先，其次 settings.yaml 里的 caption 段
    for flag, value in (("--base-url", args.base_url or caption.get("base_url", "")),
                        ("--api-key", args.api_key or caption.get("api_key", "")),
                        ("--model", args.model or caption.get("model", ""))):
        if value:
            argv += [flag, str(value)]
    trigger = args.trigger if args.trigger is not None else str(caption.get("trigger") or "")
    if trigger:
        argv += ["--trigger", trigger]
    if args.prompt_tags:
        argv += ["--prompt-tags", args.prompt_tags]
    if args.prompt_natural:
        argv += ["--prompt-natural", args.prompt_natural]
    if args.no_thinking or not caption.get("thinking", True):
        argv += ["--no-thinking"]
    if args.structured:
        argv += ["--structured", args.structured]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.out:
        argv += ["--out", args.out]
    if args.dry_run:
        argv += ["--dry-run"]
    if args.validate_only:
        argv += ["--validate-only"]
    return worker.main(argv)


def cmd_preview(args):
    """Re-render the preview images of one checkpoint (fills gaps after a crash)."""
    import glob

    from .trainer import Trainer

    config = _load(args)
    run_dir = os.path.join(config.output_dir, config.name)
    checkpoint = args.checkpoint
    step = args.step
    if not checkpoint:
        name = f"final.safetensors" if step in (0, None) else f"step{int(step):06d}.safetensors"
        checkpoint = os.path.join(run_dir, "checkpoints", name)
        if not os.path.isfile(checkpoint) and step:
            matches = glob.glob(os.path.join(run_dir, "checkpoints", f"step{int(step):06d}*.safetensors"))
            checkpoint = next((m for m in matches if ".state." not in m), "")
    if not os.path.isfile(checkpoint):
        print(f"checkpoint not found: {checkpoint}")
        return 2
    if not step:
        base = os.path.basename(checkpoint)
        step = 0 if base.startswith("final") else int(base[4:10])
    trainer = Trainer(config, log=print).load()
    trainer.out_dir = getattr(args, "samples_dir", "") or run_dir
    if getattr(args, "base", False):
        # step 0 baseline: zero every adapter so the model runs without a LoRA
        import torch

        with torch.no_grad():
            for name, param in trainer.model.named_parameters():
                if "lora" in name.lower():
                    param.zero_()
        trainer.preview(0)
        trainer.make_contact_sheet()
        return 0
    trainer._load_checkpoint(checkpoint)
    trainer.preview(step)
    trainer.make_contact_sheet()
    return 0


def cmd_baseline(args):
    """Sample the base model (no LoRA) on the workspace prompts.

    The dashboard shows these as the "no LoRA" row so a run can be compared
    against the model it started from.
    """
    import torch
    from safetensors.torch import load_file

    from .sampling import sample as run_sample
    from .trainer import Trainer
    from .weights import LatentCodec, load_vae

    config = _load(args)
    embeddings = os.path.join(config.cache.dir, "preview.safetensors")
    if not os.path.exists(embeddings):
        print(f"no preview embeddings at {embeddings}")
        return 2
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    out_dir = args.out or os.path.join(workspace, "baseline")
    os.makedirs(out_dir, exist_ok=True)

    trainer = Trainer(config, log=print).load()
    trainer.model.eval()
    data = load_file(embeddings)
    codec = LatentCodec(load_vae(config.model.vae, dtype=torch.float32, device="cuda"))
    images = run_sample(
        trainer.model, codec,
        data["embeds"].cuda().to(torch.bfloat16), data["mask"].cuda(),
        data["negative_embeds"].cuda().to(torch.bfloat16), data["negative_mask"].cuda(),
        height=config.preview.height, width=config.preview.width,
        steps=config.preview.steps, guidance=config.preview.guidance, shift=config.preview.shift,
        generator=torch.Generator("cpu").manual_seed(config.preview.seed),
        uncond_skip_layers=config.preview.uncond_skip_layers,
    )
    from PIL import Image

    for index, image in enumerate(images):
        pixels = ((image.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        path = os.path.join(out_dir, f"baseline_{index}.png")
        Image.fromarray(pixels).save(path)
        print(path)
    return 0


def cmd_export(args):
    import glob
    import os
    import shutil

    from .lora_file import convert

    config = _load(args)
    run_dir = os.path.join(config.output_dir, config.name, "checkpoints")
    checkpoint = args.checkpoint
    if checkpoint in ("final", "last", "0"):
        checkpoint = os.path.join(run_dir, "final.safetensors")
        step = 0
    elif checkpoint and not os.path.exists(checkpoint):
        matches = glob.glob(os.path.join(run_dir, f"step{int(checkpoint):06d}.safetensors"))
        if not matches:
            print(f"checkpoint not found: {checkpoint}")
            return 2
        checkpoint = matches[0]
    if not checkpoint:
        checkpoint = os.path.join(run_dir, "final.safetensors")
    if not os.path.isfile(checkpoint):
        print(f"checkpoint not found: {checkpoint}")
        return 2
    step = 0 if os.path.basename(checkpoint) == "final.safetensors" else int(os.path.basename(checkpoint)[4:10])

    out_dir = os.path.abspath(os.path.expanduser(args.out or os.path.dirname(checkpoint)))
    os.makedirs(out_dir, exist_ok=True)
    formats = [f.strip() for f in (args.formats or "kohya").split(",") if f.strip()]
    suffix = args.suffix if args.suffix is not None else ("" if not step else f"_step{step:06d}")
    if step == 0 and args.suffix is None:
        suffix = "_final"

    if formats == ["kohya"]:
        # checkpoints already are kohya LoRA files: export is a copy
        target = os.path.join(out_dir, f"{config.name}{suffix}.safetensors")
        if os.path.abspath(checkpoint) != os.path.abspath(target):
            shutil.copyfile(checkpoint, target)
        written = {"kohya": target}
    else:
        written = convert(checkpoint, config.name, formats, suffix=suffix, out_dir=out_dir,
                          alpha=float(config.lora.alpha), step=step)
    for kind, path in written.items():
        print(f"{kind}: {path}")

    if args.state:
        source = os.path.join(run_dir, "state.safetensors")
        if os.path.isfile(source):
            target = os.path.join(out_dir, f"{config.name}{suffix}_state.safetensors")
            if os.path.abspath(source) != os.path.abspath(target):
                shutil.copyfile(source, target)
            print(f"state: {target}")
        else:
            print(f"state: no training state for this run ({run_dir})")
    return 0


def cmd_export_batch(args):
    from .export_batch import export_batch, parse_range

    config = _load(args)
    run_dir = args.run_dir or os.path.join(config.output_dir, config.name)
    steps = parse_range(args.steps, os.path.join(run_dir, "checkpoints"))
    formats = [f.strip() for f in (args.formats or "kohya").split(",") if f.strip()]
    result = export_batch(
        args.config, run_dir, steps=steps, formats=formats, state=args.state,
        out_dir=args.out, zip_path=args.zip,
    )
    print(f"exported {len(result['files'])} file(s) -> {result['out_dir']}")
    if result["zip"]:
        print(f"zip: {result['zip']}")
    return 0


def cmd_queue(args):
    from .queue import run_pipeline

    return run_pipeline(args.workspace)


def cmd_train(args):
    if getattr(args, "smoke", False):
        # a pre-flight that touches every code path a long run will hit
        args.steps = 3
        args.sample_every = 1
        args.save_every = 3
        args.eval_every = 1
        args.name = (args.name or "smoke") + "-smoke"
    from .trainer import Trainer

    config = _load(args)
    overrides = {
        "steps": args.steps, "lr": args.lr, "batch_size": args.batch,
        "grad_accum": args.grad_accum, "save_every": args.save_every,
        "sample_every": args.sample_every,
        "eval_every": getattr(args, "eval_every", None),
        "save_state": getattr(args, "save_state", None),
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config.train, key, value)
    if args.name:
        config.name = args.name
    if args.rank is not None:
        config.lora.rank = args.rank
        config.lora.alpha = args.alpha if args.alpha is not None else args.rank
    if args.grad_checkpointing is not None:
        config.vram.gradient_checkpointing = args.grad_checkpointing
    trainer = Trainer(config).load()
    trainer.train(resume=args.resume or "")
    return 0


def cmd_sample(args):
    import torch
    from safetensors.torch import load_file

    from .sampling import sample as run_sample
    from .trainer import Trainer

    config = _load(args)
    if args.width:
        config.preview.width = args.width
    if args.height:
        config.preview.height = args.height
    if args.steps:
        config.preview.steps = args.steps
    embeddings_path = args.embeds or os.path.join(config.cache.dir, "preview.safetensors")
    if not os.path.exists(embeddings_path):
        print(f"no embeddings at {embeddings_path}; run `python -m moss_lora.cli encode-prompt` first")
        return 2

    trainer = Trainer(config).load()
    if args.checkpoint and not args.base:
        state = load_file(args.checkpoint)
        state = {k: v for k, v in state.items() if not k.startswith("ema.") and k != "step"}
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        print(f"loaded {args.checkpoint}: missing {len(missing)} unexpected {len(unexpected)}")
    elif args.base:
        print("base model, no LoRA")

    trainer.model.eval()
    data = load_file(embeddings_path)
    from .weights import LatentCodec, load_vae

    codec = LatentCodec(load_vae(config.model.vae, dtype=torch.float32, device="cuda"))
    images = run_sample(
        trainer.model, codec,
        data["embeds"].cuda().to(torch.bfloat16), data["mask"].cuda(),
        data["negative_embeds"].cuda().to(torch.bfloat16), data["negative_mask"].cuda(),
        height=config.preview.height, width=config.preview.width,
        steps=config.preview.steps, guidance=config.preview.guidance, shift=config.preview.shift,
        generator=torch.Generator("cpu").manual_seed(config.preview.seed),
    )
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    tag = "base" if args.base else (os.path.basename(args.checkpoint).split(".")[0] if args.checkpoint else "lora")
    for index, image in enumerate(images):
        pixels = ((image.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        path = os.path.join(args.out, f"{tag}_{index:02d}.png")
        Image.fromarray(pixels).save(path)
        print(path)
    return 0


# reference measurements at 1024^2 / 4096 image tokens, batch 1, rank 32;
# interpolated rows are marked and get replaced by `plan --measure`
MEMORY_TABLE = [  # precision, quantize, swap, peak GiB, s/step (1024^2, batch 1, rank 32, compile on)
    ("fp16acc", "none", 0, 12.7, 0.79),
    ("fp16acc", "ffn", 0, 8.4, 0.82),
    ("fp16acc", "all", 0, 6.2, 0.84),
    ("bf16", "none", 0, 12.7, 0.96),
    ("bf16", "none", 30, 6.5, 1.76),
    ("bf16", "none", 40, 4.4, 1.90),  # estimated from the eager 2.10
]

def cmd_plan(args):
    vram = args.vram or 0
    if not vram:
        from .config import detect_vram_gb

        vram = round(detect_vram_gb() or 32)
    reserve = 2.0 if args.preview else 1.0
    budget = vram - reserve
    fitting = [row for row in MEMORY_TABLE if row[3] <= budget]
    # among the rows that fit, the fastest wins
    choice = min(fitting, key=lambda row: row[4]) if fitting else MEMORY_TABLE[-1]
    print(f"target VRAM {vram:.0f} GiB (reserve {reserve:.0f} GiB for VAE/preview) -> budget {budget:.1f} GiB\n")
    print(f"{'precision':>9} {'quantize':>9} {'swap':>5} {'peak':>7} {'s/step':>8}")
    for prec, quant, swap, peak, speed in MEMORY_TABLE:
        mark = "  <- chosen" if (prec, quant, swap, peak, speed) == choice else ""
        print(f"{prec:>9} {quant:>9} {swap:5d} {peak:6.1f}G {speed:7.2f}s{mark}")
    print(f"\nrecommended: dtype={'float16' if choice[0] == 'fp16acc' else 'bfloat16'}, "
          f"fp16_accumulate={choice[0] == 'fp16acc'}, quantize_dit={choice[1]}, "
          f"blocks_to_swap={choice[2]}  (peak ~{choice[3]} GiB, ~{choice[4]} s/step)")
    if args.write:
        from .config import Config

        config = Config.load(args.config).resolve_paths()
        config.vram.blocks_to_swap = choice[2]
        config.vram.quantize_dit = choice[1]
        config.model.dtype = "float16" if choice[0] == "fp16acc" else "bfloat16"
        config.vram.fp16_accumulate = choice[0] == "fp16acc"
        config.vram.gradient_checkpointing = True
        config.save(args.write)
        print(f"written: {args.write}")
    return 0


def cmd_web(args):
    from .web.app import main as web_main

    sys.argv = ["moss-lora-web", "--host", args.host, "--port", str(args.port)]
    web_main()
    return 0


def cmd_info(args):
    from .data import assign_buckets, scan, summarise

    config = _load(args)
    items = assign_buckets(scan(config.dataset.path,
                                moss_caption_source=config.dataset.moss_caption_source),
                           area=config.dataset.max_area)
    info = summarise(items)
    print(f"images: {info['images']}  with caption: {info['with_caption']}  "
          f"styles: {info['styles']}  source: {config.dataset.moss_caption_source}")
    for bucket, count in info["buckets"].items():
        print(f"  {bucket:>10s}  {count}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="moss-lora", description="MOSS-Image LoRA trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    common = dict()
    p_cache = sub.add_parser("cache", help="build the latent + text embedding cache")
    p_cache.add_argument("--config")
    p_cache.add_argument("--dataset", help="dataset directory (relative to the current directory)")
    p_cache.add_argument("--cache-dir", dest="cache_dir", help="cache directory (relative to the current directory)")
    p_cache.add_argument("--preset")
    p_cache.add_argument("--latents-only", action="store_true")
    p_cache.add_argument("--text-only", action="store_true")
    p_cache.add_argument("--batch-size", dest="cache_batch", type=int, default=None,
                         help="latent encode batch (default: 80%% of free VRAM, clamped to 4)")
    p_cache.add_argument("--workers", type=int, default=6, help="image loading threads")
    p_cache.set_defaults(func=cmd_cache)

    p_info = sub.add_parser("info", help="describe a dataset")
    p_info.add_argument("--config")
    p_info.add_argument("--dataset", help="dataset directory (relative to the current directory)")
    p_info.add_argument("--preset")
    p_info.set_defaults(func=cmd_info)

    p_encode_text = sub.add_parser("encode-text", help="encode captions into the cache (local Qwen3.5-9B)")
    p_encode_text.add_argument("--config", required=True)
    p_encode_text.add_argument("--overwrite", action="store_true")
    p_encode_text.add_argument("--text-model", default="",
                               help="Qwen3.5-9B dir (defaults to model.text_encoder)")
    p_encode_text.add_argument("--quantize", choices=("none", "fp8", "nf4_ffn", "nf4"), default="none")
    p_encode_text.add_argument("--preview-only", action="store_true",
                               help="only refresh preview.safetensors from preview.prompts")
    p_encode_text.add_argument("--batch-size", type=int, default=None,
                               help="prompt batch (default: 80%% of free VRAM)")
    p_encode_text.set_defaults(func=cmd_encode_text)

    p_encode_prompt = sub.add_parser("encode-prompt", help="encode arbitrary prompts into a bundle")
    p_encode_prompt.add_argument("--config", required=True)
    p_encode_prompt.add_argument("--out", required=True)
    p_encode_prompt.add_argument("--prompt", action="append", required=True)
    p_encode_prompt.add_argument("--negative", default="")
    p_encode_prompt.add_argument("--text-model", default="")
    p_encode_prompt.set_defaults(func=cmd_encode_prompt)

    p_caption = sub.add_parser("caption", help="generate MOSS-Image captions (sidecar JSON next to the images)")
    p_caption.add_argument("--config", default="")
    p_caption.add_argument("--images", default="", help="image directory (default: dataset.path)")
    p_caption.add_argument("--out", default="", help="sidecar directory (default: next to each image)")
    p_caption.add_argument("--styles", default="", help="comma list: tag_caption / official (default: both)")
    p_caption.add_argument("--overwrite", action="store_true", help="regenerate styles that already exist")
    p_caption.add_argument("--migrate", action="store_true", help="upgrade old flat sidecars to schema 2")
    p_caption.add_argument("--role", default="train", choices=("train", "reg"), help=argparse.SUPPRESS)
    p_caption.add_argument("--base-url", default="", help="OpenAI-compatible endpoint")
    p_caption.add_argument("--api-key", default="", help="bearer token, or the name of an env var holding it")
    p_caption.add_argument("--model", default="", help="model id (default: first the server reports)")
    p_caption.add_argument("--trigger", default=None)
    p_caption.add_argument("--prompt-tags", default="")
    p_caption.add_argument("--prompt-natural", default="")
    p_caption.add_argument("--no-thinking", action="store_true")
    p_caption.add_argument("--temperature", type=float, default=None)
    p_caption.add_argument("--top-p", type=float, default=None)
    p_caption.add_argument("--max-tokens", type=int, default=None)
    p_caption.add_argument("--workers", type=int, default=None)
    p_caption.add_argument("--retries", type=int, default=None)
    p_caption.add_argument("--structured", default="", choices=["", "auto", "on", "off"])
    p_caption.add_argument("--limit", type=int, default=0)
    p_caption.add_argument("--dry-run", action="store_true")
    p_caption.add_argument("--validate-only", action="store_true")
    p_caption.set_defaults(func=cmd_caption)

    p_preview = sub.add_parser("preview", help="re-render one checkpoint's preview images")
    p_preview.add_argument("--config", required=True)
    p_preview.add_argument("--checkpoint", default="")
    p_preview.add_argument("--step", type=int, default=0)
    p_preview.add_argument("--samples-dir", default="",
                           help="write samples here instead of <run>/samples")
    p_preview.add_argument("--base", action="store_true",
                           help="render the base model (adapter zeroed) instead of a checkpoint")
    p_preview.set_defaults(func=cmd_preview)

    p_baseline = sub.add_parser("baseline", help="sample the base model for comparison")
    p_baseline.add_argument("--config", required=True)
    p_baseline.add_argument("--out", default="")
    p_baseline.set_defaults(func=cmd_baseline)

    p_export = sub.add_parser("export", help="export a checkpoint (or the final weights) as LoRA")
    p_export.add_argument("--config", required=True)
    p_export.add_argument("--checkpoint", default="", help="path, or a step number like 1000")
    p_export.add_argument("--out", default="", help="output directory (defaults to the run dir)")
    p_export.add_argument("--formats", default="", help="comma list: kohya,diffusers")
    p_export.add_argument("--suffix", default=None, help="filename suffix, e.g. _step001000")
    p_export.add_argument("--state", action="store_true",
                          help="also copy the checkpoint (LoRA + optimizer/RNG) next to the export")
    p_export.set_defaults(func=cmd_export)

    p_batch = sub.add_parser("export-batch", help="export many checkpoints of one run at once")
    p_batch.add_argument("--config", required=True)
    p_batch.add_argument("--run-dir", default="", help="run directory (defaults to output_dir/name)")
    p_batch.add_argument("--steps", default="", help="200-2000:200, a comma list, or empty for all")
    p_batch.add_argument("--formats", default="kohya")
    p_batch.add_argument("--out", default="", help="target directory (absolute is fine)")
    p_batch.add_argument("--zip", default="", help="also write a zip to this path")
    p_batch.add_argument("--state", action="store_true")
    p_batch.set_defaults(func=cmd_export_batch)

    p_queue = sub.add_parser("queue", help="resume every unfinished run in a workspace")
    p_queue.add_argument("--workspace", required=True)
    p_queue.set_defaults(func=cmd_queue)

    p_train = sub.add_parser("train", help="train a LoRA on a built cache")
    p_train.add_argument("--config")
    p_train.add_argument("--dataset", help="dataset directory (relative to the current directory)")
    p_train.add_argument("--cache-dir", dest="cache_dir", help="cache directory (relative to the current directory)")
    p_train.add_argument("--output-dir", dest="output_dir", help="run output directory (relative to the current directory)")
    p_train.add_argument("--preset")
    p_train.add_argument("--steps", type=int)
    p_train.add_argument("--name", help="output name (avoid clobbering a real run with a smoke test)")
    p_train.add_argument("--rank", type=int)
    p_train.add_argument("--alpha", type=int)
    p_train.add_argument("--lr", type=float)
    p_train.add_argument("--batch", type=int)
    p_train.add_argument("--grad-accum", dest="grad_accum", type=int)
    p_train.add_argument("--save-every", dest="save_every", type=int)
    p_train.add_argument("--sample-every", dest="sample_every", type=int)
    p_train.add_argument("--eval-every", dest="eval_every", type=int)
    p_train.add_argument("--save-state", dest="save_state", default=None, nargs="?",
                         const="last", choices=["last", "all"],
                         help="save optimizer/RNG: last (rolling file) or all (one per step)")
    p_train.add_argument("--no-save-state", dest="save_state", action="store_const", const="off",
                         help="do not save optimizer state at all")
    p_train.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true", default=None)
    p_train.add_argument("--smoke", action="store_true",
                           help="3-step pre-flight: touches every code path a long run hits")
    p_train.add_argument("--resume")
    p_train.set_defaults(func=cmd_train)

    p_plan = sub.add_parser("plan", help="pick a memory/time trade-off for this GPU")
    p_plan.add_argument("--vram", type=float, help="target VRAM in GiB (default: detect)")
    p_plan.add_argument("--config", default="runs/default.yaml")
    p_plan.add_argument("--write", help="write the chosen settings into this yaml")
    p_plan.add_argument("--preview", action="store_true", help="reserve room for sample decoding")
    p_plan.set_defaults(func=cmd_plan)

    p_web = sub.add_parser("web", help="run the web UI")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=7861)
    p_web.set_defaults(func=cmd_web)

    p_sample = sub.add_parser("sample", help="sample with a checkpoint (or the base model)")
    p_sample.add_argument("--config")
    p_sample.add_argument("--checkpoint")
    p_sample.add_argument("--embeds")
    p_sample.add_argument("--out", default="output/samples")
    p_sample.add_argument("--base", action="store_true", help="skip the LoRA (baseline)")
    p_sample.add_argument("--width", type=int)
    p_sample.add_argument("--height", type=int)
    p_sample.add_argument("--steps", type=int)
    p_sample.set_defaults(func=cmd_sample)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # 分配器选项必须在任何 CUDA 分配之前生效；settings.yaml 里可切换
    # expandable（默认）/ async / off 三种机制。文本编码固定用默认分配器
    # （expandable_segments 下加载 9B 权重会报 GPU mapping 失败）。
    from .settings import configure_allocator

    kind = str(getattr(args, "command", "") or "").strip()
    if not kind:
        kind = {
            "cmd_encode_text": "encode-text",
            "cmd_encode_prompt": "encode-prompt",
        }.get(getattr(args.func, "__name__", ""), "")
    configure_allocator(kind=kind)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
