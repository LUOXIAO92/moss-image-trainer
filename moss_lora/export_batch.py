"""Batch-export checkpoints of one run: load the 5B model once, walk the steps.

    python -m moss_lora.cli export-batch --config <run.yaml> --run-dir <dir> \
        --steps 200-2000:200 --formats kohya --out /abs/path [--state] [--zip out.zip]
"""

from __future__ import annotations

import os
import shutil
import zipfile

from .config import Config


FINAL = 0  # sentinel: the "final.safetensors" checkpoint


def available_steps(checkpoint_dir: str) -> list[int]:
    steps = []
    if not os.path.isdir(checkpoint_dir):
        return steps
    for name in os.listdir(checkpoint_dir):
        if name.startswith("step") and name.endswith(".safetensors"):
            try:
                steps.append(int(name[4:10]))
            except ValueError:
                continue
    steps.sort()
    if os.path.isfile(os.path.join(checkpoint_dir, "final.safetensors")):
        steps.append(FINAL)
    return steps


def parse_range(spec: str, checkpoint_dir: str) -> list[int]:
    """``200-2000:200`` -> [200, 400, ...]; a bare list is also accepted."""
    steps = available_steps(checkpoint_dir)
    if not spec:
        return steps
    numeric = [s for s in steps if s != FINAL]
    if spec.strip().lower() == "final":
        return [FINAL] if FINAL in steps else []
    if "," in spec:
        wanted = {int(part) for part in spec.split(",") if part.strip()}
        return [step for step in steps if step in wanted]
    if "-" not in spec and ":" not in spec:
        # a bare number means exactly that step, not "from here to the end"
        value = int(spec)
        return [value] if value in steps else []
    start, _, rest = spec.partition("-")
    end, _, every = rest.partition(":")
    lo = int(start) if start.strip() else (numeric[0] if numeric else 0)
    hi = int(end) if end.strip() else (numeric[-1] if numeric else 0)
    step = int(every) if every.strip() else 1
    return [value for value in numeric if lo <= value <= hi and (value - lo) % step == 0]


def export_batch(
    config_path: str,
    run_dir: str,
    steps: list[int] | None = None,
    formats: list[str] | None = None,
    state: bool = False,
    out_dir: str = "",
    zip_path: str = "",
    log=print,
) -> dict:
    from .lora_file import convert, is_kohya, kohya_key
    from safetensors.torch import load_file, save_file

    config = Config.load(config_path).resolve_paths()
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    selected = steps or available_steps(checkpoint_dir)
    if not selected:
        raise SystemExit(f"no checkpoints under {checkpoint_dir}")

    target = os.path.abspath(os.path.expanduser(out_dir or os.path.join(run_dir, "exports")))
    os.makedirs(target, exist_ok=True)
    formats = formats or ["kohya"]
    alpha = float(config.lora.alpha)

    written: list[str] = []
    for step in selected:
        if step == FINAL:
            path = os.path.join(checkpoint_dir, "final.safetensors")
            suffix = "_final"
        else:
            path = os.path.join(checkpoint_dir, f"step{step:06d}.safetensors")
            suffix = f"_step{step:06d}"
        if not os.path.isfile(path):
            log(f"skip {suffix.strip('_')}: no checkpoint")
            continue
        files = {}
        if formats == ["kohya"]:
            # checkpoints already are kohya LoRA files: export is a plain copy
            dest = os.path.join(target, f"{config.name}{suffix}.safetensors")
            if os.path.abspath(dest) != os.path.abspath(path):
                shutil.copyfile(path, dest)
            files["kohya"] = dest
        else:
            # plain key renaming; the 5B model is never instantiated here
            files = convert(path, config.name, formats, suffix=suffix, out_dir=target,
                            alpha=alpha, step=step)
        written.extend(files.values())
        log(f"exported {suffix.strip('_')} -> {len(files)} file(s)")

    if state:
        source = os.path.join(checkpoint_dir, "state.safetensors")
        if os.path.isfile(source):
            copy = os.path.join(target, f"{config.name}_state.safetensors")
            if os.path.abspath(source) != os.path.abspath(copy):
                shutil.copyfile(source, copy)
            written.append(copy)
            log(f"state -> {copy}")
        else:
            log("state: run has no state.safetensors")

    if zip_path:
        os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
            for path in written:
                archive.write(path, arcname=os.path.basename(path))
        log(f"zip -> {zip_path}")
    return {"out_dir": target, "files": written, "zip": zip_path}
