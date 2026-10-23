"""Configuration objects, presets and YAML (de)serialisation.

Expressions like ``${MODELS}/...`` are expanded only when the ``MODELS``
environment variable is set.  An unresolved expression is kept verbatim so
validation can report it clearly.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

from .problems import Problem, problem

DEFAULT_MODELS_ROOT = ""
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def models_root() -> str:
    return os.environ.get("MODELS", DEFAULT_MODELS_ROOT)


def expand(value):
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            replacement = os.environ.get(name)
            if name == "MODELS" and not replacement:
                replacement = models_root()
            # Leave ${VAR} untouched when no value is configured.  Turning it
            # into "/..." would silently point at the filesystem root.
            return replacement if replacement else match.group(0)

        value = _ENV_PATTERN.sub(replace, value)
        return os.path.expanduser(value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


@dataclass
class ModelConfig:
    # all three are empty by default: the user points them at their own files
    dit: str = ""
    # local HuggingFace Qwen3.5-9B directory used for text encoding
    text_encoder: str = ""
    # 当 text_encoder 指向单个 .safetensors（ConvRot int8 量化权重）时，配置文件从这里取
    text_encoder_config: str = ""
    vae: str = ""
    dtype: str = "bfloat16"


@dataclass
class DatasetConfig:
    path: str = ""
    caption_ext: list = field(default_factory=lambda: [".txt", ".caption"])
    metadata: str = ""          # optional metadata.jsonl
    # 每张图按自己的宽高比缩放到这个像素面积上限（不放大），两边对齐 16。
    # 没有固定比例集合：DiT 是自注意力，batch 不需要共用画布。
    max_area: int = 1024 * 1024
    # 重复次数只来自目录名 "<n>_<name>"，不再有全局倍数
    # 正则数据集：只提供"长什么样但不是这个角色"的图，防止 LoRA 把角色概念扩散到别的提示词
    reg_path: str = ""
    reg_repeat: int = 1
    reg_caption: str = ""
    caption_prefix: str = ""
    caption_suffix: str = ""
    # <name>_moss-image-trainer.json 里优先取哪个字段（caption 已是合成好的一整行）
    moss_caption_source: str = "tag_caption"   # tag_caption | official_long|medium|short


@dataclass
class CacheConfig:
    dir: str = ""
    # 正则集的缓存单独放（留空 = <dir>/reg）；训练时两边的逐条文件一起读
    reg_dir: str = ""
    text_max_length: int = 4096
    text_dtype: str = "bfloat16"
    latent_dtype: str = "float16"
    latent_sample: bool = False
    overwrite: bool = False
    # 0 = pick from free VRAM. A fixed number keeps re-encodes comparable: the
    # same caption comes back a few 1e-4 apart at batch 1 and at batch 64.
    text_batch: int = 0
    vae_batch: int = 0


@dataclass
class LoraConfig:
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    targets: list = field(default_factory=lambda: ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"])
    # empty = all DiT blocks; a list restricts the regex to these layer indices
    layers: list = field(default_factory=list)
    include_text_embedder: bool = False
    init: str = "kaiming"       # kaiming | zero
    bias: str = "none"


@dataclass
class SigmaConfig:
    mode: str = "logit_normal"  # logit_normal | uniform | shifted
    mu: float = -1.09
    sigma: float = 0.95
    cutoff: float = 0.95
    uniform_mix: float = 0.01
    shift: float = 1.0


@dataclass
class ExportConfig:
    # one canonical file by default (kohya naming = what ComfyUI loads);
    # other layouts are generated on demand by `cli export --formats ...`
    formats: list = field(default_factory=lambda: ["kohya"])


@dataclass
class TrainConfig:
    steps: int = 1000
    epochs: int = 0             # 0 = use steps
    batch_size: int = 1
    grad_accum: int = 1
    # 取样与前向：都按 token 预算取若干张，尺寸可以各不相同
    #   auto   = 装了 FlashAttention-2 就 packed，否则 padded
    #   packed = 拼成一条序列，注意力走 varlen（FA-2）或块对角 mask
    #   padded = 在 token 维补齐到 L_max + key-padding mask（SDPA）
    batch_mode: str = "auto"
    # 注意力后端：auto = 装了 FlashAttention-2 就用，否则回退 SDPA；flash / sdpa 为强制
    attention: str = "auto"
    # >0 时每步按 token 预算装箱（装到预算为止）；0 = 用 batch_size 的张数
    bin_tokens: int = 0
    lr: float = 1e-4
    optimizer: str = "adamw"    # adamw | adamw8bit | muon | adafactor
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # 步数（int）或比例（0<w<1，按总步数换算）
    warmup: float = 50
    lr_scheduler: str = "constant"  # constant | cosine
    sigma: SigmaConfig = field(default_factory=SigmaConfig)
    caption_dropout: float = 0.05
    seed: int = 42
    save_every: int = 250
    sample_every: int = 250
    eval_every: int = 100       # 0 disables the held-out loss
    eval_batches: int = 8
    eval_holdout: float = 0.05  # fraction of the dataset reserved for eval
    ema: float = 0.0            # 0 disables
    # 训练状态保存策略：off / last（滚动一份）/ all（每步一份）
    save_state: str = "last"
    log_every: int = 1


@dataclass
class VramConfig:
    preset: str = "auto"        # auto | 32 | 24 | 16 | 12 | 8 | 6
    gradient_checkpointing: str = "auto"
    blocks_to_swap: int = -1    # -1 = from preset
    optimizer_in_8bit: bool = False
    # weight-only quantization of the frozen DiT: none (bf16) | ffn | all (nf4)
    quantize_dit: str = "none"
    # fp16 only: let cuBLAS accumulate in fp16 (2x the bf16 tensor-core rate, less precise)
    fp16_accumulate: bool = False
    compile: bool = True   # torch.compile the block loop (~25% faster, dynamic shapes)


@dataclass
class PreviewConfig:
    prompts: list = field(default_factory=list)
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 50
    guidance: float = 3.0
    shift: float = 5.0
    uncond_skip_layers: list = field(default_factory=lambda: [22])
    seed: int = 42


@dataclass
class Config:
    name: str = "moss-lora"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    vram: VramConfig = field(default_factory=VramConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    # 与文档、WebUI 生成的新工作区以及 runs/ 现有产物保持一致
    output_dir: str = "runs"

    # -- conversion ------------------------------------------------------
    def to_dict(self) -> dict:
        return expand(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: str) -> "Config":
        import yaml

        with open(path) as f:
            config = cls.from_dict(yaml.safe_load(f) or {})
        # 运行目录里的配置把 cache/output 存成相对路径，整个工作区目录可以随便搬；
        # 其它位置的配置按进程 cwd 解析。
        absolute = os.path.abspath(path)
        parent = os.path.dirname(absolute)
        if os.path.basename(os.path.dirname(parent)) == "runs":
            # <workspaces>/<name>/runs/<run>/config.yaml -> resolve against <name>
            config._base_dir = os.path.dirname(os.path.dirname(parent))
        elif os.path.basename(parent) == "logs":
            # <workspaces>/<name>/logs/<job>.yaml, the snapshot a job runs from
            config._base_dir = os.path.dirname(parent)
        return config.resolve_paths()

    def save(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)

    def resolve_paths(self) -> "Config":
        """Expand ``${MODELS}``/``~`` once, in place.

        A config loaded from a run directory resolves relative cache/output/dataset
        paths against the workspace; one loaded from anywhere else keeps the
        process cwd behaviour.
        """
        base = str(getattr(self, "_base_dir", "") or "")

        def local(value):
            value = expand(value)
            if base and value and not os.path.isabs(value):
                value = os.path.join(base, value)
            return os.path.expanduser(value)

        self.model.dit = expand(self.model.dit)
        self.model.text_encoder = expand(self.model.text_encoder)
        self.model.vae = expand(self.model.vae)
        self.dataset.path = local(self.dataset.path)
        self.dataset.metadata = local(self.dataset.metadata)
        self.dataset.reg_path = local(self.dataset.reg_path)
        # 留空 = 默认目录：工作区里是 <ws>/train_cache 和 <ws>/runs，
        # 没有工作区基准时退化成当前目录下的 train_cache / runs
        self.cache.dir = local(self.cache.dir) or (os.path.join(base, "train_cache") if base else "train_cache")
        self.cache.reg_dir = local(self.cache.reg_dir)
        self.output_dir = local(self.output_dir) or (os.path.join(base, "runs") if base else "runs")
        return self

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def validate(self) -> list["Problem"]:
        """Path/parameter problems. An empty ``text_encoder`` means the text
        embeddings were produced elsewhere and the trainer only reads the cache.

        Each entry is a code plus parameters, not a sentence: the CLI renders it
        in Chinese, the WebUI translates the same code, and neither has to parse
        the other's text.
        """
        problems: list[Problem] = []
        if not self.dataset.path:
            problems.append(problem("err.config.dataset_path_empty"))
        elif not os.path.isdir(self.dataset.path):
            problems.append(problem("err.dataset.missing", path=self.dataset.path))
        for key in ("dit", "vae"):
            path = expand(getattr(self.model, key))
            field = f"model.{key}"
            if not path:
                problems.append(problem("err.config.field_empty", field=field))
            elif "${" in path:
                problems.append(problem("err.config.field_variable", field=field, path=path))
            elif not os.path.exists(path):
                problems.append(problem("err.config.field_missing", field=field, path=path))
        text_encoder = expand(self.model.text_encoder)
        if text_encoder and "${" in text_encoder:
            problems.append(problem("err.config.field_variable", field="model.text_encoder",
                                    path=text_encoder))
        elif text_encoder and not os.path.exists(text_encoder):
            problems.append(problem("err.config.field_missing", field="model.text_encoder",
                                    path=text_encoder))
        elif text_encoder and os.path.isdir(text_encoder):
            if not os.path.isfile(os.path.join(text_encoder, "config.json")):
                problems.append(problem("err.config.text_encoder_no_config", path=text_encoder))
        elif text_encoder and not text_encoder.lower().endswith(".safetensors"):
            problems.append(problem("err.config.text_encoder_kind", path=text_encoder))
        te_config = expand(self.model.text_encoder_config)
        if te_config and not os.path.isdir(te_config):
            problems.append(problem("err.config.te_config_missing", path=te_config))
        elif te_config and not os.path.isfile(os.path.join(te_config, "config.json")):
            problems.append(problem("err.config.te_config_no_config", path=te_config))
        if self.lora.rank <= 0:
            problems.append(problem("err.config.lora_rank"))
        for layer in self.lora.layers or []:
            try:
                index = int(layer)
            except (TypeError, ValueError):
                problems.append(problem("err.config.lora_layer_int", layer=layer))
                continue
            if index < 0 or index > 41:
                problems.append(problem("err.config.lora_layer_range", index=index))
        if self.train.batch_size <= 0 or self.train.grad_accum <= 0:
            problems.append(problem("err.config.batch_size"))
        for key in ("text_batch", "vae_batch"):
            value = getattr(self.cache, key, 0) or 0
            if int(value) < 0:
                problems.append(problem("err.config.cache_batch", field=f"cache.{key}"))
        source = str(getattr(self.dataset, "moss_caption_source", "tag_caption"))
        if source not in ("tag_caption", "official_long", "official_medium", "official_short"):
            problems.append(problem("err.config.caption_source", value=source))
        if str(self.train.attention).lower() not in ("auto", "flash", "sdpa"):
            problems.append(problem("err.config.attention", value=self.train.attention))
        if str(self.train.batch_mode).lower() not in ("auto", "packed", "padded"):
            problems.append(problem("err.config.batch_mode", value=self.train.batch_mode))
        if int(self.train.bin_tokens or 0) < 0:
            problems.append(problem("err.config.bin_tokens"))
        if str(self.train.optimizer).lower() not in ("adamw", "adamw8bit", "muon", "adafactor"):
            problems.append(problem("err.config.optimizer", value=self.train.optimizer))
        if str(self.lora.init).lower() not in ("kaiming", "zero"):
            problems.append(problem("err.config.lora_init", value=self.lora.init))
        preview = self.preview
        for name in ("steps", "width", "height", "seed"):
            if not isinstance(getattr(preview, name), int):
                problems.append(problem("err.config.preview_int", field=name))
        for name in ("guidance", "shift"):
            if not isinstance(getattr(preview, name), (int, float)):
                problems.append(problem("err.config.preview_number", field=name))
        if isinstance(preview.steps, int) and preview.steps <= 0:
            problems.append(problem("err.config.preview_steps"))
        if isinstance(preview.width, int) and (preview.width <= 0 or preview.width % 16):
            problems.append(problem("err.config.preview_width"))
        if isinstance(preview.height, int) and (preview.height <= 0 or preview.height % 16):
            problems.append(problem("err.config.preview_height"))
        if isinstance(preview.guidance, (int, float)) and preview.guidance < 0:
            problems.append(problem("err.config.preview_guidance"))
        if isinstance(preview.shift, (int, float)) and preview.shift < 0:
            problems.append(problem("err.config.preview_shift"))
        if isinstance(preview.seed, int) and preview.seed < -1:
            problems.append(problem("err.config.preview_seed"))
        for layer in preview.uncond_skip_layers or []:
            try:
                index = int(layer)
            except (TypeError, ValueError):
                problems.append(problem("err.config.uncond_layer_int", layer=layer))
                continue
            if index < 0 or index > 41:
                problems.append(problem("err.config.uncond_layer_range", index=index))
        return problems

def _build(cls, data: dict):
    """Dataclass construction that tolerates unknown keys and nested dicts."""
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in known:
            continue
        if value is None:
            # a blank field in the UI means "use the default", not "set None"
            continue
        ftype = known[key].type
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        elif is_dataclass(known[key].default_factory if known[key].default_factory is not None else object) and isinstance(value, dict):
            kwargs[key] = _build(known[key].default_factory, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# VRAM presets: index is the card size in GB
# ---------------------------------------------------------------------------
VRAM_PRESETS: dict[str, dict[str, Any]] = {
    "32": {"gradient_checkpointing": True, "blocks_to_swap": 0, "optimizer_in_8bit": False, "quantize_dit": "none"},
    "24": {"gradient_checkpointing": True, "blocks_to_swap": 0, "optimizer_in_8bit": False, "quantize_dit": "none"},
    "16": {"gradient_checkpointing": True, "blocks_to_swap": 0, "optimizer_in_8bit": False, "quantize_dit": "none"},
    "12": {"gradient_checkpointing": True, "blocks_to_swap": 0, "optimizer_in_8bit": False, "quantize_dit": "ffn"},
    "8": {"gradient_checkpointing": True, "blocks_to_swap": 0, "optimizer_in_8bit": False, "quantize_dit": "all"},
    "6": {"gradient_checkpointing": True, "blocks_to_swap": 20, "optimizer_in_8bit": True, "quantize_dit": "all"},
}


def apply_preset(config: Config, vram_gb: float | None = None) -> Config:
    """Fill ``vram`` defaults from the card size unless the user overrode them."""
    if config.vram.preset != "auto":
        key = str(config.vram.preset).replace("GB", "")
        preset = VRAM_PRESETS.get(key, {})
        for field_name, value in preset.items():
            setattr(config.vram, field_name, value)
        return config
    if vram_gb is None:
        vram_gb = detect_vram_gb()
    if vram_gb is None:
        return config
    for size in ("32", "24", "16", "12", "8", "6"):
        if vram_gb >= int(size) - 1:
            preset = VRAM_PRESETS[size]
            break
    else:
        preset = VRAM_PRESETS["6"]
    if config.vram.gradient_checkpointing == "auto":
        config.vram.gradient_checkpointing = preset["gradient_checkpointing"]
    else:
        config.vram.gradient_checkpointing = str(config.vram.gradient_checkpointing).lower() in ("1", "true", "yes")
    if config.vram.blocks_to_swap < 0:
        config.vram.blocks_to_swap = preset["blocks_to_swap"]
    if not config.vram.optimizer_in_8bit:
        config.vram.optimizer_in_8bit = preset["optimizer_in_8bit"]
    if config.vram.quantize_dit == "none":
        config.vram.quantize_dit = preset["quantize_dit"]
    return config


def detect_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return None
