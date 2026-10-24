"""User-facing messages as codes, with the Chinese template as the source.

Every message the trainer or the WebUI can show is a ``Problem``: a code plus
the parameters it needs. The Chinese template lives here so the CLI can print a
sentence with no web layer involved, and the four i18n packs translate the same
code. :func:`render` fills a template; a missing parameter stays visible as
``{name}`` instead of vanishing.

    problem("err.run.not_found", run=name).render()
"""

from __future__ import annotations

import re
from typing import NamedTuple


class Problem(NamedTuple):
    """One thing wrong, in the form every surface can render."""

    code: str
    params: dict

    def render(self) -> str:
        return render(self.code, self.params)

    def detail(self) -> dict:
        """The payload an HTTP error body carries."""
        return {"code": self.code, "params": self.params, "message": self.render()}


def problem(code: str, **params) -> Problem:
    return Problem(code, params)


CODES: dict[str, str] = {
    "err.cache.dir_empty": "没有设置缓存目录，没法检查缓存",
    "err.cache.legacy": "{dir} 里的缓存是旧格式，训练集和正则集混在一起，去“缓存”页重新生成一次",
    "err.cache.missing_text": "{dir} 里的缓存缺了 {missing} 条文本编码（一共 {items} 条），去“缓存”页勾上“文本编码”补一遍",
    "err.cache.not_built": "{dir} 还没有生成缓存，先去“缓存”页生成一遍",
    "err.cache.parts_required": "至少要选一项：latent 缓存或文本编码",
    "err.cache.sizing": "{dir} 里的缓存是按另一个面积上限生成的，跟现在的设置对不上，去“缓存”页重新生成 latent 缓存",
    "err.cache.stop_dead": "那个缓存进程已经不在了",
    "err.cache.stop_denied": "没有权限停止这个进程",
    "err.cache.stop_finished": "这个缓存任务已经结束了",
    "err.cache.stop_foreign": "进程 {pid} 看起来不是缓存任务，没有向它发送停止信号",
    "err.cache.stop_no_job": "现在没有正在运行的缓存任务",
    "err.cache.text_encoder_required": "文本编码需要先在训练设置里填文本编码器（本地 Qwen3.5-9B 目录），填好再来生成",
    "err.caption.dataset_required": "先勾选要生成提示词的数据集：训练集或正则集",
    "err.caption.key_unreadable": "设置里存的是环境变量名 {name}，但当前程序读不到它的值：先导出这个变量并重启，或者直接把密钥填进“提示词生成”页",
    "err.caption.no_endpoint": "提示词生成还没填服务地址，到“提示词生成”页填好再开始",
    "err.caption.no_key": "提示词生成还没填密钥，到“提示词生成”页填好再开始",
    "err.caption.reg_missing": "还没有设置正则集目录，没法给正则集生成提示词",
    "err.caption.style_required": "先勾选要生成的风格：tag + caption 或官方描述",
    "err.config.attention": "注意力实现只有 auto、flash、sdpa，现在填的是 {value}",
    "err.config.batch_mode": "批次模式只有 auto、packed、padded，现在填的是 {value}",
    "err.config.batch_size": "batch_size 和 grad_accum 都要大于 0",
    "err.config.bin_tokens": "bin_tokens 不能小于 0（0 表示自动）",
    "err.config.cache_batch": "缓存批次的 {field} 要填 0（自动）或正整数",
    "err.config.caption_source": "提示词来源只有 tag_caption、official_long、official_medium、official_short 四种，现在填的是 {value}",
    "err.config.dataset_path_empty": "配置里的 dataset.path 是空的",
    "err.config.field_empty": "配置里的 {field} 没填",
    "err.config.field_missing": "找不到 {field} 指向的路径：{path}",
    "err.config.field_variable": "配置里的 {field} 还带着没展开的变量：{path}",
    "err.config.invalid": "配置有问题：{problems}",
    "err.config.lora_init": "LoRA 初始化只有 kaiming 或 zero，现在填的是 {value}",
    "err.config.lora_layer_int": "lora.layers 里有个值不是整数：{layer}",
    "err.config.lora_layer_range": "lora.layers 里的 {index} 超出 0..41",
    "err.config.lora_rank": "LoRA 的 rank 要大于 0",
    "err.config.optimizer": "优化器只有 adamw、adamw8bit、muon、adafactor，现在填的是 {value}",
    "err.config.blocks_to_swap": "常驻层数只能卸载 0–42 块，现在填的是 {value}",
    "err.config.preview_batch": "每次采样几条不能是负数",
    "err.config.preview_vae_device": "样图 VAE 只能解码到 gpu 或 cpu，现在填的是 {value}",
    "err.config.preview_vae_dtype": "样图 VAE 精度只有 fp32 或 fp16，现在填的是 {value}",
    "err.config.preview_guidance": "样图的引导强度不能小于 0",
    "err.config.preview_blocks_to_swap": "预览常驻层数只能是 -1（跟随训练）或卸载 0–42 块，现在填的是 {value}",
    "err.config.preview_height": "样图的高度要是 16 的正整数倍",
    "err.config.preview_int": "样图的 {field} 要填整数",
    "err.config.preview_number": "样图的 {field} 要填数字",
    "err.config.preview_seed": "样图的 seed 要么填 -1（每次随机），要么填非负整数",
    "err.config.preview_shift": "样图的 shift 不能小于 0",
    "err.config.preview_steps": "样图的步数要大于 0",
    "err.config.preview_width": "样图的宽度要是 16 的正整数倍",
    "err.config.quantize_conditioning": "conditioning 量化只有 none、fp8、nf4，现在填的是 {value}",
    "err.config.structure": "配置结构不对：{error}",
    "err.config.te_config_missing": "找不到文本编码器的 config.json 目录：{path}",
    "err.config.te_config_no_config": "那个目录里没有 config.json：{path}",
    "err.config.text_encoder_kind": "文本编码器要填 HF 目录或 .safetensors 文件：{path}",
    "err.config.text_encoder_no_config": "文本编码器目录里没有 config.json：{path}",
    "err.config.uncond_layer_int": "uncond_skip_layers 里有个值不是整数：{layer}",
    "err.config.uncond_layer_range": "uncond_skip_layers 里的 {index} 超出 0..41",
    "err.dataset.missing": "找不到训练集目录：{path}",
    "err.dataset.reg_path_missing": "找不到正则集目录：{path}",
    "err.dataset.reg_repeat": "正则集的重复次数要填 1 或更大的整数",
    "err.dataset.source_unknown": "不认识的提示词来源：{source}",
    "err.dir.not_found": "找不到这个目录：{path}",
    "err.export.checkpoint_missing": "找不到检查点 {file}",
    "err.export.format_required": "至少选一种导出格式",
    "err.export.step_invalid": "{step} 不是有效的步数",
    "err.file.not_found": "找不到这个文件",
    "err.gpu.busy": "GPU 正被另一个任务占着（进程 {pid}），等它结束再来",
    "err.job.busy": "已经有一个任务在跑了（编号 {job}），等它结束，或先到训练页停掉它",
    "err.job.kind_unknown": "不认识的任务类型：{kind}",
    "err.job.log_missing": "没有找到这个任务的日志文件",
    "err.job.not_found": "找不到任务 {job}",
    "err.job.not_running": "这个任务已经不在运行了",
    "err.job.stop_failed": "发不出停止信号：{error}",
    "err.job.yaml_required": "没有配置内容：传一份 YAML，或指定一个运行",
    "err.lang.unknown": "没有这个语言：{code}",
    "err.lang.code_invalid": "语言代码只能用小写字母、数字和短横线，2 到 8 位，比如 ru、pt-br",
    "err.lang.pack_empty": "语言包是空的：至少要有 $name 和一条 key",
    "err.layout.not_object": "布局数据格式不对",
    "err.name.charset.config": "配置名称只能使用字母、数字、点、下划线和短横线，长度 1 到 64 个字符",
    "err.name.charset.name": "名称只能使用字母、数字、点、下划线和短横线，长度 1 到 64 个字符",
    "err.name.charset.run": "运行名称只能使用字母、数字、点、下划线和短横线，长度 1 到 64 个字符",
    "err.name.charset.workspace": "工作区名称只能使用字母、数字、点、下划线和短横线，长度 1 到 64 个字符",
    "err.name.empty.config": "配置名称不能为空",
    "err.name.empty.name": "名称不能为空",
    "err.name.empty.run": "运行名称不能为空",
    "err.name.empty.workspace": "工作区名称不能为空",
    "err.run.exists": "已经有名为 {run} 的运行了，换一个名字",
    "err.run.frozen": "这个运行正在跑，配置不能改",
    "err.run.no_config": "运行 {run} 还没有配置文件",
    "err.run.not_found": "找不到运行 {run}",
    "err.run.required": "没有指定运行",
    "err.settings.key_invalid": "密钥无效（{reason}），没有保存",
    "err.settings.key_rejected": "密钥无效（服务拒绝了这把钥匙），没有保存",
    "err.settings.invalid": "设置校验没通过，没有保存",
    "err.settings.workspaces_empty": "工作区根目录不能为空",
    "err.settings.workspaces_not_dir": "工作区根路径不是目录：{path}",
    "err.settings.path_empty": "路径为空",
    "err.settings.path_variable": "路径里的 ${MODELS} 还没展开：先设好 MODELS 环境变量，或者直接填绝对路径",
    "err.settings.path_missing": "路径不存在：{path}",
    "err.settings.te_not_safetensors": "文本编码器文件必须是 .safetensors",
    "err.settings.te_no_config": "目录里没有 config.json，这不是 HuggingFace 模型目录",
    "err.settings.te_kind": "文本编码器要填 HF 目录或 .safetensors 文件",
    "err.settings.file_expected": "这是个目录，这里要填 .safetensors 文件",
    "err.settings.not_safetensors": "模型文件必须是 .safetensors",
    "err.settings.key_env_unreachable": "环境变量 {name} 已经保存，但服务现在连不上（{reason}）",
    "err.settings.key_literal_note": "填的是密钥本身，只在这次运行有效，不会写进设置文件；想长期用请填环境变量名",
    "err.settings.key_unreachable": "服务现在连不上：{reason}",
    "err.settings.model_missing": "服务里没有模型 {model}（现有：{ids}），跑起来会失败",
    "err.settings.model_kind": "模型种类只能是 dit、text_encoder 或 vae",
    "err.settings.not_mapping": "设置要是一组“键: 值”，现在不是",
    "err.thumbnail.failed": "生成缩略图失败：{error}",
    "err.warning.eval_off": "验证间隔是 0：训练不会记录验证损失",
    "err.warning.prompts_empty": "样图提示词是空的：不会生成预览图，训练时看不到样图",
    "err.warning.samples_off": "采样间隔是 0：训练不会生成样图",
    "err.workspace.delete_reserved": "这个名称是系统保留的，不能删除",
    "err.workspace.exists": "已经有名为 {name} 的工作区了",
    "err.workspace.none": "还没有工作区，先建一个",
    "err.workspace.not_found": "找不到工作区 {name}",
    "err.workspace.reserved": "{name} 是系统保留的名字，不能用作工作区",
    "err.yaml.empty": "YAML 是空的",
    "err.yaml.not_mapping": "配置要是一组“键: 值”，现在不是",
    "err.yaml.syntax": "YAML 写错了：第 {line} 行第 {column} 列附近，{problem}",
    "err.yaml.syntax_plain": "YAML 写错了：{problem}",
}


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def render(code: str, params: dict | None = None) -> str:
    """Fill the Chinese template; unknown codes come back as the code itself."""
    template = CODES.get(code)
    if template is None:
        return code
    values = params or {}

    def swap(match):
        name = match.group(1)
        return str(values[name]) if name in values else match.group(0)

    return _PLACEHOLDER.sub(swap, template)


def render_all(problems) -> list[str]:
    return [item.render() if isinstance(item, Problem) else str(item) for item in problems]
