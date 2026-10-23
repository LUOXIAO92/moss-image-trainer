"""User-facing errors as codes.

Every message the WebUI can show is a code plus its parameters. The code travels
in ``detail.code``, the rendered Chinese in ``detail.message`` so logs, the CLI
and any older client still read a sentence. The four i18n packs translate
``err.<code>``; :func:`render` is the last-resort fallback.

    raise fail(404, "err.workspace.not_found", name=name)
"""

from __future__ import annotations

import re

from fastapi import HTTPException


# Code -> Chinese template. Keep in sync with .tmp/err-manifest.json;
# tests/test_error_keys.py fails when a raise site uses a code missing here.
CODES: dict[str, str] = {
    "err.cache.dir_empty": "没有设置缓存目录，没法检查缓存",
    "err.cache.legacy": "{dir} 里的缓存是旧格式，训练集和正则集混在一起，去“缓存”页重新生成一次",
    "err.cache.missing_text": "{dir} 里的缓存缺了 {missing} 条文本编码（一共 {items} 条），去“缓存”页勾上“文本编码”补一遍",
    "err.cache.not_built": "{dir} 还没有生成缓存，先去“缓存”页生成一遍",
    "err.cache.parts_required": "至少要选一项：latent 缓存或文本编码",
    "err.cache.text_encoder_required": "文本编码需要先在训练设置里填文本编码器（本地 Qwen3.5-9B 目录），填好再来生成",
    "err.cache.sizing": "{dir} 里的缓存是按另一个面积上限生成的，跟现在的设置对不上，去“缓存”页重新生成 latent 缓存",
    "err.cache.stop_dead": "那个缓存进程已经不在了",
    "err.cache.stop_denied": "没有权限停止这个进程",
    "err.cache.stop_finished": "这个缓存任务已经结束了",
    "err.cache.stop_foreign": "进程 {pid} 看起来不是缓存任务，没有向它发送停止信号",
    "err.cache.stop_no_job": "现在没有正在运行的缓存任务",
    "err.caption.dataset_required": "先勾选要生成提示词的数据集：训练集或正则集",
    "err.caption.key_unreadable": "设置里存的是环境变量名 {name}，但当前程序读不到它的值：先导出这个变量并重启，或者直接把密钥填进“提示词生成”页",
    "err.caption.no_endpoint": "提示词生成还没填服务地址，到“提示词生成”页填好再开始",
    "err.caption.no_key": "提示词生成还没填密钥，到“提示词生成”页填好再开始",
    "err.caption.reg_missing": "还没有设置正则集目录，没法给正则集生成提示词",
    "err.caption.style_required": "先勾选要生成的风格：tag + caption 或官方描述",
    "err.config.invalid": "配置有问题：{problems}",
    "err.config.structure": "配置结构不对：{error}",
    "err.dataset.missing": "找不到训练集目录：{path}",
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
    """Fill the Chinese template; unknown placeholders stay visible."""
    template = CODES.get(code)
    if template is None:
        return code
    values = params or {}

    def swap(match):
        name = match.group(1)
        return str(values[name]) if name in values else match.group(0)

    return _PLACEHOLDER.sub(swap, template)


def error_detail(code: str, **params) -> dict:
    """The JSON body every error carries: code, parameters, fallback message."""
    return {"code": code, "params": params, "message": render(code, params)}


def fail(status: int, code: str, headers: dict | None = None, **params) -> HTTPException:
    """Build the exception; callers ``raise fail(...)``."""
    return HTTPException(status_code=status, detail=error_detail(code, **params), headers=headers)
