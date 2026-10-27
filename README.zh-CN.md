# MOSS-Image LoRA Trainer

[English](README.md) · **[简体中文](README.zh-CN.md)** · [日本語](README.ja.md) · [한국어](README.ko.md)

面向 [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（5B 单流 DiT、冻结 Qwen3.5-9B 文本编码器与 FLUX.2 VAE）的本地 LoRA 微调工具，包含 PyTorch 训练栈、FastAPI 后端与原生 ES module 前端。

本仓库复用官方 `comfyui/moss-image/model.py` 与 `loaders.py` 的架构和权重映射，实现上游缺失的微调组件：数据集扫描与尺寸计算、潜变量与文本特征缓存、Flow Matching 训练、采样预览、提示词生成与 WebUI。

训练器将流程拆为离线缓存与在线训练两阶段，避免 5B DiT 与 9B 文本编码器同时驻留显存。训练前先将潜变量和文本特征写入磁盘，训练时仅加载 5B DiT，可在单张消费级显卡上完成 1024² 微调。产出的 `.safetensors` 检查点默认采用 `lora_unet_*` 键名，可直接放入 ComfyUI 加载，训练与导出不依赖 ComfyUI 运行时。

## 主要特性

- 分阶段缓存：FLUX.2 VAE 编码潜变量，本地 Qwen3.5-9B 编码文本特征并落盘。文本批处理采用右侧因果 padding，训练前向按掩码切取有效长度，避免 padding 污染文本特征。
- 流匹配训练与显存分档：采用截断 logit-normal $\sigma$ 采样与按 token 归一化的速度场 MSE 损失。六档显存预设（`32`、`24`、`16`、`12`、`8`、`6` GB）组合激活检查点、权重量化（`quantize_dit: none|ffn|all`，格式 `nf4` 或 `fp8`，conditioning 半边可选）、从主机内存流式加载块、`torch.compile` 与 `fp16` 累加，训练支持 `packed`（FlashAttention-2 varlen）与 `padded`（token 维补齐）两种批处理模式。仪表盘在运行前后并排显示预测显存峰值与实测峰值。
- 正则集与留出验证：通过 `dataset.reg_path` 引入正则集抑制概念漂移，按 `train.eval_holdout` 切分留出集定期计算验证损失。
- 从已有适配器预热：`lora.init_from` 指向一个 `.safetensors` LoRA（kohya、PEFT 或 ComfyUI 命名），训练开始前把其中的权重装进新适配器。文件的 rank 必须等于 `lora.rank`；alpha 不同会折进 `lora_B`，加载后的模型因此仍复现文件原来的计算。WebUI 在启动时检查文件，rank 不一致直接拒绝，缩放 alpha 前会先询问。
- 双风格提示词生成：调用 OpenAI 兼容视觉模型接口，支持保留原 tag 并补写场景句的 `tag_caption` 与长中短三档描述的 `official` 风格，写入同名 sidecar JSON。
- 免构建 WebUI：前端运行原生 ES module（含中控台、训练配置、数据集、缓存、采样对比、导出、工作区、设置、提示词生成九个标签页），修改代码后刷新直接生效。仪表盘只有一个训练操作：有检查点时继续训练选中的运行；勾选**重新训练**复选框后，下一次训练会先确认，再从第 0 步开始。
- 原生 ComfyUI 格式：检查点与导出文件默认采用 `lora_unet_*` 键名，支持转换 `diffusers` 格式或按步数区间批量打包。
- 多语言界面：内置简体中文、English、日本語、한국어 四套词典，支持无刷新切换、模板导出与自定义语言包覆盖。

## 环境要求

- 系统与解释器：Linux、WSL2 或 Windows 10/11（PowerShell 5.1+），Python 3.11+。
- 硬件与驱动：支持 `bfloat16` 或 `float16` 的 NVIDIA GPU，CUDA 驱动匹配 PyTorch 2.9+。
- 显存设置，预设只是起点：显存的每一项都是运行配置里的普通值，训练器照着这些值跑。WebUI 里的预设下拉（`32`、`24`、`16`、`12`、`8`、`6`）和 CLI 的 `--preset` 只把某一档的值一次性填进来——梯度检查点、常驻层数、激活值卸载、量化范围与格式、`torch.compile`、8-bit 优化器、预览 VAE 精度——填完就不再参与。新建运行从默认值开始，不套用任何档位，也不按显卡推导任何东西；只有单步真的显存不足时，训练器才会自己调高 `blocks_to_swap`。
- 本地模型：MOSS-Image DiT、Qwen3.5-9B 文本编码器（HuggingFace 目录，或 ConvRot int8 量化的单文件 `.safetensors`）与 FLUX.2 VAE。

要按具体显卡挑设置，看[显存指南](<docs/VRAM.md>)：写清了量化与块流式加载的实测开销、分配器设置，并按显卡档位给出预览尺寸。

## 安装

仓库通过根目录 `.venv` 隔离依赖，CLI 与 WebUI 后台任务均直接调用 venv 里的解释器（Linux 为 `.venv/bin/python`，Windows 为 `.venv\Scripts\python.exe`），不依赖外部 conda 或 ComfyUI 环境。

[setup.sh](<setup.sh>) 按 `python3.14` 至 `python3.11` 查找解释器，创建 `.venv` 并安装 [requirements.txt](<requirements.txt>)，再调用 [setup_fa2.py](<setup_fa2.py>) 匹配当前 Python、PyTorch、CUDA 与 GPU 架构安装预编译 FlashAttention-2 轮子（未匹配时告警并回退至 SDPA）。[run.sh](<run.sh>) 检查端口占用并在缺失 `.venv` 时先执行 `./setup.sh`，随后启动 WebUI（默认监听 `127.0.0.1:7861`）。

原生 Windows 上由 [setup.ps1](<setup.ps1>) 承担同样的工作：通过 `py` 启动器找解释器，创建 `.venv`，按当前 NVIDIA 驱动挑 CUDA 索引装 torch（驱动 527+ 用 `cu126`，570+ 用 `cu128`，580+ 用 `cu130`），再装 [requirements.txt](<requirements.txt>)；`.venv` 缺失时 [run.ps1](<run.ps1>) 会先调用它。FlashAttention-2 的预编译轮子只发布 Linux 版本，因此 Windows 上用 `padded` 批次加 SDPA 训练，除非 `-WithFlashAttention` 能匹配到轮子。

```bash
# 创建 .venv 并安装依赖（可通过 PYTHON 或 TORCH_INDEX_URL 指定环境）
./setup.sh
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./setup.sh --force

# 启动或停止指定端口上的 WebUI
./run.sh --port 7869
./run.sh --port 7869 --stop
```

```powershell
# Windows：同样的安装，CUDA 索引按当前驱动自动选
.\setup.ps1
.\setup.ps1 -Force -TorchIndex cu130
.\setup.ps1 -DryRun

# 启动 WebUI（首次运行会先创建 .venv）
.\run.ps1 --port 7869
```

若 Windows 禁止运行本地脚本，改用 `powershell -ExecutionPolicy Bypass -File .\setup.ps1`。

## 快速开始

WebUI 模式下，运行 `./run.sh --port 7869` 并访问 `http://127.0.0.1:7869`，在「工作区」新建工作区，在「训练设置」新建一个运行并填入 `<MODELS_DIR>` 与 `<DATASET_DIR>`，再依次启动潜变量缓存、文本特征缓存与训练。

CLI 模式下，基于某次运行的配置分三步执行：

```bash
WS=workspaces/<WORKSPACE>
RUN=<run>
CONFIG="$WS/runs/$RUN/config.yaml"

# 1. 编码图像潜变量缓存（写入 $WS/train_cache）
.venv/bin/python -m moss_lora.cli cache --config "$CONFIG"

# 2. 编码文本特征缓存（本地 Qwen3.5-9B）
.venv/bin/python -m moss_lora.cli encode-text --config "$CONFIG"

# 3. 启动 LoRA 训练（输出至 $WS/runs/<name>）
.venv/bin/python -m moss_lora.cli train --config "$CONFIG"
```

数据集扫描器递归读取 `.jpg`、`.jpeg`、`.png`、`.webp` 与 `.bmp` 图片，配对 sidecar JSON、同名 `.txt`/`.caption` 或 `metadata.jsonl` 标注；`3_concept` 子目录将内部图片重复采样 3 次。

常用辅助子命令：

```bash
# 查看数据集统计与字幕覆盖
.venv/bin/python -m moss_lora.cli info --config "$CONFIG"

# 按显存预算（GB）规划精度、量化与块交换参数并写回配置
.venv/bin/python -m moss_lora.cli plan --vram 12 --config "$CONFIG" --write "$CONFIG"

# 生成无 LoRA 基准图，或重绘指定步数检查点的预览图
.venv/bin/python -m moss_lora.cli baseline --config "$CONFIG"
.venv/bin/python -m moss_lora.cli preview --config "$CONFIG" --step 500

# 编码自定义提示词并用指定检查点采样
.venv/bin/python -m moss_lora.cli encode-prompt --config "$CONFIG" \
    --out "$WS/custom_prompts.safetensors" --prompt "a portrait in soft daylight"
.venv/bin/python -m moss_lora.cli sample --config "$CONFIG" \
    --checkpoint "$WS/runs/<name>/checkpoints/<ckpt>.safetensors" \
    --embeds "$WS/custom_prompts.safetensors" --out "$WS/samples"

# 顺序续跑工作区内未完成的训练任务
.venv/bin/python -m moss_lora.cli queue --workspace "$WS"

# 起一个掉线也继续跑的训练，再用看板同款的方式读它的状态
.venv/bin/python -m moss_lora.cli train --config "$CONFIG" --detach
.venv/bin/python -m moss_lora.cli status
.venv/bin/python -m moss_lora.cli status --workspace my-lora --live --json
```

`status` 直接读工作区文件：不连服务，不读缓存。看板、`GET /api/jobs` 和这条命令投影的是同一份心跳，所以终端里起的运行会出现在浏览器里，浏览器里起的运行也会出现在终端里。[docs/RUNTIME-CONTRACT.zh-CN.md](<docs/RUNTIME-CONTRACT.zh-CN.md>) 规定这些文件，以及保证三者一致的那个测试。

## 配置

配置分为机器级 `settings.yaml` 与每次运行自带的配置两层。

`settings.yaml` 存放本机参数（已被 `.gitignore` 忽略），首次启动从 [settings.example.yaml](<settings.example.yaml>) 复制生成，管理工作区根目录 `paths.workspaces`、CUDA 分配器 `runtime`（`allocator: expandable|async|off` 与 `gc_threshold`）、界面语言 `ui.language` 及提示词生成服务 `caption`（`base_url`、环境变量名 `api_key`、模型名、并发数与模板）。设置页直接输入的明文密钥仅驻留当前进程内存，不写盘。

每次运行的完整配置在 `workspaces/<WORKSPACE>/runs/<RUN>/config.yaml`，定义模型、数据集、缓存、LoRA 结构、训练超参、显存设置与预览提示词。相对路径（如 `cache.dir`、`output_dir`）相对于工作区目录解析，移动工作区目录不破坏引用。该运行训练期间这份配置只读，运行结束后可改。在 WebUI 里新建运行即可得到一份，也可以在运行对话框里从已有运行复制：

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

`train.seed` 驱动训练里的每一次随机抽取：批次采样、sigma、每步的流匹配噪声、提示词丢弃与适配器初始化。检查点会带上 RNG 状态，续训因此继续同一条轨迹。同一台机器上，同一配置跑两次得到相同的损失，代价是 FlashAttention-2 的 varlen 反向改用确定性模式，单步时间约多十分之一。`preview.seed` 只控制预览采样。

## 提示词生成

提示词生成模块（[captioner.py](<moss_lora/captioner.py>)）通过 OpenAI 兼容接口调用视觉模型，可在 WebUI「提示词生成」页或命令行运行：

```bash
.venv/bin/python -m moss_lora.captioner \
    --images <DATASET_DIR> \
    --styles tag_caption,official \
    --base-url http://127.0.0.1:8080/v1 \
    --api-key VLM_API_KEY
```

地址与密钥也可以走环境变量 `VLM_BASE_URL`、`VLM_API_KEY`、`VLM_MODEL`；`--api-key` 既接受密钥本身，也接受存有密钥的环境变量名（`cli caption` 会读取 `settings.yaml` 的对应字段）。

`--styles` 可单独或同时指定两种风格：

- `tag_caption`（`tag_caption_style`）：读取 `<stem>.txt`/`.caption`（缺失时回退至 sidecar 的 `kept_tags`），保留原有 tag，由视觉模型补写姿态、景别、角度、背景与光影场景句并过滤发色、瞳色等身份词；无可用 tag 时跳过。
- `official`（`official_style`）：按官方 recaptioner 规范忽略现有标注，输出 `caption_long`、`caption_medium`、`caption_short` 三档描述及水印坐标与边框方向。

两种风格共用 `<stem>_moss-image-trainer.json`（`schema: 2`），不修改原 `.txt`。重跑单风格只更新对应字段；默认跳过已有结果，`--overwrite` 强制重算，`--migrate` 升级旧版扁平 sidecar，`--validate-only` 仅校验现有文件：

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

`dataset.moss_caption_source` 指定训练读取字段（`tag_caption`、`official_long`、`official_medium` 或 `official_short`）；配置 `dataset.metadata` 时优先读 `metadata.jsonl`，缺失所选风格时回退至同名 `.txt`/`.caption` 与 `reg_caption`。切换来源后只需重跑 `encode-text`，无需重建潜变量缓存。

批次是缓存配置而不是隐式猜测：`cache.text_batch` 与 `cache.vae_batch` 可以把它固定下来（`0` 表示按空闲显存自适应），构建这份缓存时用的批次记录在 `train_cache/index.json` 的 `batch` 字段。VAE latent 与批次无关（`LatentCodec` 用全精度 fp32 卷积计算），文本嵌入则随批次变化，需要可复现的缓存时把文本批次固定下来。

`train.caption_dropout` 是逐样本的提示词丢弃率，取值 `[0, 1]`，默认 `0`。`encode-text` 会把无条件提示词写到缓存旁的 `empty_text.safetensors`。比例大于 0 时，训练按该比例把提示词嵌入替换为这份缓存提示词；为 0 时保持原样。要在旧缓存上启用丢弃，需要重跑一次 `encode-text` 补上提示词。

## 界面语言

WebUI 内置简体中文（`zh`）、English（`en`）、日本語（`ja`）、한국어（`ko`）词典（`moss_lora/web/static/i18n/<code>.json`），通过顶栏下拉框无刷新切换。

「设置 → 界面语言」支持导出模板并导入自定义语言包。填写 `$code`（如 `ru`、`pt-br`）、`$name` 与译文（保留 `{n}`、`{path}` 等占位符）后上传，存入 `i18n/<code>.json`（已被 `.gitignore` 忽略）。同名 `$code` 覆盖内置词典，删除后恢复内置版本；未翻译条目按「当前语言 → 简体中文 → 原始 key」回退。

## 导出到 ComfyUI

训练检查点（`runs/<name>/checkpoints/stepXXXXXX.safetensors`、`final.safetensors`）与最终导出文件 `runs/<name>/<name>.safetensors` 均采用 ComfyUI 兼容的 kohya 键名。优化器与随机数状态单独写入 `state.safetensors`（或 `<ckpt>.state.safetensors`）。

张量键名以 `lora_unet_` 开头并将层路径点号替换为下划线，由 ComfyUI `Load LoRA` 节点映射至 `diffusion_model.layers.<层号>.proj.<模块名>.weight`：

```text
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_down.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_up.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.alpha
```

将 `.safetensors` 放入 `<COMFYUI_DIR>/models/loras/` 后连接 `Load LoRA` 节点加载。转换 `diffusers` 格式（`transformer.layers.*`）或按步数区间批量导出可在 WebUI「导出」页或通过 CLI 执行（纯键名映射，不加载 5B 模型）：

```bash
# 导出单个步数检查点（同时生成 kohya 与 diffusers 格式）
.venv/bin/python -m moss_lora.cli export \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --checkpoint 1000 \
    --formats kohya,diffusers \
    --out "workspaces/<WORKSPACE>/exports"

# 按步数区间批量导出并打包为 zip
.venv/bin/python -m moss_lora.cli export-batch \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --steps 250-1000:250 \
    --formats kohya \
    --out "workspaces/<WORKSPACE>/exports" \
    --zip "workspaces/<WORKSPACE>/exports/batch.zip"
```

## 测试

单元测试无需 GPU，覆盖 sidecar 读写与迁移、四语词典完整性、API 密钥存储规则与文档脱敏扫描；另提供本地文本编码器右 padding 一致性测试：

```bash
.venv/bin/python tests/test_sidecar.py
.venv/bin/python tests/test_i18n.py
.venv/bin/python tests/test_caption_key.py
.venv/bin/python tests/test_docs_sanitized.py

# 可选：验证右 padding 批处理与单条编码的一致性
MOSS_TE=<MODELS_DIR>/text_encoders/Qwen3.5-9B .venv/bin/python tests/test_text_padding.py
```

前端回归夹具 [smoke.mjs](<tests/ui/smoke.mjs>) 基于 Node.js 原生 ESM 与 Chromium DevTools Protocol（无外部 npm 依赖），在 1440×1000 视口下遍历 4 种语言与 9 个标签页，检查控制台异常、失败请求、横向溢出、文本裁切与漏翻裸 key：

```bash
.venv/bin/python -m moss_lora.cli web --port 7891 &
chromium --headless=new --remote-debugging-port=9401 \
    --user-data-dir=.tmp/chrome --no-sandbox --disable-gpu about:blank &
node tests/ui/smoke.mjs --base http://127.0.0.1:7891 --cdp http://127.0.0.1:9401 \
    --langs zh,en,ja,ko --out .tmp/ui.json
```

## 目录结构

```text
moss-image-trainer/
├── setup.sh / setup.ps1 / setup_fa2.py   # 创建 .venv、安装依赖与匹配 FlashAttention-2 轮子
├── run.sh / run.ps1          # 环境自检并启动 WebUI
├── settings.example.yaml     # 本机全局配置模板
├── moss_lora/
│   ├── cli.py                # CLI 入口（cache / encode-text / train / export / web 等）
│   ├── config.py             # 配置 dataclass、路径解析、校验与显存预设包
│   ├── model.py              # 官方 MOSS-Image DiT 架构适配（SDPA / FA2 与检查点）
│   ├── weights.py / convrot.py        # DiT、VAE 权重映射与 ConvRot int8 反量化
│   ├── data.py / cache.py    # 数据集扫描、逐图尺寸与图像潜变量缓存
│   ├── hf_text.py / encode.py # 本地 Qwen3.5-9B 加载与文本特征缓存
│   ├── captioner.py          # 提示词生成 worker 与 sidecar JSON 管理
│   ├── trainer.py            # Flow Matching 训练循环、采样预览与检查点管理
│   ├── quant.py / offload.py # NF4 冻结层量化与 CPU 块交换
│   ├── lora_file.py / export_batch.py # 免加载模型的 LoRA 格式转换与批量导出
│   ├── i18n.py / settings.py # 语言包合并与机器级配置读写
│   └── web/                  # FastAPI 路由/服务层与原生 ES module 前端
└── tests/                    # Python 单元测试与 CDP 前端回归夹具
```

模块边界、数据流与前后端约定见 [docs/DESIGN.zh-CN.md](<docs/DESIGN.zh-CN.md>)。

## 已知限制

- 批大小与吞吐：一个微批按 token 预算组装，图片各自保留分辨率，因此决定单步工作量的是 `train.bin_tokens` 而不是 `batch_size`。装上 FlashAttention-2 后默认从 `padded`（token 补齐 + key mask）切到 `packed`（varlen，无 mask）。
- 低显存精度权衡：`model.dtype: float16` 配合 `vram.fp16_accumulate: true` 在高学习率或长步数下比 `bfloat16` 更易数值不稳定；量化（`quantize_dit: ffn|all`）面向 8 GB 和 6 GB 两档；16 GB 和 12 GB 两档保留 bf16 权重，把尾部块从主机内存流式加载，代价是单步时间，不是权重精度。`nf4` 最省、`fp8` 更稳，而 conditioning 半边会被每个块乘进去，刻意量化等于把误差叠加到全网络，所以默认关闭。
- 激活值卸载：`vram.offload_activations: true` 把梯度检查点边界写入锁页主机内存，在 1024² 下省约 1 GB 显存，代价是单步时间约多十分之一。它要求开启梯度检查点，并会关闭 `torch.compile`。
- 训练范围：仅对 5B DiT 注入 LoRA（默认覆盖 42 层的 `q`、`k`、`v`、`o`、`ffn_up_gate`、`ffn_down` 及可选 `text_embedder`），不支持微调 Qwen3.5-9B 文本编码器或 VAE。

## 许可与上游致谢

- 模型架构（[model.py](<moss_lora/model.py>)）、调度器（[scheduler.py](<moss_lora/scheduler.py>)）与权重映射改编自 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（Apache-2.0 License）。基础权重与重打包权重见 [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)、[OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) 与 [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image)。
- 内置 Qwen3.5 分词器与模型配置（`moss_lora/assets/`）遵循 Apache-2.0 许可证，见 [QWEN_LICENSE.txt](<moss_lora/assets/QWEN_LICENSE.txt>)。
