# MOSS-Image LoRA Trainer 设计书

[English](DESIGN.md) · **简体中文**

本文档规定 `moss-image-trainer` 的架构边界、核心机制与工程取舍。模型结构与权重映射复用官方 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) 实现；本仓库只实现数据集处理、两阶段特征缓存、Flow Matching 训练循环、采样预览、VLM 提示词生成与无构建步骤的 WebUI。

## 1. 项目定位与边界

本仓库提供面向 MOSS-Image-SFT 与 MOSS-Image-Base 的单机单卡 LoRA 微调工具链，通过离线预缓存将 9B 文本编码器与 FLUX.2 VAE 移出训练进程。

官方仓库公开了推理管线与 ComfyUI 节点，未包含微调代码。MOSS-Image 由 5B 单流扩散 Transformer（DiT，42 层）、冻结的 Qwen3.5-9B 文本编码器与 32 通道 FLUX.2 VAE 组成。三者在训练期同时驻留显存会超出消费级显卡容量。训练器把管线拆成离散子命令：先把图像潜变量与文本特征写入磁盘，再在独立进程中只加载 5B DiT 执行梯度反传，最后直接写出 ComfyUI 原生识别的 `lora_unet_*` 权重文件。所有功能既可通过 [cli.py](<../moss_lora/cli.py>) 在终端运行，也可通过 [server.py](<../moss_lora/web/server.py>) 提供的 WebUI 按工作区管理。

本仓库严守以下工程边界：
- 只训练 DiT 的低秩适配器（`layers.<0..41>.proj.<q|k|v|o|ffn_up_gate|ffn_down>` 及可选的 `text_embedder`），不实现全参数微调，也不微调 Qwen3.5-9B 与 FLUX.2 VAE。
- 只面向单机单卡环境与本地文件系统，不引入分布式通信（DDP / FSDP / DeepSpeed）、多租户鉴权或远端对象存储。
- 运行时不依赖外部 ComfyUI 进程或宿主机 conda 环境，所有任务固定由项目根目录下的 `.venv/bin/python` 执行，也不在后台自动联网拉取模型权重。

这样取舍是因为角色与画风微调的瓶颈在于显存容量而非跨机带宽。砍掉分布式与全量微调抽象后，训练循环可以直接围绕单卡显存分档（NF4 量化、CPU 块交换、变长序列打包）做针对性优化；代价是无法通过多卡并行缩短单次长训耗时，也不适用于需要改写文本塔表征的生僻概念注入。

## 2. 总体架构

系统按「提示词生成 → 潜变量缓存 → 文本特征缓存 → 训练与采样 → 导出」的单向数据流运转，WebUI 仅作为无状态调度壳通过子进程调用 CLI。

各阶段的输入、输出与代码入口全部收敛在工作区目录 `workspaces/<WORKSPACE>/` 内：

```text
workspaces/<WORKSPACE>/
├── train_cache/                      # 训练集缓存目录（cache.dir 默认值）
│   ├── <uid>.safetensors             # 单样本缓存（含 latent、text_embeds、text_mask）
│   ├── preview.safetensors           # 预览提示词与负面提示词特征包
│   └── index.json                    # 缓存索引（schema: 2）
├── reg_cache/                        # 正则集缓存目录（cache.reg_dir 默认值）
│   └── <uid>.safetensors             # 正则样本缓存
├── baseline/                         # 无 LoRA 基线对比样图（baseline_N.png）
├── logs/                             # 任务运行日志与配置快照
│   ├── <job_id>.yaml                 # 任务启动时的实际配置快照
│   └── <run_id|cache-*|caption-*>.log
├── runs/<run_id>/                    # 一次运行：它自己的配置和全部产物
│   ├── config.yaml                   # 这次运行的完整配置；训练进行中只读
│   ├── metrics.jsonl                 # 逐歩训练损失、验证损失、学习率与显存记录
│   ├── training.json                 # CLI 训练进程心跳文件（pid / step / time）
│   ├── checkpoints/
│   │   ├── stepXXXXXX.safetensors    # 周期检查点（kohya 键名，仅含 LoRA 权重）
│   │   ├── final.safetensors         # 结束检查点
│   │   └── state.safetensors         # 滚动更新的优化器动量与 RNG 状态
│   ├── samples/
│   │   ├── stepXXXXXX_N.png          # 周期采样预览图及同名 .png.json 元数据
│   │   ├── seeds.json                # 采样种子与提示词汇总
│   │   └── trajectory.png            # 训练全周期预览网格拼图
│   └── <name>.safetensors            # 默认导出的最终 LoRA 文件
└── exports/                          # 批量导出与 zip 归档目录
```

进程边界实行严格的物理隔离。FastAPI 主进程（[server.py](<../moss_lora/web/server.py>)）不创建 CUDA 上下文，连 GPU 状态查询都通过子进程调用 `nvidia-smi` 完成（见 [system.py](<../moss_lora/web/services/system.py>)），避免 Web 服务常驻占用数百兆显存。当用户在页面发起缓存、提示词生成、训练或导出任务时，[launcher.py](<../moss_lora/web/services/launcher.py>) 将当前配置快照写入 `logs/<job_id>.yaml`，再通过 `subprocess.Popen` 启动独立的 `python -m moss_lora.cli <subcommand>` 子进程。子进程退出后立即向操作系统归还全部显存与主机内存。

这套子进程隔离机制消灭了跨阶段显存泄漏与 CUDA 上下文污染，并且保证 CLI 与 WebUI 行为完全等价；代价是每次启动任务都要承受 Python 解释器初始化与模型权重从磁盘加载的冷启动延迟，不适用于亚秒级交互式推理。

## 3. 模型复用策略

模型架构、时间步调度与权重量化反解全部复用上游参考实现，本仓库仅通过轻量适配层抹平磁盘权重格式差异并注入训练钩子。

### 3.1 DiT 架构适配与 4D RoPE

[model.py](<../moss_lora/model.py>) 直接移植自官方 `MOSSImageTransformer`（42 层单流块，隐藏维 3072，24 头，每头 128 维，FFN 维 8192，SiTU-GLU 激活，共享时间步调制 `shared_block_modulation` 与 4D 轴向 RoPE），只做五处训练适配：
1. 注意力后端分流：将上游注意力分发器替换为 `optimized_attention`。当调用方传入 `cu_seqlens` 与 `max_seqlens` 时，直接调用 FlashAttention-2 的 `flash_attn_varlen_func`（要求 `B=1` 的拼接序列）；否则调用 `F.scaled_dot_product_attention` 并传入 `attn_mask`。
2. 块级检查点与交换钩子：在 `_Block` 中将前向拆分为 `attn` 与 `ffn` 两半，并在 `MOSSImageTransformer._run_block` 中接入 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 与 `block_swap` 钩子。
3. 多批处理路径：新增 `_forward_batched`（等长序列向量化前向）、`forward_packed`（变长序列拼接前向）与 `forward_padded`（token 维补齐前向）。
4. 4D 轴向位置编码：`_RoPE` 将 128 维头维度切分为 4 个 32 维轴 `(r, f, h, w)`，频率基底 `base=10000.0` 始终在 CPU 上以 `float32` 计算。序列顺序固定为「图像 token 在前、文本 token 在后」（`torch.cat((image, text), 0)`）：图像 token 的参考轴 $r=0$，时间轴 $f$ 恒取文本有效长度 $L$，空间轴 $(h, w)$ 减去 $(H-1)/2$ 与 $(W-1)/2$ 居中（支持传入 `center` 参数防止画布补齐引发相位漂移）；文本 token 的坐标为 $(0, 0 \dots L-1, 0, 0)$。
5. 无条件跳层（SLG）：增加 `skip_layers` 元组，采样阶段运行无条件分支时跳过指定层（默认跳过第 22 层）。

### 3.2 权重格式转换与 ConvRot 反量化

权重加载层（[weights.py](<../moss_lora/weights.py>)、[convrot.py](<../moss_lora/convrot.py>) 与 [hf_text.py](<../moss_lora/hf_text.py>)）在不依赖 ComfyUI 的前提下读取三种本地权重形态：
- DiT 融合投影拆分：官方与 ComfyUI 单文件权重将注意力投影合并为 `layers.<i>.proj.qkv.weight`（形状 `[9216, 3072]`）。`split_fused_qkv` 在加载时沿第 0 维按 `chunk(3, dim=0)` 拆分为独立的 `q.weight`、`k.weight`、`v.weight`，并弹出由 `_RoPE` 动态重建的 `rope.bands`（`text_embedder.blocks.*.qkv` 保持融合不变）。
- FLUX.2 VAE 键名映射与潜变量规约：`legacy_vae_to_diffusers` 将单文件 VAE 的旧版键名（如 `.mid.attn_1.`、四维 `1x1` 卷积权重、`nin_shortcut`、以及逆序排列的 `decoder.up.<3-i>`）映射为 `diffusers.AutoencoderKLFlux2` 结构。`LatentCodec` 严格复刻官方管线顺序：
  1. 编码：图像先经 `vae.encode` 取分布众数 `dist.mode()`（或 `cache.latent_sample: true` 时取 `dist.sample()`），得到 `[B, 32, H/8, W/8]`；再经 `pack_latents` 做 $2\times 2$ space-to-depth 重排为 `[B, 128, H/16, W/16]`（总下采样步幅 16）；最后用 `vae.bn` 的 `running_mean` 与 `running_var`（`BN_EPS = 1e-4`）按 $(z - \mu) / \sqrt{\sigma^2 + \epsilon}$ 归一化。
  2. 解码：先乘以标准差并加回均值，再经 `unpack_latents` 还原为 32 通道后送入 `vae.decode`。
- Qwen3.5-9B 文本塔与 ConvRot 反量化：`hf_text.load_text_model` 只在目标设备上直接实例化纯文本塔 `Qwen3_5TextModel`（跳过耗时的 CPU 随机初始化），通过 `convrot.normalize_key` 剥离 `model.language_model.`、`language_model.`、`text_model.` 或 `model.` 前缀，丢弃 `visual.`、`vision.`、`mtp.`、`lm_head.`、`audio.` 与 `projector.`。当 `model.text_encoder` 指向 ConvRot int8 量化的单文件时，`convrot.dequantize` 读取 `.weight_scale` 与 `.comfy_quant` JSON 标记（后者只是打包格式里的元数据键名），构造分块正则 Hadamard 矩阵：

$$H_4 = \begin{bmatrix} 1 & 1 & 1 & -1 \\ 1 & 1 & -1 & 1 \\ 1 & -1 & 1 & 1 \\ -1 & 1 & 1 & 1 \end{bmatrix}, \quad H = \frac{1}{\sqrt{G}} \bigotimes H_4$$

  其中块大小 $G$ 取 `convrot_groupsize`（默认 256），按 $W = (q \cdot s) \operatorname{blockdiag}(H)$ 在内存中还原为 `bfloat16`，与 `comfy_kitchen.dequantize_int8_convrot_weight_dtype` 逐位一致。分词器与文本塔结构配置内置于 `moss_lora/assets/`，单文件权重无需外部 `config.json` 即可独立运行。

### 3.3 调度器与噪声采样

推理与训练时间步调度分别位于 [scheduler.py](<../moss_lora/scheduler.py>) 与 [sigma.py](<../moss_lora/sigma.py>)：
- `MOSSImageScheduler` 继承 `FlowMatchEulerDiscreteScheduler`，先按推理 `shift`（默认 `5.0`）生成偏移网格 $\sigma' = \frac{s \cdot \sigma}{1 + (s - 1)\sigma}$，再通过 `torch.searchsorted` 将每个点吸附到 `training_shift=1.0` 的 1000 步离散训练网格上并去重（`torch.unique_consecutive`），最后将基类 `_shift` 置为 `1.0` 防止二次偏移。
- `SigmaSampler` 在训练期按 `train.sigma.mode` 采样 $\sigma \in (0, 1]$：默认的 `logit_normal` 模式根据 `mu`（`-1.09`）、`sigma`（`0.95`）、截断上限 `cutoff`（`0.95`）与均匀混合比 `uniform_mix`（`0.01`），按累积分布函数 $F(x) = 1 - \Phi\left(\frac{\log((1-x)/x) - \mu}{s}\right)$ 在 $j \in \{1 \dots 1000\}$ 格点（$x = a \cdot j / 1000$）上预构建概率表，同样支持 `uniform` 与 `shifted` 模式。

直接复用官方模块并把格式转换收拢在加载边界，保证了训练前向与官方推理在数学上严格一致；代价是拆分 `qkv` 与反量化 `int8` 文本塔都发生在加载阶段，不支持直接对融合 `qkv` 注入单个联合 LoRA 矩阵。

## 4. 数据层

数据层通过非侵入式的同名 sidecar JSON（`schema: 2`）管理多风格标注，并与逐图尺寸、目录重复倍数和正则集机制协同工作。

### 4.1 扫描与标注优先级

[data.py](<../moss_lora/data.py>) 中的 `scan` 递归遍历 `dataset.path`，识别 `.jpg`、`.jpeg`、`.png`、`.webp` 与 `.bmp`。每张图片的训练文本按四级优先级解析（命中即停）：
1. `dataset.metadata`：若指定了 `metadata.jsonl`，读取其中 `file_name` / `image` / `path` 匹配的 `caption` / `prompt` / `text` 字段（`caption_source="metadata"`）。
2. Sidecar JSON：读取 `<stem>_moss-image-trainer.json` 中由 `dataset.moss_caption_source` 指定的字段（`caption_source="moss"`）。
3. 同名文本文件：按 `dataset.caption_ext`（默认 `[".txt", ".caption"]`）顺序查找同名标注文件（`caption_source="file"`）。
4. 默认回退文本：若上述均为空且提供了默认标注（正则集使用 `dataset.reg_caption`），则采用默认文本（`caption_source="default"`）。最终送入编码器的文本由 `CacheBuilder._caption` 拼接 `dataset.caption_prefix` 与 `dataset.caption_suffix`。

### 4.2 Sidecar v2 结构与双风格生成

提示词生成模块 [captioner.py](<../moss_lora/captioner.py>) 通过 HTTP 调用任意 OpenAI 兼容的本地视觉服务，将结果写入图片旁的 `<stem>_moss-image-trainer.json`。服务探活会先用 3 秒 TCP 连接探测（`_tcp_ok`）拦截丢弃 SYN 包的黑洞端口，再请求 `/models` 确认服务在线，并探测 `/status` 或 `/props` 判断后端是否支持 `json_schema` 受限解码（`--structured auto|on|off`）；对于开启思考模式的模型，`last_json_object` 从响应末尾反向匹配平衡花括号提取 JSON，并将前置推理文本存入 `_analysis`。

Sidecar 文件固定采用 `schema: 2` 结构，容纳两个互不干扰的风格字典：

```json
{
  "schema": 2,
  "tool": "moss-image-trainer",
  "version": 2,
  "created": "2026-10-20T12:00:00+0900",
  "role": "train",
  "tag_caption_style": {
    "caption": "<trigger>, 1girl, solo, standing in a sunlit courtyard, medium shot",
    "kept_tags": ["1girl", "solo"],
    "scene": "standing in a sunlit courtyard, medium shot",
    "reference": "1girl, solo",
    "trigger": "<trigger>",
    "analysis": "Medium shot from eye level...",
    "watermark": [],
    "borders": ["none"],
    "generated": { "model": "...", "temperature": 1.0, "top_p": 0.95, "max_tokens": 16384, "thinking": true }
  },
  "official_style": {
    "caption_long": "<trigger>. Detailed composition...",
    "caption_medium": "<trigger>. One-paragraph summary...",
    "caption_short": "<trigger>. One-sentence summary.",
    "trigger": "<trigger>",
    "analysis": "...",
    "watermark": [{ "box_2d": [920, 800, 980, 990], "label": "signature" }],
    "borders": ["none"],
    "generated": { "model": "...", "temperature": 1.0, "top_p": 0.95, "max_tokens": 16384, "thinking": true }
  }
}
```

两种风格的生成与校验规则如下：
- `tag_caption_style`（对应 CLI `--styles tag_caption`）：服务于保留触发词与 Danbooru 标签的角色微调。`reference_tags` 优先读取同名 `.txt`，若无则回退至 sidecar 上次保存的 `kept_tags`（两者皆无则跳过该图）。`clean_tags` 剥离与触发词重名的标签并去重后填入 `TAGS_PROMPT` 的 `{tags}` 占位符，要求视觉模型先输出 2–4 句 `analysis`（不少于 10 个词），再输出不超过 25 词、仅描述姿态、景别、镜头角度、背景与光影的 `scene`。校验器 `validate_scene` 强制拦截超过 30 词或包含 `hair`、`blonde`、`brunette`、`skin color`、`breast` 及 `<color> eyes` 等身份特征词的 `scene`（当 `role="reg"` 时豁免身份词拦截且不加触发词）。最终 `caption` 由 `compose_caption` 将触发词、`kept_tags` 与 `scene` 以逗号拼接。
- `official_style`（对应 CLI `--styles official`）：复刻官方 recaptioner 规范（`NATURAL_PROMPT`），忽略现有 `.txt`，要求视觉模型输出 `caption.long`（原样保留画面内非水印文字与换行排版）、`caption.medium` 与 `caption.short`（校验词数分别不少于 8、8、3 词），若设置了触发词则以 `<trigger>. <body>` 句号格式前置。两种风格均校验 `borders`（仅允许 `left|right|up|down` 或单独的 `["none"]`）与 `watermark`（`0..1000` 归一化坐标 `[y_min, x_min, y_max, x_max]`，自动修正颠倒坐标并剔除零面积框）。`merge_sidecar` 每次只覆盖当前运行的风格键，通过 `.tmp` 文件与 `os.replace` 原子写入；`migrate_sidecar`（`--migrate`）可将旧版扁平 sidecar 无损升级为 `schema: 2`。

### 4.3 逐图尺寸、Repeats 与正则集

逐图尺寸与采样权重在 [data.py](<../moss_lora/data.py>) 与 [cache.py](<../moss_lora/cache.py>) 中计算：
- 逐图尺寸：`assign_sizes` 给每张图各自的分辨率。`target_size` 保持原图宽高比，把像素面积封在 `dataset.max_area`（默认 `1048576`，不设上限）以内，两边向下对齐到 `ALIGN = 16`；原本就在预算内的图片不放大。没有固定比例集合：自注意力按 token 计算，一个 batch 不需要共用画布，一步里装几张图由 token 预算决定。
- 重复次数（repeats）：子目录名匹配 `^(\d+)[_-](.+)$`（如 `3_concept`）时，基础重复次数设为前缀整数，再乘以全局 `dataset.repeats`。
- 正则集（Regularisation）：设置 `dataset.reg_path` 后，`CacheBuilder` 扫描该目录并将条目设为 `reg=True`，其重复次数由子目录前缀乘以 `dataset.reg_repeat` 决定，用于抑制触发词概念向通用类别漂移。

把生成标注放进独立 sidecar JSON 并按 `dataset.moss_caption_source`（`tag_caption|official_long|official_medium|official_short`）动态选源，让同一份图片目录无需复制即可同时喂给 tag 类训练器与自然语言训练器；代价是目录内文件数量翻倍，且 `tag_caption` 强依赖预先存在的 `.txt` 标签文件。

## 5. 缓存层

缓存层将 VAE 潜变量编码与 Qwen3.5-9B 文本编码拆成两个串行阶段，按样本路径哈希落盘并通过 `index.json`（`schema: 2`）与 safetensors 元数据追踪失效状态。

每个样本的唯一文件名由 `item_uid` 计算：`hashlib.sha1(os.path.abspath(item.image).encode()).hexdigest()[:16] + ".safetensors"`。训练集写入 `cache.dir`（工作区下默认为 `train_cache/`），正则集写入 `cache.reg_dir`（未显式配置时由 `reg_cache_dir` 推导为与 `train_cache` 并排的 `reg_cache/`），物理隔离防止清理或统计时混淆两类样本。

两条编码管线分别由 [cache.py](<../moss_lora/cache.py>) 与 [encode.py](<../moss_lora/encode.py>) 驱动：
1. 潜变量编码（`CacheBuilder.run_latents`）：以 `float32` 加载 FLUX.2 VAE，按各自目标尺寸分组；使用 `ThreadPoolExecutor`（默认 6 线程）在 CPU 上预取图片、双三次插值缩放到该尺寸并归一化至 $[-1, 1]$，与 GPU 编码重叠执行。单批大小优先取 `cache.vae_batch`，未设置时由 [memory.py](<../moss_lora/memory.py>) 的 `latent_batch` 按空闲显存的 80% 除以单图估算开销动态计算，两者都硬性封顶为 `VAE_BATCH_CAP = 4`，规避 WSL2 大批量显存映射停顿。输出潜变量转为 `float16` 存入 `<uid>.safetensors` 的 `latent` 键。
2. 文本特征编码（`run_encode_text`）：通过 [hf_text.py](<../moss_lora/hf_text.py>) 的 `HfTextEncoder` 将每条标注填入固定聊天模板 `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n`（上限 `cache.text_max_length: 4096`，`PAD_ID = 248044`）。编码器支持在加载时通过 `--quantize none|fp8|nf4_ffn|nf4` 在内存中压缩权重。待编码列表先按字符长度排序以减少同批长度差，单批大小优先取 `cache.text_batch`，未设置时由 `memory.text_batch` 动态计算（上限 `TEXT_BATCH_CAP = 64`）；批内采用右侧因果 padding，前向后提取 `last_hidden_state` 与布尔 `attention_mask`，写入同一 `<uid>.safetensors` 的 `text_embeds` 与 `text_mask` 键。由于 Qwen3.5 为因果注意力，右侧 padding 不会污染前方有效 token 的隐状态。若配置了 `preview.prompts`，同阶段还会将正负预览提示词（空负向提示词同样经过同一聊天模板编码为 10 个 token）写入 `train_cache/preview.safetensors`。

每次写盘时，`CacheBuilder._write` 与 `run_encode_text` 都会把 `image`、`caption`（含前后缀的完整训练文本）、`width`、`height`、`repeats` 写入 safetensors 文件头部的 `metadata`，最后生成 `train_cache/index.json`：

```json
{
  "schema": 2,
  "cache_dir": "<WORKSPACE>/train_cache",
  "reg_dir": "<WORKSPACE>/reg_cache",
  "caption_source": "tag_caption",
  "batch": { "latent": 4, "text": 64 },
  "items": [
    {
      "file": "<uid>.safetensors",
      "image": "<DATASET_DIR>/001.png",
      "caption": "...",
      "width": 1024,
      "height": 1024,
      "repeats": 1,
      "reg": false,
      "text_len": 42
    }
  ]
}
```

对于正则集条目，`file` 字段保存相对于 `train_cache` 的路径（即 `../reg_cache/<uid>.safetensors`），因此训练器只需拼接 `os.path.join(cache_dir, entry["file"])` 即可统一读取。

这份索引必须记下 batch，因为 batch 会改变数值：同一段提示词单独编码与放在 64 条里编码，进入 GPU 的矩阵形状不同、落到不同 kernel 上，结果会差出 `1e-4` 量级。`batch.latent` 由 `CacheBuilder.run_latents` 写入，`batch.text` 由 `run_encode_text` 写入；只跑其中一个阶段时，另一个阶段已记录的值原样保留而不是被抹掉。早于该字段生成的索引没有 `batch`，这就是它诚实的答案：没人记过。

缓存失效与增量重编码遵循三条判定规则：
- 潜变量增量判定：`CacheBuilder._pending("latent")` 检查目标 `<uid>.safetensors` 是否存在且包含 `latent` 张量；已存在且 `cache.overwrite: false` 时直接跳过。
- 文本特征增量与过期判定：`run_encode_text` 逐条读取已缓存 safetensors 头部的 `metadata["caption"]`，与当前配置计算出的目标文本比对。当用户切换 `dataset.moss_caption_source`、修改 `caption_prefix` / `caption_suffix` 或重写了 sidecar JSON 时，文本不匹配会触发该条目的单独重编码，而文件中的 `latent` 张量原样保留，无需重新运行 VAE。
- 训练前就绪校验：[cache_cmd.py](<../moss_lora/web/services/cache_cmd.py>) 的 `_cache_summary` 要求 `index.json` 存在、`schema == 2` 且所有条目的 `text_len > 0`；任一条件不满足时 `ready` 为 `false`，WebUI 与 CLI 均拒绝启动训练。

将 `latent` 与 `text_embeds` 合并在同一个 `<uid>.safetensors` 中，使训练期每步只需一次磁盘打开操作，同时通过头部元数据实现了「改提示词只重跑文本编码」的细粒度增量更新；代价是 `dataset.max_area` 一旦改动，所有已缓存的 latent 都作废，因为 `_pending("latent")` 仅检查张量键存在性而不校验潜变量尺寸；索引里记录了 `sizing` 指纹（`mode`、`area`、`align`），缓存构建器据此自动重编 latent，训练器读到指纹不一致会直接拒绝加载。

## 6. 训练层

训练层围绕标准 Flow Matching 速度场目标构建，通过显存预设档位组合 PEFT LoRA、NF4 冻结层量化、CPU 块交换与三种批处理注意力路径。

### 6.1 训练目标与批处理模式

[trainer.py](<../moss_lora/trainer.py>) 实现的损失函数严格对齐官方技术报告：对干净潜变量 $x_0$ 与标准正态噪声 $\epsilon \sim \mathcal{N}(0, I)$，按采样得到的 $\sigma \in (0, 1]$ 构造线性插值 $x_t = (1 - \sigma) x_0 + \sigma \epsilon$，将缩放时间步 $t = 1000\sigma$ 与文本特征喂入 DiT 预测速度场 $v_\theta(x_t, 1000\sigma, c)$，计算其与目标速度 $v = \epsilon - x_0$ 在图像 token 上的均方误差 `F.mse_loss(prediction.float(), noise - x0.float())`。不加任何 SNR 权重。

每个微批次都由 `_sample_by_tokens` 组装：按随机顺序累加样本，直到 token 预算（`train.bin_tokens`，为 `0` 时取 $\text{max\_area} / 256 + 1024$，即一张满尺寸图加文本）用尽，每张样本按 `repeats` 加权。样本各自保留分辨率与文本长度；`train.batch_mode` 决定它们怎么合并，配合 `train.attention: auto|flash|sdpa`：
- `auto`（默认）：FlashAttention-2 可用时取 `packed`，否则取 `padded`。
- `packed`：`forward_packed` 把样本拼成一条序列并共享单个标量时间步调制。FA2 可用时构造 `cu_seqlens` 调用 `flash_attn_varlen_func`，零填充零冗余计算；不可用时用稠密块对角布尔掩码把各样本隔开，交由 SDPA 计算。
- `padded`：`forward_padded` 先把每个样本展平为一维 token 序列，再在序列维补齐到批内最大长度 $L_{\max}$ 并附上 `token_mask` 交由 SDPA 处理，避免按二维画布补齐造成的面积浪费。

### 6.2 LoRA 注入、优化器与调度

`Trainer._attach_lora` 使用 `peft.get_peft_model` 注入适配器，正则模式固定为 `^layers\.<layer_expr>\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)$`（由 `lora.targets` 与 `lora.layers` 控制），避免误匹配 `text_embedder` 内部同名层；仅当 `lora.include_text_embedder: true` 时才追加 `text_embedder\.blocks\.\d+\.(qkv|proj|ffn_up|ffn_gate|ffn_down)$`。默认使用 Kaiming 初始化，`lora.init: zero` 时将 `lora_B` 置零。

`train.optimizer` 支持四类配置：
- `adamw`（默认）：`torch.optim.AdamW`，固定 $\beta=(0.9, 0.95)$。
- `adamw8bit`（或 `vram.optimizer_in_8bit: true`）：调用 `bitsandbytes.optim.AdamW8bit` 压缩一阶与二阶动量。
- `muon`：优先导入官方 `muon.SingleDeviceMuon`，不可用时回退至内置 [muon.py](<../moss_lora/muon.py>)。内置 `Muon` 对二维 LoRA 参数矩阵维护单动量缓冲（$\mu=0.95$，Nesterov 动量），经 5 步 Newton-Schulz 迭代（系数 $3.4445, -4.7750, 2.0315$）正交化更新方向，并按 $0.2 \sqrt{\max(m, n)}$ 缩放幅值以对齐 AdamW 学习率量级，优化器状态显存减半。
- `adafactor`：`transformers.Adafactor`（`scale_parameter=False, relative_step=False`）。

学习率由 `_lr_at` 按步更新：`train.warmup` 为 `(0, 1)` 之间的小数时按总步数比例折算预热步数，大于等于 1 时直接作为步数；预热结束后按 `train.lr_scheduler`（`constant` 或 `cosine`）衰减。每步经 `grad_accum` 次反向传播累加后，按 `train.max_grad_norm`（默认 `1.0`）裁剪梯度并更新可选的 FP32 指数滑动平均字典（`train.ema`）。

### 6.3 精度、量化、块交换与 OOM 自愈

显存控制集中在 [config.py](<../moss_lora/config.py>) 的 `VRAM_PRESETS`（`auto|32|24|16|12|8|6`）、[quant.py](<../moss_lora/quant.py>) 与 [offload.py](<../moss_lora/offload.py>)：

| `vram.preset` | `gradient_checkpointing` | `quantize_dit` | `blocks_to_swap` | `optimizer_in_8bit` |
| --- | --- | --- | --- | --- |
| `32` / `24` / `16` | `true` | `none` | `0` | `false` |
| `12` | `true` | `ffn` | `0` | `false` |
| `8` | `true` | `all` | `0` | `false` |
| `6` | `true` | `all` | `20` | `true` |

各项机制的底层行为如下：
- 精度与累加器：`model.dtype` 默认 `bfloat16`；设为 `float16` 时自动启用 `GradScaler`，若同时开启 `vram.fp16_accumulate: true`，则设置 `torch.backends.cuda.matmul.allow_fp16_accumulation = True` 以换取消费级显卡更高的半精度 Tensor Core 吞吐。
- NF4 冻结层量化：`vram.quantize_dit` 设为 `ffn` 或 `all` 时，`quantize_nf4` 将 `layers.*` 下不含 `lora_` 的冻结 `nn.Linear` 替换为 `bitsandbytes.nn.Linear4bit`（`quant_type="nf4"`, `compress_statistics=True`），跳过非块模块（嵌入层与调制层）。
- CPU 块交换（Block Swap）：`vram.blocks_to_swap > 0`（要求开启 `gradient_checkpointing`）时，`BlockSwap` 将最后 $N$ 个 Transformer 块的冻结参数移至主机锁页内存（`pin_memory()`），而可训练的 LoRA 参数始终留在 GPU。自定义 `torch.autograd.Function`（`_SwapCheckpoint`）在前向与反向重算进入该块前调用 `non_blocking=True` 拷入 GPU，计算完毕立即释放回 CPU，并用零张量 `_dummy` 维持计算图连通，使 GPU 上同一时刻最多只驻留一个被交换块的基座权重。
- OOM 动态降级：训练循环捕获 `torch.cuda.OutOfMemoryError` 后调用 `_degrade_memory`，自动将 `blocks_to_swap` 增加 8 层（上限 40 层），清空 CUDA 缓存后立即重试当前步。
- 编译加速：`vram.compile: true` 时对 `model.forward` 施加 `torch.compile(..., dynamic=True, fullgraph=False)`，编译失败则自动回退至 eager 模式。

### 6.4 验证、采样、检查点与断点续训

- 确定性留出集验证：`train.eval_holdout > 0`（默认 `0.05`）时，`_split_eval` 计算每张图像绝对路径的 SHA1 前 8 位十六进制值，按模 1000 切分验证集。哈希切分保证跨运行、跨中断续训的验证样本集合恒定；每 `train.eval_every` 步使用固定种子 `1234` 计算前 `train.eval_batches` 批的验证损失，追加写入 `metrics.jsonl`。
- 周期采样与轨迹拼图：每 `train.sample_every` 步调用 [sampling.py](<../moss_lora/sampling.py>) 的 `sample` 生成预览图，训练结束时由 `make_contact_sheet` 按「每步一行、每条提示词一列」拼接为 `samples/trajectory.png`。采样异常被捕获并记录日志，绝不中断训练主循环。
- 检查点与状态分离：`save_checkpoint` 每 `train.save_every` 步写出仅含 LoRA 权重且直接采用 `lora_unet_*` 键名的 `checkpoints/stepXXXXXX.safetensors`（结束时写 `final.safetensors`），并在头部 `metadata` 写入步数与配置摘要。优化器动量和 CPU/CUDA 随机数状态则根据 `train.save_state`（`off|last|all`，默认 `last`）单独写入滚动更新的 `checkpoints/state.safetensors` 或按步保存的 `<tag>.state.safetensors`。
- 断点续训与队列：传入 `--resume <ckpt>` 时，`_load_checkpoint` 通过 `_denormalise_keys` 将 `lora_unet_*` 键名逆映射回 PEFT 参数名，随后查找同目录匹配的 `.state.safetensors` 或 `state.safetensors`，校验 `state_step == step` 后恢复优化器动量与 RNG 状态。[queue.py](<../moss_lora/queue.py>) 的 `run_pipeline` 可扫描工作区内所有 `metrics.jsonl` 未达目标步数的运行，先补齐缺失缓存，再按各自 `runs/<run>/config.yaml` 逐一续训。

这套设计通过分离权重与优化器状态，把每个检查点体积压缩了三分之二且立即可用于推理；代价是在默认的 `save_state: last` 下，只有最新一个检查点拥有匹配的优化器动量，若指定从更早的历史步数 `--resume`，训练器只能恢复模型权重而优化器动量将从零重建。

## 7. 导出

导出层直接以 ComfyUI 原生支持的 kohya `lora_unet_*` 格式作为内部检查点标准格式，并提供无需加载 5B 模型的亚秒级格式转换与批量打包。

训练器在保存检查点与最终导出时，统一将 PEFT 模块路径 `base_model.model.layers.<L>.proj.<M>` 转换为社区通用的 kohya 键名布局：

```text
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_down.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_up.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.alpha
```

ComfyUI 的标准 `Load LoRA` 节点会将 `lora_unet_` 前缀与下划线路径自动还原并挂载到 `diffusion_model.layers.<L>.proj.<M>.weight`（即使底层 ComfyUI 模型使用融合 `qkv`，其内置加载器也会将拆分的 `q`/`k`/`v` LoRA 切片注入对应通道）。当用户请求 `diffusers` 格式（`export.formats: [diffusers]`）时，键名则写为 `transformer.layers.<L>.proj.<M>.lora_A.weight` 与 `.lora_B.weight`。若只导出一种格式，输出文件名为 `<name>.safetensors`；同时导出两种格式时自动追加 `_kohya` 与 `_diffusers` 后缀。

离线导出与批量导出由 [lora_file.py](<../moss_lora/lora_file.py>) 与 [export_batch.py](<../moss_lora/export_batch.py>) 实现：
- 免模型转换：`lora_file.convert` 直接在 CPU 上打开检查点 `.safetensors`，用正则 `^lora_unet_layers_(\d+)_proj_(q|k|v|o|ffn_up_gate|ffn_down)\.lora_(down|up)\.weight$` 解析出层号与投影名，重命名键名后直接写盘，全程不实例化 5B DiT。当目标格式仅为 `kohya` 时，`cli.py export` 与 `export_batch` 直接执行 `shutil.copyfile` 原样复制检查点。
- 步数区间批量导出：`export_batch.parse_range` 支持解析步数区间语法（如 `250-2000:250`）、逗号列表（`250,500,1000`）、单步数字、`final`（内部以哨兵值 `FINAL = 0` 表示）或空串（导出全部现有步数）。指定 `--state` 时一并复制 `state.safetensors`，指定 `--zip` 时以 `ZIP_STORED`（仅打包不二次压缩浮点数张量）生成归档包。

将检查点直接存为 `lora_unet_*` 让「训练中保存的任意步检查点」本身就是成品 LoRA，导出操作退化为文件复制或纯字典键重命名；代价是 `_kohya_module` 只能按显式词表解析下划线路径，`layers.<n>.proj.<模块>` 与 `text_embedder.blocks.<n>.<模块>` 之外的新模块名会被跳过，改动适配器作用范围时必须同步更新两张正则表。

## 8. WebUI 后端

WebUI 后端采用「路由层 → 服务层 → 共享状态与任务注册表」的三层 FastAPI 结构，通过全局单任务锁、SSE 事件流与磁盘文件回退机制兼顾实时性与进程重启恢复。

代码按职责切分为五个部分：
- 应用装配（[server.py](<../moss_lora/web/server.py>) 与兼容入口 [app.py](<../moss_lora/web/app.py>)）：`create_app` 按固定顺序注册 11 个 `APIRouter`（[i18n.py](<../moss_lora/web/routers/i18n.py>)、[system.py](<../moss_lora/web/routers/system.py>)、[settings.py](<../moss_lora/web/routers/settings.py>)、[workspaces.py](<../moss_lora/web/routers/workspaces.py>)、[dataset.py](<../moss_lora/web/routers/dataset.py>)、[config.py](<../moss_lora/web/routers/config.py>)、[jobs.py](<../moss_lora/web/routers/jobs.py>)、[export.py](<../moss_lora/web/routers/export.py>)、[files.py](<../moss_lora/web/routers/files.py>)、[validate.py](<../moss_lora/web/routers/validate.py>)、[samples.py](<../moss_lora/web/routers/samples.py>)），随后调用 `_flatten_included_routes` 将子路由展开为顶层扁平路由表，确保字面量路径（如 `/api/i18n/template`、`/api/i18n/current`）严格排在通配路由 `/api/i18n/{code}` 之前，最后将 `moss_lora/web/static` 挂载到根路径 `/`。
- 共享状态（[deps.py](<../moss_lora/web/deps.py>)）：启动时首先检查 `sys.prefix/pyvenv.cfg`，若非虚拟环境立即报错退出。模块内持有由 `threading.RLock` 保护的全局配置快照 `SETTINGS`、`RESOLVED`、`FIRST_RUN` 以及仅驻留内存的明文视觉服务密钥 `_LITERAL_API_KEY`。所有文本落盘统一调用带 PID 与随机后缀的 `_write_text` 原子替换函数。
- 任务模型与生命周期（[jobs.py](<../moss_lora/web/jobs.py>) 与 [launcher.py](<../moss_lora/web/services/launcher.py>)）：全局单例 `JOBS` 维护内存中所有 `Job` 对象。`JOBS.active()` 实施全项目单任务互斥锁：同一时刻只允许一个 `train`、`cache` 或 `caption` 任务处于 `running` 状态，并发发起新任务会收到 `HTTP 409`。后台守护线程运行 `_run(job)`，先顺序执行 `pre_commands`（如训练前的无 LoRA `baseline` 采样，失败只发 `warning` 事件而不阻断主训练），再通过 `subprocess.Popen(..., start_new_session=True)` 启动主命令。`_stream_command` 实时读取子进程合并后的 stdout，将原始行写入 `logs/<log_name>.log`，同时用 `STEP_RE` 解析训练步进日志并推入 `job.events` 队列（上限 5000 条，溢出时每次裁剪前 1000 条）；训练任务正常结束后，`_write_preview_sidecars` 为每张采样图生成 `<filename>.json` 与 `seeds.json`。停止任务时，`_terminate` 向整个进程组发送 `SIGTERM`（`os.killpg`），确保 `bash -c` 包装下的子进程一并退出。
- 实时推送与 CLI 任务发现（[routers/jobs.py](<../moss_lora/web/routers/jobs.py>) 与 [routers/workspaces.py](<../moss_lora/web/routers/workspaces.py>)）：`GET /api/jobs/{job_id}/events?from_index=N` 提供 `text/event-stream` SSE 流，每 0.3 秒轮询新增事件，空闲满 15 秒发送 `: keep-alive` 注释帧，支持前端断线后按已收事件下标无损续订。对于用户在终端直接用 CLI 启动的训练，后端无需经过 `JOBS`：训练器每 20 步更新 `runs/<run>/training.json` 心跳文件（含 `pid`、`step`、`total`、`time`），`/api/workspaces/{name}/active` 与 `/stop` 会结合心跳文件与 `/proc/<pid>/cmdline` 扫描识别并控制外部 CLI 训练进程。
- 静态与媒体服务（[services/dataset.py](<../moss_lora/web/services/dataset.py>) 与 [routers/files.py](<../moss_lora/web/routers/files.py>)）：`/api/thumb` 使用 Pillow 按原图路径、修改时间、字节大小与目标宽度（默认 `384px`）的 SHA1 生成 WebP 缩略图，样图缩略图存于 `runs/<run>/.thumbs/`（随 run 删除自动清理），数据集缩略图存于工作区 `.thumbs/`，响应头固定带 `Cache-Control: no-cache`。`/api/files` 以白名单后缀只读提供原图与 `.safetensors` 下载，并附带 `Content-Disposition: attachment; filename=...` 防止浏览器丢失扩展名。

内存事件队列配合成对的 `logs/*.log` 与 `metrics.jsonl` 落盘，使页面在 SSE 连接正常时获得毫秒级推送，而在浏览器刷新或 Web 服务重启后仍能从磁盘完整还原历史曲线与日志；代价是 `JOBS` 注册表不跨进程持久化，Web 服务重启期间正在运行的后台子进程会脱离内存句柄（需依赖 `/proc` 与心跳文件接管）。

## 9. WebUI 前端

前端采用零构建步骤、零外部 npm 依赖、零 CDN 的原生浏览器 ES module 架构，通过运行时 HTML 片段注入与双向隔离的状态机管理九个功能标签页。

### 9.1 外壳、片段注入与启动引导

入口文件 [index.html](<../moss_lora/web/static/index.html>) 只保留应用外壳：内联 SVG 图标雪碧图、顶栏（品牌标识、GPU 监控 chip、全局任务状态 chip、语言切换器、9 个 tab 导航链接）、全局活动进度条 `#activity`、`<main id="main">` 挂载点以及 8 个原生 `<dialog>` / 模态框。样式表按设计 token（`css/tokens.css`）、基础组件与页面专属样式拆分为独立 CSS 文件，使用系统原生字体栈与深色主题。

启动引导（[main.js](<../moss_lora/web/static/js/main.js>) 中的 `boot`）严格按时序执行：
1. 调用 [fragments.js](<../moss_lora/web/static/js/core/fragments.js>) 的 `mountFragments(TABS)`，并发 `fetch` 拉取 [dashboard.html](<../moss_lora/web/static/pages/dashboard.html>)、[train.html](<../moss_lora/web/static/pages/train.html>)、[dataset.html](<../moss_lora/web/static/pages/dataset.html>)、[cache.html](<../moss_lora/web/static/pages/cache.html>)、[samples.html](<../moss_lora/web/static/pages/samples.html>)、[export.html](<../moss_lora/web/static/pages/export.html>)、[workspace.html](<../moss_lora/web/static/pages/workspace.html>)、[settings.html](<../moss_lora/web/static/pages/settings.html>) 与 [caption.html](<../moss_lora/web/static/pages/caption.html>) 九个片段，通过 `<template>` 解析并校验根节点 ID 为 `#tab-<name>` 后，依序插入 `#main` 内第一个 `<dialog>` 之前。
2. 调整中控台面板 DOM 顺序，调用 [events.js](<../moss_lora/web/static/js/events.js>) 的 `bindEvents()` 一次性绑定全部事件监听器，初始化 Canvas 图表。
3. 单独等待 `loadI18n()` 完成词典加载并渲染顶栏语言选择器，防止并发初始化时顶栏闪现未翻译的裸 key。
4. 并发拉取 `/api/settings`、`/api/workspaces`、`/api/system`、`/api/jobs` 与 `/api/presets`，随后加载当前工作区配置，执行 `applyI18n(document)` 并根据 `location.hash` 路由至对应标签页；若后端标记 `first_run: true` 则弹出五步首次使用引导对话框。
5. 将核心状态与操作函数挂载到 `window.__mossUI`，供无头浏览器回归测试脚本直接调用。

`static/` 下的所有文件由 [server.py](<../moss_lora/web/server.py>) 的 `UncachedStaticFiles` 提供，响应带 `Cache-Control: no-store`，且不带 `ETag` 与 `Last-Modified`。URL 里没有内容哈希，一旦允许条件请求，`304` 就会让上一版模块图继续存活，与刚写好的 HTML 拼在一起；这个窗口里新增导出的模块会链接失败，页面于是白屏。刷新必须拿到磁盘上的文件。

### 9.2 双模表单状态机与 i18n

- 表单与 YAML 双模互斥状态机（[form.js](<../moss_lora/web/static/js/config/form.js>)、[model.js](<../moss_lora/web/static/js/config/model.js>) 与 [yaml.js](<../moss_lora/web/static/js/config/yaml.js>)）：全局状态（[state.js](<../moss_lora/web/static/js/core/state.js>)）中的 `state.mode` 只有 `form` 与 `yaml` 两种取值。在表单模式下，36 个 `FIELD_DEFS` 控件与层范围选择器直接读写内存中的 `state.config` 对象，右侧 YAML 预览框为只读投影（由自研的缩进子集序列化器 `dumpYaml` 生成）；切换至「高级 YAML」模式后，左侧表单控件全部禁用，YAML 文本框成为唯一数据源（由 `parseYaml` 实时解析并定位行列语法错误），彻底杜绝双源双向同步互相覆盖。
- 多语言词典与无刷新切换（[core/i18n.js](<../moss_lora/web/static/js/core/i18n.js>) 与后端 [i18n.py](<../moss_lora/i18n.py>)）：内置简体中文（`zh`）、English（`en`）、日本語（`ja`）、한국어（`ko`）四套词典（`static/i18n/<code>.json`），支持在设置页导出模板并将自定义或覆盖词典上传至项目根 `i18n/<code>.json`。后端 `i18n.merged` 按「用户词典 $\rightarrow$ 内置目标语言词典 $\rightarrow$ 内置中文基准词典」三层合并，前端 `t(key, vars)` 若仍未命中则回退为原始 `key` 字符串并替换 `{name}` 占位符。`applyI18n` 自动扫描 DOM 上的 `data-i18n`、`data-i18n-title`、`data-i18n-placeholder` 与 `data-i18n-aria` 属性，切换语言时无需刷新页面即可重绘全部静态标签、图表图例与状态提示。

### 9.3 三条前端硬约定

1. 禁止顶层跨模块调用：所有 `static/js/**/*.js` 的模块顶层只允许声明常量、纯数据结构与函数定义，严禁在模块加载期执行 DOM 查询（`$()`）、网络请求或调用其他模块的初始化函数；一切副作用必须由 `main.js` 的 `boot()` 在 `mountFragments` 完成后按序触发，避免循环依赖与空节点引用。
2. 页面 markup 归口 `static/pages/`：九个标签页的 HTML 结构必须存放在 `moss_lora/web/static/pages/<tab>.html` 中，禁止塞回 `index.html`，也禁止在 JS 中拼接大段静态页面骨架。
3. 单文件 700 行上限：任何前端 JS 模块与后端 Python 模块不得超过 700 行（当前最大文件为 [events.js](<../moss_lora/web/static/js/events.js>)，693 行）；新增功能必须按职责拆入 `core/`、`config/`、`components/` 或 `pages/` 子模块。

无构建步骤的原生 ES module 让开发者修改任意 JS/HTML/CSS 后刷新浏览器即可生效，彻底消除了 Node.js 打包链与静态产物失步问题；代价是没有 TypeScript 静态类型检查与构建期摇树优化，且手写的 YAML 子集解析器（[yaml.js](<../moss_lora/web/static/js/config/yaml.js>)）不支持多行折行标量或锚点引用（切换回表单模式重新序列化时会丢弃 YAML 注释）。

## 10. 配置系统

配置系统将机器环境参数（`settings.yaml`）与实验超参严格分离，并通过显存预设与预检接口在启动前拦截配置错误。训练配置只有一个家：`runs/<run>/config.yaml`，是完整配置而不是覆盖层。新运行从 `Config()` 默认值开始，或复制另一次运行的配置，此后不依赖其它文件。训练进行中拒绝写入，运行结束后可改。

两层配置文件的职责边界如下：
- 机器级配置 `settings.yaml`（由 [settings.py](<../moss_lora/settings.py>) 管理）：位于仓库根目录，被 `.gitignore` 忽略；首次启动时由 `ensure_settings_file` 从 [settings.example.yaml](<../settings.example.yaml>) 自动复制生成。文件仅包含四个顶层段：
  - `paths.workspaces`：工作区根目录（默认 `workspaces`，相对仓库根目录解析）。
  - `runtime`：CUDA 缓存分配器策略 `allocator`（`expandable|async|off`，默认 `expandable`）与回收阈值 `gc_threshold`（默认 `0.9`）。`alloc_conf_for` 会为训练任务注入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.90`（或 `backend:cudaMallocAsync`），但对 `encode-text` 与 `encode-prompt` 强制清空该变量，防止一次性拷贝 9B 权重时在部分驱动或 WSL2 下触发 GPU 映射失败。
  - `caption`：OpenAI 兼容的本地视觉服务配置，含 `base_url`、`api_key`、`model`、`trigger`、`workers`（`1..32`，默认 `6`）、`retries`（`1..10`，默认 `3`）、`temperature`（默认 `1.0`）、`top_p`（默认 `0.95`）、`max_tokens`（默认 `16384`）、`thinking`（默认 `true`）及自定义模板 `prompt_tags` / `prompt_natural`（与内置模板相同时自动存为空串）。`api_key` 遵循安全持久化规则（见 [caption.py](<../moss_lora/web/services/caption.py>) 与 [settings_svc.py](<../moss_lora/web/services/settings_svc.py>)）：若填入形如 `^[A-Z_][A-Z0-9_]{0,63}$` 的环境变量名且环境中存在该值，则将变量名写入 `settings.yaml`；若直接填入明文密钥且通过 `/models` 探测，则只存入进程内存 `deps._LITERAL_API_KEY`，`settings.yaml` 中写空串；向提示词生成子进程传递密钥时一律通过子进程环境变量注入，绝不写入命令行参数或日志文件。
  - `ui.language`：当前界面语言代码（正则 `^[a-z][a-z0-9-]{1,7}$`，默认 `zh`）。
- 每次运行自带的 `workspaces/<WORKSPACE>/runs/<run>/config.yaml`（由 [config.py](<../moss_lora/config.py>) 管理）：映射为嵌套 dataclass `Config`，下辖 `name`、`output_dir` 以及 `model`（`ModelConfig`）、`dataset`（`DatasetConfig`）、`cache`（`CacheConfig`）、`lora`（`LoraConfig`）、`train`（`TrainConfig`，内嵌 `sigma: SigmaConfig`）、`vram`（`VramConfig`）、`export`（`ExportConfig`）、`preview`（`PreviewConfig`）八个子段。从 `runs/<run>/config.yaml` 加载时，`Config.load` 将工作区目录设为 `_base_dir`，使 `dataset.path`、`dataset.reg_path`、`cache.dir`（留空默认 `train_cache`）、`cache.reg_dir` 与 `output_dir`（留空默认 `runs`）全部相对于该工作区目录解析；Web 端保存时，`_portable_workspace_config` 还会将指向工作区内部的绝对缓存和输出路径自动重写回相对名称，保证整个工作区文件夹可整体移动或备份。`${VAR}` 环境变量表达式仅在该变量非空时展开，未定义时原样保留以便校验器报错，避免静默展开为根目录 `/...`。

校验与预设机制贯穿前后端与 CLI：
- 静态与运行时校验：`Config.validate()` 检查必填路径（`dataset.path`、`model.dit`、`model.vae` 必须存在；`model.text_encoder` 可为空以支持纯缓存训练，非空时须为含 `config.json` 的目录或 `.safetensors` 文件）、枚举合法性（`dataset.moss_caption_source`、`train.attention`、`train.batch_mode`）、数值范围（`lora.rank > 0`、`lora.layers` 与 `preview.uncond_skip_layers` 落在 `0..41`、`preview.width` 与 `preview.height` 为 16 的正整数倍）。Web 端 `POST /api/validate`（[validate.py](<../moss_lora/web/routers/validate.py>)）在此基础上叠加缓存就绪检查，并返回 `next_tab` 与 `focus_field` 引导前端自动跳转到首个出错控件。
- 显存预设与 CLI 规划：`apply_preset` 根据 `vram.preset`（`32|24|16|12|8|6`）或 `detect_vram_gb()` 自动探测的显卡容量填充 `gradient_checkpointing`、`blocks_to_swap`、`optimizer_in_8bit` 与 `quantize_dit`。CLI 子命令 `python -m moss_lora.cli plan --vram <GB> --config workspaces/<WORKSPACE>/runs/<RUN>/config.yaml --write workspaces/<WORKSPACE>/runs/<RUN>/config.yaml` 则扣除采样预留余量后，在内置基准表 `MEMORY_TABLE` 中挑选满足显存预算且单步最快的精度、NF4 量化与块交换组合写回配置文件。

两层配置分离配合相对路径解析，确保工作区配置不含任何本机敏感信息，而将 `settings.yaml` 排除在版本控制之外则保护了本机目录结构与 API 密钥；代价是同一工作区下的每次运行共享同一个 `train_cache/`，若某次运行的配置改动了分辨率或提示词来源，启动训练前必须手动重跑缓存。

## 11. 测试与验收

仓库配备了无需 GPU 的单元测试、可选的编码器数值一致性测试、零依赖的 CDP 前端回归夹具以及强制性的文档脱敏扫描。

自动化检查脚本位于 `tests/` 目录：
1. Sidecar 读写与迁移单测（[test_sidecar.py](<../tests/test_sidecar.py>)）：在临时目录中验证 `schema: 2` 的单风格写入、双风格无损合并、单风格覆盖更新、`reference_tags` 的 `.txt` 与历史 `kept_tags` 优先级回退，以及旧版扁平 JSON 向 `tag_caption_style` / `official_style` 的自动迁移与校验。
2. 多语言词典完整性单测（[test_i18n.py](<../tests/test_i18n.py>)）：以内置中文包 `zh.json` 为基准，检查 `zh`、`en`、`ja`、`ko` 四套内置语言包的键集合完全相等、无空值翻译，且每条译文中的 `{placeholder}` 占位符集合与中文基准完全一致。
3. 视觉服务密钥规则单测（[test_caption_key.py](<../tests/test_caption_key.py>)）：在不联网条件下验证 `_plan_api_key` 与 `resolve_api_key` 的五种状态分支：环境变量名落盘并在运行时解出真值、明文密钥仅存进程内存 `deps._LITERAL_API_KEY` 不写文件、空串清空、旧版文件中的非变量名字符串按 `legacy` 兼容读取、已存变量名但进程环境未设置时返回 `missing` 阻止误将变量名当作密钥发送。
4. 文本编码器右 padding 隔离测试（[test_text_padding.py](<../tests/test_text_padding.py>)）：设置环境变量 `MOSS_TE=<MODELS_DIR>/text_encoders/Qwen3.5-9B` 后运行，对比短提示词单独编码（`batch_size=1`）与跟长提示词混批右 padding 编码（`batch_size=2`）的输出向量，断言余弦相似度大于 `0.999`，验证因果掩码未发生右侧 padding 泄漏。
5. 静态资源缓存头契约（[test_static_cache.py](<../tests/test_static_cache.py>)）：以 ASGI 传输构建应用，断言 `/`、ES 模块与页面片段在请求携带 `If-None-Match` 或未来时间戳的 `If-Modified-Since` 时仍返回 `200` 且带 `Cache-Control: no-store`，响应不返回任何校验器头，API 路由仍优先匹配于挂在 `/` 上的静态挂载，缺失模块仍返回 `404`。

前端回归测试与拆分等价性检查由 [smoke.mjs](<../tests/ui/smoke.mjs>) 及路由展平机制保障：
- 后端拆分等价性：[server.py](<../moss_lora/web/server.py>) 通过 `_flatten_included_routes` 将拆分后的 11 个子路由原地展开进 `app.router.routes`，保证路由匹配顺序与拆分前的单文件 `app.py` 逐条一致，同时保留 [app.py](<../moss_lora/web/app.py>) 作为兼容导出垫片。
- 前端 CDP 冒烟回归：[smoke.mjs](<../tests/ui/smoke.mjs>) 使用 Node.js 原生 ESM 与 WebSocket 直连 Chromium DevTools Protocol（CDP），不依赖 Playwright 或 Puppeteer。在 `1440×1000` 固定视口下，脚本依次调用 `POST /api/i18n/current` 切换 `zh`、`en`、`ja`、`ko` 四种语言，逐一导航至全部 9 个 tab，等待 DOM 结构签名连续稳定 700ms 后注入 `browserMetrics` 函数，自动采集并断言四项指标：
  1. 运行时异常：`Runtime.exceptionThrown`、控制台 `error/warning/assert` 与非 favicon 的 `Network.loadingFailed` 必须为零。
  2. 横向溢出：`document.documentElement` 与当前可见 `.tabpane` 的 `scrollWidth - clientWidth` 必须为 `0`。
  3. 意外文本截断：沿着文本节点向上查找 `overflow: hidden|clip` 且发生溢出的容器，排除显式设置了 `text-overflow: ellipsis`、`-webkit-line-clamp` 或带有 `title` 悬停提示的节点后，`clippedText` 必须为零。
  4. 空翻译与漏翻裸 key：所有可见 `[data-i18n*]` 属性目标不得为空串，且非代码区（排除 `pre`、`code`、`.mono`）不得出现形如 `nav.dashboard` 的点分裸 key。

对外文档脱敏由 [test_docs_sanitized.py](<../tests/test_docs_sanitized.py>) 守卫。脚本扫描 `README*.md`、`AGENTS.md`、`settings.example.yaml` 与 `docs/`、`skills/` 下的文本文件，用两组正则拦截违规内容：`HARD` 规则拦截本机绝对与挂载路径、凭据形态（命中即返回非零退出码）；`SOFT` 规则对显存实测值、单步耗时/吞吐率、温度与显卡型号给出告警。与具体机器有关的词表放在不跟踪的本地清单里，脚本存在时加载、不存在时只用通用规则。每次修改文档后必须运行该脚本并确认 `0 hard hit(s), 0 warning(s)`。

## 12. 已知限制与后续方向

当前架构在多样本批处理、低显存数值精度与微调覆盖范围上存在明确边界，后续演进将围绕变长打包与新模态适配展开。

当前版本存在四项主要限制：
1. 微批的粒度是 token 预算：一个批次装到预算用尽为止，因此装几张图取决于抽到的样本的宽高比与文本长度。预算约束的是计算量而不是张数，这让单步开销可预期，但也使 `batch_size` 失去意义；真正的旋钮是 `train.grad_accum` 与 `train.bin_tokens`。
2. 每个微批共享时间步：把多个样本拼进一次前向，要求它们共用一个标量时间步 $\sigma$（即共用一份 `shared_block_modulation`）。这比每样本独立采样的梯度方差略高，需要通过步数或梯度累积平滑。
3. 低显存档位的精度与画质折衷：在 `model.dtype: float16` 下开启 `vram.fp16_accumulate: true` 会降低矩阵乘累加精度，在高学习率或长步数训练中比 `bfloat16` 更容易出现梯度溢出；面向小显存显卡的 NF4 基座量化（`vram.quantize_dit: ffn|all`）会引入静态量化误差，对极细粒度笔触与复杂排版文字的还原度略逊于全精度 `bfloat16` 基座。
4. 适配器作用范围：训练器不支持微调 Qwen3.5-9B 文本编码器与 FLUX.2 VAE，只能训练 DiT 与可选 `text_embedder` 的低秩适配器；离线转换器 [lora_file.py](<../moss_lora/lora_file.py>) 覆盖上述两类键名，其余自定义模块需自行扩展词表。

后续迭代可沿三个方向扩展：

- 打包模式独立时间步调制：改造 `_Block.attn` 与 `_Block.ffn` 的调制广播逻辑，按 `cu_seqlens` 区间用 `torch.repeat_interleave` 将每个样本各自的 `[1, 6, D]` 调制向量展开到 token 维，消除 `packed` 模式共用时间步的限制。
- 参考图与排版控制微调：接入 MOSS-Image 的 4D RoPE 首轴（reference index `r`）与带边界框坐标的 layout JSON 监督，支持图像编辑（Edit）变体与精确文字排版 LoRA 训练。
