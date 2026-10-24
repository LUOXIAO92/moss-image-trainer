# MOSS-Image LoRA Trainer

[English](README.md) · [简体中文](README.zh-CN.md) · **[日本語](README.ja.md)** · [한국어](README.ko.md)

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（5B シングルストリーム DiT、凍結 Qwen3.5-9B テキストエンコーダー、FLUX.2 VAE）向けのローカル LoRA トレーナーです。PyTorch 学習スタック、FastAPI バックエンド、ビルド不要のネイティブ ES module フロントエンドで構成されています。

本リポジトリは公式の `comfyui/moss-image/model.py` および `loaders.py` のモデル構造と重みマッピングを再利用し、上流に含まれていないファインチューニング用コンポーネント（データセットのバケット分割、latent とテキスト埋め込みのキャッシュ、Flow Matching 学習、サンプルプレビュー、キャプション生成、WebUI）を実装しています。

5B DiT と 9B テキストエンコーダーが同時に VRAM に常駐しないよう、ワークフローをオフラインキャッシュとオンライン学習の2段階に分離しています。学習前に latent とテキスト埋め込みをディスクへ書き出し、学習時は 5B DiT のみを読み込むため、単一のコンシューマー向け GPU で 1024² のファインチューニングを実行できます。出力される `.safetensors` チェックポイントはデフォルトで `lora_unet_*` キー形式を採用しており、学習やエクスポートに ComfyUI ランタイムを必要とせず、そのまま ComfyUI に読み込めます。

## 主な機能

- **2段階キャッシュ**: FLUX.2 VAE で画像 latent を、ローカル Qwen3.5-9B でテキスト埋め込みをエンコードしてディスクへ保存します。テキストのバッチ処理には右側因果パディングを使用し、学習の順伝播ではアテンションマスクに従って各系列の有効長を切り出すため、パディングトークンが有効なテキスト特徴を汚染しません。
- **Flow Matching 学習と VRAM プリセット**: 切断 logit-normal $\sigma$ サンプリングとトークン単位で正規化した速度場 MSE 損失で学習します。6 段階の VRAM プリセット（`32`、`24`、`16`、`12`、`8`、`6` GB）は、勾配チェックポイント、重みのみの量子化（`quantize_dit: none|ffn|all` を `nf4` または `fp8` で、conditioning 側も任意で）、ホストメモリからのブロックストリーミング、`torch.compile`、`fp16` 累積を組み合わせ、`packed`（FlashAttention-2 varlen）と `padded`（トークン次元のパディング）の両バッチモードに対応します。`packed` と `padded` の両方に対応します。ダッシュボードは実行前と実行中に、実測値の隣に予測 VRAM ピークを表示します。
- **正則化セットとホールドアウト検証**: `dataset.reg_path` から正則化画像を読み込んで概念ドリフトを抑制し、`train.eval_holdout` で決定論的な検証セットを分割して定期的に検証損失を計算します。
- **2種類のスタイルによるキャプション生成**: OpenAI 互換の視覚モデル API を呼び出し、既存の tag を保持してシーン文を追記する `tag_caption` と、3段階（long / medium / short）の記述を生成する `official` スタイルをサポートし、同名の sidecar JSON に保存します。
- **ビルド不要の WebUI**: 9つのタブ（ダッシュボード、学習設定、データセット、キャッシュ、サンプル画像、エクスポート、ワークスペース、設定、キャプション生成）でネイティブ ES module を直接実行します。ソースコードの変更はブラウザの再読み込みだけで反映されます。
- **ネイティブ ComfyUI 形式**: チェックポイントとエクスポートファイルはデフォルトで `lora_unet_*` キー形式で保存され、`diffusers` 形式への変換やステップ範囲指定による一括 ZIP パッケージ化にも対応しています。
- **多言語対応 UI**: 簡体字中国語、英語、日本語、韓国語の辞書を内蔵し、ページを再読み込みせずに切り替え可能です。テンプレートのエクスポートやカスタム言語パックによる上書きもサポートしています。

## 動作要件

- **OS と Python**: Linux または WSL2、Python 3.11 以上。
- **GPU とドライバー**: `bfloat16` または `float16` に対応した NVIDIA GPU、および PyTorch 2.9 以上に対応する CUDA ドライバー。
- **VRAM 設定。プリセットは出発点にすぎない**: VRAM の各項目は実行設定に書かれた普通の値で、学習側はその値をそのまま使います。WebUI のプリセット選択（`32`、`24`、`16`、`12`、`8`、`6`）と CLI の `--preset` は、そのティアの値——勾配チェックポイント、常駐レイヤー数、量子化の範囲と形式、`torch.compile`、8bit オプティマイザー、プレビュー VAE の精度——を一度だけ書き込み、その後は関与しません。新しい run はデフォルト値で始まり、どのティアも適用されず、カードから何かを導くこともありません。1 ステップで本当にメモリが足りないときだけ、学習側が `blocks_to_swap` を自分で引き上げます。
- **ローカル重み**: MOSS-Image DiT、Qwen3.5-9B テキストエンコーダー（HuggingFace ディレクトリ、または ConvRot int8 量子化済み単一ファイル `.safetensors`）、および FLUX.2 VAE。

特定のカード向けに設定を選ぶときは [VRAM ガイド](<docs/VRAM.md>) を読んでください。量子化とブロックストリーミングの実測コストを示し、アロケーターの設定を説明し、カードのクラスごとにプレビューサイズを一覧にしています。

## インストール

本リポジトリはプロジェクト直下の `.venv` に依存関係を隔離します。CLI と WebUI のバックグラウンドワーカーはいずれも `.venv/bin/python` を直接呼び出し、外部の conda 環境や ComfyUI 環境には依存しません。

[setup.sh](<setup.sh>) は `python3.14` から `python3.11` の順にインタープリターを探して `.venv` を作成し、[requirements.txt](<requirements.txt>) をインストールした後、[setup_fa2.py](<setup_fa2.py>) を実行して現在の Python、PyTorch、CUDA、GPU アーキテクチャに一致するビルド済み FlashAttention-2 wheel をインストールします（一致する wheel がない場合は警告を出して SDPA にフォールバックします）。[run.sh](<run.sh>) は対象ポートの空きを確認し、`.venv` が未作成の場合は `./setup.sh` を実行してから WebUI を起動します（デフォルトは `127.0.0.1:7861`）。

```bash
# .venv を作成して依存関係をインストール（PYTHON または TORCH_INDEX_URL で上書き可能）
./setup.sh
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./setup.sh --force

# 指定ポートで WebUI を起動または停止
./run.sh --port 7869
./run.sh --port 7869 --stop
```

## クイックスタート

WebUI では、`./run.sh --port 7869` を実行して `http://127.0.0.1:7869` を開きます。「**ワークスペース**」タブでワークスペースを新規作成し、「**学習設定**」で実行を作成して `<MODELS_DIR>` と `<DATASET_DIR>` を設定してから、latent キャッシュ、テキスト埋め込みキャッシュ、学習の順に実行します。

CLI では、1 つの実行の設定に対して以下の3ステップを順に実行します。

```bash
WS=workspaces/<WORKSPACE>
RUN=<run>
CONFIG="$WS/runs/$RUN/config.yaml"

# 1. 画像 latent をディスクへエンコード（$WS/train_cache に書き出し）
.venv/bin/python -m moss_lora.cli cache --config "$CONFIG"

# 2. テキスト埋め込みをディスクへエンコード（ローカル Qwen3.5-9B）
.venv/bin/python -m moss_lora.cli encode-text --config "$CONFIG"

# 3. LoRA 学習を実行（$WS/runs/<name> に出力）
.venv/bin/python -m moss_lora.cli train --config "$CONFIG"
```

データセットスキャナーは `.jpg`、`.jpeg`、`.png`、`.webp`、`.bmp` 画像を再帰的に検索し、各画像を sidecar JSON、同名の `.txt`/`.caption` ファイル、または `metadata.jsonl` のエントリとペアリングします。`3_concept` のようなプレフィックス付きサブディレクトリ内の画像は、エポックごとに 3 回反復されます。

よく使う補助サブコマンド：

```bash
# データセットの統計情報とバケット分布を確認
.venv/bin/python -m moss_lora.cli info --config "$CONFIG"

# VRAM 予算（GB）に合わせて精度・量子化・ブロックスワップを計画し設定へ書き戻す
.venv/bin/python -m moss_lora.cli plan --vram 12 --config "$CONFIG" --write "$CONFIG"

# LoRA なしのベースライン画像を生成、または特定ステップのチェックポイントでプレビューを再描画
.venv/bin/python -m moss_lora.cli baseline --config "$CONFIG"
.venv/bin/python -m moss_lora.cli preview --config "$CONFIG" --step 500

# カスタムプロンプトをエンコードし、指定チェックポイントでサンプリング
.venv/bin/python -m moss_lora.cli encode-prompt --config "$CONFIG" \
    --out "$WS/custom_prompts.safetensors" --prompt "a portrait in soft daylight"
.venv/bin/python -m moss_lora.cli sample --config "$CONFIG" \
    --checkpoint "$WS/runs/<name>/checkpoints/<ckpt>.safetensors" \
    --embeds "$WS/custom_prompts.safetensors" --out "$WS/samples"

# ワークスペース内の未完了の学習タスクを順次再開
.venv/bin/python -m moss_lora.cli queue --workspace "$WS"

# SSH 切断後も続く学習を開始し、ダッシュボードと同じ方法で状態を読む
.venv/bin/python -m moss_lora.cli train --config "$CONFIG" --detach
.venv/bin/python -m moss_lora.cli status
.venv/bin/python -m moss_lora.cli status --workspace my-lora --live --json
```

`status` はワークスペースのファイルを直接読みます。サーバーもキャッシュも使いません。ダッシュボード、`GET /api/jobs`、このコマンドは同じハートビートを投影するため、ターミナルで開始した実行はブラウザに現れ、ブラウザで開始した実行はターミナルに現れます。ファイルの定義と三者を一致させ続けるテストは [docs/RUNTIME-CONTRACT.md](<docs/RUNTIME-CONTRACT.md>) にあります。

## 設定

設定はマシン単位の `settings.yaml` と、実行ごとの設定の2階層に分かれています。

`settings.yaml` はローカルマシンの設定を保持します（`.gitignore` で除外されており、初回起動時に [settings.example.yaml](<settings.example.yaml>) からコピーされます）。ワークスペースのルート `paths.workspaces`、CUDA アロケーター設定 `runtime`（`allocator: expandable|async|off` と `gc_threshold`）、表示言語 `ui.language`、およびキャプション生成サービス `caption`（`base_url`、環境変数名 `api_key`、モデル名、並列数、プロンプトテンプレート）を管理します。設定画面で直接入力した平文の API キーはプロセスメモリ内にのみ保持され、ディスクには書き込まれません。

各実行の完全な設定は `workspaces/<WORKSPACE>/runs/<RUN>/config.yaml` にあり、モデル、データセット、キャッシュ、LoRA の rank と alpha、学習ハイパーパラメーター、VRAM 設定、プレビュー用プロンプトを定義します。相対パス（`cache.dir` や `output_dir` など）はワークスペースディレクトリ基準で解決されるため、ワークスペースディレクトリごと移動しても参照は維持されます。このファイルは実行中は読み取り専用で、終了後は編集できます。WebUI で実行を作成すると 1 つの設定ができ、実行ダイアログから既存の実行を複製することもできます。

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
  batch_size: 1
  grad_accum: 1
  lr: 0.0001
  optimizer: adamw
  save_every: 250
  sample_every: 250
vram:
  gradient_checkpointing: true
  blocks_to_swap: 0
  quantize_dit: none
  compile: true
export:
  formats: [kohya]
```

## キャプション生成

キャプション生成モジュール（[captioner.py](<moss_lora/captioner.py>)）は、WebUI の「**キャプション生成**」タブまたは CLI から OpenAI 互換の視覚モデルを呼び出します。

```bash
.venv/bin/python -m moss_lora.captioner \
    --images <DATASET_DIR> \
    --styles tag_caption,official \
    --base-url http://127.0.0.1:8080/v1 \
    --api-key VLM_API_KEY
```

エンドポイントと認証情報は環境変数 `VLM_BASE_URL`、`VLM_API_KEY`、`VLM_MODEL` でも指定できます。`--api-key` にはキー文字列そのもの、またはキーを格納した環境変数名のどちらも指定可能です（`cli caption` は `settings.yaml` の対応フィールドを読み取ります）。

`--styles` には片方または両方のスタイルを指定できます。

- `tag_caption`（`tag_caption_style`）：`<stem>.txt`/`.caption` を読み取り（存在しない場合は既存 sidecar の `kept_tags` にフォールバック）、元の tag を保持したまま、視覚モデルに自然言語のシーン記述（ポーズ、構図、カメラアングル、背景、照明）を生成させ、髪色や瞳の色などのアイデンティティ特徴を除外します。利用可能な tag がない画像はスキップします。
- `official`（`official_style`）：既存の tag ファイルを無視し、公式 recaptioner 仕様に従って `caption_long`、`caption_medium`、`caption_short` の3段階記述および透かし矩形と枠線の方向を出力します。

両スタイルは `<stem>_moss-image-trainer.json`（`schema: 2`）を共有し、元の `.txt` ファイルは変更しません。単一スタイルを再実行した場合は該当ブロックのみを更新します。デフォルトでは既存の結果をスキップし、`--overwrite` で強制再生成、`--migrate` で旧形式のフラットな sidecar をアップグレード、`--validate-only` で既存ファイルの検証のみを行います。

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

`dataset.moss_caption_source` で学習時に読み込むフィールド（`tag_caption`、`official_long`、`official_medium`、または `official_short`）を選択します。`dataset.metadata` を設定した場合は `metadata.jsonl` が優先され、選択したスタイルが存在しないサンプルは `<stem>.txt`/`.caption` および `reg_caption` にフォールバックします。キャプションソースを切り替えた後は `encode-text` のみを再実行すればよく、latent キャッシュの再構築は不要です。

バッチサイズはキャッシュの設定です。`cache.text_batch` と `cache.vae_batch` で固定でき（`0` は空き VRAM から自動）、実際に使った値は `train_cache/index.json` の `batch` に記録されます。VAE の latent はバッチに依存しません（`LatentCodec` は fp32 畳み込みで計算）。テキスト埋め込みは依存するため、再現したい場合はテキスト側のバッチを固定してください。

## 表示言語

WebUI は簡体字中国語（`zh`）、英語（`en`）、日本語（`ja`）、韓国語（`ko`）の辞書（`moss_lora/web/static/i18n/<code>.json`）を内蔵しており、トップバーからページを再読み込みせずに言語を切り替えられます。

「**設定 → 表示言語**」では、テンプレートをエクスポートしてカスタム言語パックをインポートできます。`$code`（`ru` や `pt-br` など）、`$name`、翻訳文字列（`{n}` や `{path}` などのプレースホルダーは保持）を記入してアップロードすると、`i18n/<code>.json`（`.gitignore` で除外）に保存されます。同じ `$code` のファイルは削除されるまで内蔵辞書を上書きし、未翻訳のキーは「現在の言語 → 簡体字中国語 → 元のキー名」の順にフォールバックします。

## ComfyUI へのエクスポート

学習チェックポイント（`runs/<name>/checkpoints/stepXXXXXX.safetensors`、`final.safetensors`）と最終エクスポート `runs/<name>/<name>.safetensors` は、ComfyUI 互換の kohya キー形式を使用します。オプティマイザーと乱数状態は `state.safetensors`（または `<ckpt>.state.safetensors`）に別ファイルとして保存されます。

テンソルキーは `lora_unet_` で始まり、レイヤーパスのドットをアンダースコアに置換した形式で、ComfyUI の `Load LoRA` ノードによって `diffusion_model.layers.<layer>.proj.<module>.weight` にマッピングされます。

```text
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_down.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_up.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.alpha
```

`.safetensors` ファイルを `<COMFYUI_DIR>/models/loras/` に配置し、`Load LoRA` ノードを接続して使用します。`diffusers` 形式（`transformer.layers.*`）への変換やステップ範囲の一括エクスポートは、WebUI の「**エクスポート**」タブまたは CLI から実行できます（5B モデルを読み込まない純粋なキー変換です）。

```bash
# 単一ステップのチェックポイントを kohya と diffusers の両形式でエクスポート
.venv/bin/python -m moss_lora.cli export \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --checkpoint 1000 \
    --formats kohya,diffusers \
    --out "workspaces/<WORKSPACE>/exports"

# ステップ範囲を一括エクスポートして ZIP アーカイブにパッケージ化
.venv/bin/python -m moss_lora.cli export-batch \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --steps 250-1000:250 \
    --formats kohya \
    --out "workspaces/<WORKSPACE>/exports" \
    --zip "workspaces/<WORKSPACE>/exports/batch.zip"
```

## テスト

単体テストは GPU なしで実行でき、sidecar の読み書きとマイグレーション、4言語辞書の整合性、API キーの保存ルール、ドキュメントのサニタイズ検査をカバーしています。また、ローカルのテキストエンコーダーで右側パディングバッチ処理と単一サンプル処理の一致を検証する任意のテストも含まれています。

```bash
.venv/bin/python tests/test_sidecar.py
.venv/bin/python tests/test_i18n.py
.venv/bin/python tests/test_caption_key.py
.venv/bin/python tests/test_docs_sanitized.py

# 任意：右側パディングによるバッチエンコードと単一エンコードの一致を検証
MOSS_TE=<MODELS_DIR>/text_encoders/Qwen3.5-9B .venv/bin/python tests/test_text_padding.py
```

フロントエンドの回帰ハーネス [smoke.mjs](<tests/ui/smoke.mjs>) は、Node.js ネイティブ ESM と Chromium DevTools Protocol（外部 npm パッケージゼロ）を使用し、1440×1000 のビューポートで全4言語・全9タブを巡回して、コンソールエラー、失敗したリクエスト、横スクロールのはみ出し、テキストの切れ、未翻訳の生キーを検査します。

```bash
.venv/bin/python -m moss_lora.cli web --port 7891 &
chromium --headless=new --remote-debugging-port=9401 \
    --user-data-dir=.tmp/chrome --no-sandbox --disable-gpu about:blank &
node tests/ui/smoke.mjs --base http://127.0.0.1:7891 --cdp http://127.0.0.1:9401 \
    --langs zh,en,ja,ko --out .tmp/ui.json
```

## ディレクトリ構成

```text
moss-image-trainer/
├── setup.sh / setup_fa2.py   # .venv の作成、依存関係のインストール、FlashAttention-2 wheel のマッチング
├── run.sh                    # 環境チェックと WebUI の起動
├── settings.example.yaml     # マシン単位の設定テンプレート
├── moss_lora/
│   ├── cli.py                # CLI エントリポイント（cache / encode-text / train / export / web など）
│   ├── config.py             # 設定 dataclass、パス解決、バリデーション、VRAM プリセット
│   ├── model.py              # 公式 MOSS-Image DiT の適応実装（SDPA / FA2 と勾配チェックポイント）
│   ├── weights.py / convrot.py        # DiT・VAE のキーマッピングと ConvRot int8 逆量子化
│   ├── data.py / cache.py    # データセットスキャン、バケット分割、画像 latent キャッシュ
│   ├── hf_text.py / encode.py # ローカル Qwen3.5-9B ローダーとテキスト埋め込みキャッシュ
│   ├── captioner.py          # キャプション生成ワーカーと sidecar JSON 管理
│   ├── trainer.py            # Flow Matching 学習ループ、サンプルプレビュー、チェックポイント管理
│   ├── quant.py / offload.py # 凍結層の NF4 量子化と CPU ブロックスワップ
│   ├── lora_file.py / export_batch.py # モデル読み込み不要の LoRA 形式変換と一括エクスポート
│   ├── i18n.py / settings.py # 言語パックのマージとマシン設定の読み書き
│   └── web/                  # FastAPI ルート/サービス層とネイティブ ES module フロントエンド
└── tests/                    # Python 単体テストと CDP フロントエンド回帰ハーネス
```

モジュール境界、データフロー、フロントエンドとバックエンドの契約については [docs/DESIGN.md](<docs/DESIGN.md>) を参照してください。

## 既知の制限事項

- **バッチサイズとスループット**: マイクロバッチはトークン予算で組まれ、画像はそれぞれの解像度を保ちます。1 ステップの仕事量を決めるのは `batch_size` ではなく `train.bin_tokens` です。FlashAttention-2 を入れると既定が `padded`（トークンパディング + key mask）から `packed`（varlen、マスクなし）に切り替わります。
- **低 VRAM 設定での精度トレードオフ**: `model.dtype: float16` と `vram.fp16_accumulate: true` の組み合わせは、高い学習率や長時間の学習において `bfloat16` よりも数値的不安定性を起こしやすくなります。量子化（`quantize_dit: ffn|all`）とブロックストリーミングは 16 GB 未満のカードを対象とし、載せきるために細かな再現性をわずかに手放します。`nf4` が最も小さい形式、`fp8` がより安定した形式で、conditioning 側まで意図的に量子化すると誤差が全ブロックに積み重なるため、明示しない限り無効のままです。
- **学習対象の範囲**: LoRA アダプターは 5B DiT（デフォルトで 42 層の `q`、`k`、`v`、`o`、`ffn_up_gate`、`ffn_down`、および任意の `text_embedder`）にのみ適用され、Qwen3.5-9B テキストエンコーダーと VAE は凍結されます。

## ライセンスと謝辞

- モデル構造（[model.py](<moss_lora/model.py>)）、スケジューラー（[scheduler.py](<moss_lora/scheduler.py>)）、および重みマッピングは [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（Apache-2.0 License）を基にしています。ベース重みと再パッケージ重みについては [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)、[OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)、および [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) を参照してください。
- 同梱されている Qwen3.5 トークナイザーとモデル設定（`moss_lora/assets/`）は Apache-2.0 License に従います。詳細は [QWEN_LICENSE.txt](<moss_lora/assets/QWEN_LICENSE.txt>) を参照してください。
