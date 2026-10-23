# MOSS-Image LoRA Trainer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **[한국어](README.ko.md)**

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)(5B 단일 스트림 DiT, 동결된 Qwen3.5-9B 텍스트 인코더, FLUX.2 VAE)를 위한 로컬 LoRA 트레이너로, PyTorch 학습 스택, FastAPI 백엔드, 빌드가 필요 없는 네이티브 ES module 프런트엔드로 구성되어 있습니다.

이 저장소는 공식 `comfyui/moss-image/model.py` 및 `loaders.py`의 모델 구조와 가중치 매핑을 재사용하며, 업스트림에 없는 파인튜닝 구성 요소(데이터셋 버킷팅, latent 및 텍스트 임베딩 캐싱, Flow Matching 학습, 샘플 프리뷰, 캡션 생성, WebUI)를 구현합니다.

5B DiT와 9B 텍스트 인코더가 동시에 VRAM에 상주하지 않도록 워크플로를 오프라인 캐싱과 온라인 학습의 두 단계로 분리합니다. 학습 전에 latent와 텍스트 임베딩을 디스크에 기록하고 학습 시에는 5B DiT만 로드하므로 단일 소비자용 GPU에서 1024² 파인튜닝이 가능합니다. 출력되는 `.safetensors` 체크포인트는 기본적으로 `lora_unet_*` 키 형식을 사용하며, 학습이나 내보내기에 ComfyUI 런타임 없이도 ComfyUI에서 바로 로드할 수 있습니다.

## 주요 기능

- **2단계 캐싱**: FLUX.2 VAE로 이미지 latent를, 로컬 Qwen3.5-9B로 텍스트 임베딩을 인코딩하여 디스크에 저장합니다. 텍스트 배치는 우측 인과적(causal) 패딩을 사용하며, 학습 순전파 시 어텐션 마스크에 따라 각 시퀀스의 유효 길이를 슬라이싱하므로 패딩 토큰이 유효한 텍스트 특징을 오염시키지 않습니다.
- **Flow Matching 학습 및 VRAM 프리셋**: 절단된 logit-normal $\sigma$ 샘플링과 토큰 단위로 정규화된 속도장 MSE 손실로 학습합니다. 내장된 `auto|32|24|16|12|8|6` VRAM 프리셋은 그래디언트 체크포인팅, NF4 양자화(`none|ffn|all`), CPU 블록 스왑, `torch.compile`, `fp16` 누적을 조합하며 `auto`는 `packed`(FlashAttention-2 varlen)와 `padded`(토큰 차원 패딩) 중 하나를 자동으로 고릅니다.
- **정규화 세트 및 홀드아웃 검증**: `dataset.reg_path`를 통해 정규화 이미지를 로드하여 개념 드리프트를 억제하고, `train.eval_holdout`으로 결정론적 검증 세트를 분리하여 주기적으로 검증 손실을 계산합니다.
- **이중 스타일 캡션 생성**: OpenAI 호환 비전 모델 엔드포인트를 호출하여 기존 태그를 유지하고 장면 문장을 추가하는 `tag_caption`과 3단계(long / medium / short) 설명을 생성하는 `official` 스타일을 지원하며, 동일한 이름의 sidecar JSON에 저장합니다.
- **무빌드 WebUI**: 9개 탭(대시보드, 학습 설정, 데이터셋, 캐시, 샘플 이미지, 내보내기, 작업 공간, 설정, 캡션 생성)에서 네이티브 ES module을 직접 실행합니다. 소스 수정 사항은 브라우저 새로고침만으로 즉시 반영됩니다.
- **네이티브 ComfyUI 형식**: 체크포인트와 내보내기 파일은 기본적으로 `lora_unet_*` 키 형식으로 저장되며, `diffusers` 형식 변환과 스텝 범위 일괄 ZIP 패키징을 지원합니다.
- **다국어 UI**: 간체 중국어, 영어, 일본어, 한국어 사전을 내장하여 페이지 새로고침 없이 전환할 수 있으며, 템플릿 내보내기와 사용자 언어 팩 덮어쓰기를 지원합니다.

## 시스템 요구 사항

- **OS 및 Python**: Linux 또는 WSL2, Python 3.11 이상.
- **GPU 및 드라이버**: `bfloat16` 또는 `float16`을 지원하는 NVIDIA GPU 및 PyTorch 2.9 이상에 맞는 CUDA 드라이버.
- **VRAM 등급**: 16 GB 이상은 비양자화 정밀도(`preset: 16|24|32`), 12 GB는 `preset: 12`(FFN NF4 양자화), 8 GB와 6 GB는 각각 `preset: 8`(전체 선형 계층 NF4 양자화)과 `preset: 6`(블록 스왑 및 8-bit 옵티마이저 추가)을 사용합니다.
- **로컬 가중치**: MOSS-Image DiT, Qwen3.5-9B 텍스트 인코더(HuggingFace 디렉터리 또는 ConvRot int8 양자화 단일 파일 `.safetensors`), FLUX.2 VAE.

## 설치

이 저장소는 프로젝트 루트의 `.venv`에 의존성을 격리합니다. CLI와 WebUI 백그라운드 워커 모두 외부 conda나 ComfyUI 환경에 의존하지 않고 `.venv/bin/python`을 직접 호출합니다.

[setup.sh](<setup.sh>)는 `python3.14`부터 `python3.11`까지 인터프리터를 탐색하여 `.venv`를 생성하고 [requirements.txt](<requirements.txt>)를 설치한 뒤, [setup_fa2.py](<setup_fa2.py>)를 실행하여 현재 Python, PyTorch, CUDA, GPU 아키텍처에 맞는 사전 빌드된 FlashAttention-2 휠을 설치합니다(일치하는 휠이 없으면 경고를 출력하고 SDPA로 대체합니다). [run.sh](<run.sh>)는 대상 포트 사용 여부를 확인하고 `.venv`가 없으면 `./setup.sh`를 먼저 실행한 후 WebUI를 시작합니다(기본값 `127.0.0.1:7861`).

```bash
# .venv 생성 및 의존성 설치 (PYTHON 또는 TORCH_INDEX_URL로 지정 가능)
./setup.sh
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 ./setup.sh --force

# 특정 포트에서 WebUI 시작 또는 중지
./run.sh --port 7869
./run.sh --port 7869 --stop
```

## 빠른 시작

WebUI 모드에서는 `./run.sh --port 7869`를 실행하고 `http://127.0.0.1:7869`에 접속합니다. **작업 공간** 탭에서 새 작업 공간을 만들고 **학습 설정**에서 실행을 새로 만든 뒤 `<MODELS_DIR>`와 `<DATASET_DIR>`를 입력하고, latent 캐싱, 텍스트 임베딩 캐싱, 학습을 순서대로 실행합니다.

CLI 모드에서는 한 실행의 설정을 기준으로 다음 세 단계를 실행합니다:

```bash
WS=workspaces/<WORKSPACE>
RUN=<run>
CONFIG="$WS/runs/$RUN/config.yaml"

# 1. 이미지 latent를 디스크에 인코딩 ($WS/train_cache에 기록)
.venv/bin/python -m moss_lora.cli cache --config "$CONFIG"

# 2. 텍스트 임베딩을 디스크에 인코딩 (로컬 Qwen3.5-9B)
.venv/bin/python -m moss_lora.cli encode-text --config "$CONFIG"

# 3. LoRA 학습 실행 ($WS/runs/<name>에 출력)
.venv/bin/python -m moss_lora.cli train --config "$CONFIG"
```

데이터셋 스캐너는 `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` 이미지를 재귀적으로 탐색하여 각 이미지를 sidecar JSON, 동일한 이름의 `.txt`/`.caption` 파일 또는 `metadata.jsonl` 항목과 매칭합니다. `3_concept`처럼 접두사가 붙은 하위 디렉터리는 내부 이미지를 에폭당 3회 반복합니다.

자주 사용하는 보조 하위 명령:

```bash
# 데이터셋 통계 및 버킷 분포 확인
.venv/bin/python -m moss_lora.cli info --config "$CONFIG"

# VRAM 예산(GB)에 맞춰 정밀도, 양자화, 블록 스왑을 계획하고 설정에 반영
.venv/bin/python -m moss_lora.cli plan --vram 12 --config "$CONFIG" --write "$CONFIG"

# LoRA 없는 베이스라인 이미지 생성 또는 특정 스텝 체크포인트의 프리뷰 렌더링
.venv/bin/python -m moss_lora.cli baseline --config "$CONFIG"
.venv/bin/python -m moss_lora.cli preview --config "$CONFIG" --step 500

# 사용자 프롬프트 인코딩 및 특정 체크포인트로 샘플링
.venv/bin/python -m moss_lora.cli encode-prompt --config "$CONFIG" \
    --out "$WS/custom_prompts.safetensors" --prompt "a portrait in soft daylight"
.venv/bin/python -m moss_lora.cli sample --config "$CONFIG" \
    --checkpoint "$WS/runs/<name>/checkpoints/<ckpt>.safetensors" \
    --embeds "$WS/custom_prompts.safetensors" --out "$WS/samples"

# 작업 공간 내 미완료 학습 작업을 순차적으로 재개
.venv/bin/python -m moss_lora.cli queue --workspace "$WS"

# SSH 연결이 끊겨도 계속되는 학습을 시작하고, 대시보드와 같은 방식으로 상태를 읽기
.venv/bin/python -m moss_lora.cli train --config "$CONFIG" --detach
.venv/bin/python -m moss_lora.cli status
.venv/bin/python -m moss_lora.cli status --workspace my-lora --live --json
```

`status` 는 워크스페이스 파일을 직접 읽습니다. 서버도 캐시도 쓰지 않습니다. 대시보드, `GET /api/jobs`, 이 명령은 같은 하트비트를 투영하므로 터미널에서 시작한 실행은 브라우저에 나타나고, 브라우저에서 시작한 실행은 터미널에 나타납니다. 파일 정의와 세 가지 읽기를 일치시키는 테스트는 [docs/RUNTIME-CONTRACT.md](<docs/RUNTIME-CONTRACT.md>) 에 있습니다.

## 구성

구성은 머신 수준 `settings.yaml`과 실행별 설정의 두 계층으로 나뉩니다.

`settings.yaml`은 로컬 머신 설정을 저장하며(`.gitignore`에서 제외되며 첫 실행 시 [settings.example.yaml](<settings.example.yaml>)에서 복사됨), 작업 공간 루트 `paths.workspaces`, CUDA 할당자 설정 `runtime`(`allocator: expandable|async|off` 및 `gc_threshold`), UI 언어 `ui.language`, 캡션 생성 서비스 `caption`(`base_url`, 환경 변수 이름 `api_key`, 모델 이름, 동시성 수, 프롬프트 템플릿)을 관리합니다. 설정 탭에서 직접 입력한 평문 API 키는 프로세스 메모리에만 상주하며 디스크에는 기록되지 않습니다.

각 실행의 전체 설정은 `workspaces/<WORKSPACE>/runs/<RUN>/config.yaml`에 있으며 모델, 데이터셋, 캐시, LoRA rank 및 alpha, 학습 하이퍼파라미터, VRAM 프리셋, 프리뷰 프롬프트를 정의합니다. 상대 경로(`cache.dir`, `output_dir` 등)는 작업 공간 디렉터리를 기준으로 해석되므로 작업 공간 디렉터리를 이동해도 경로가 유효하게 유지됩니다. 이 파일은 실행 중에는 읽기 전용이며 끝난 뒤에는 수정할 수 있습니다. WebUI에서 실행을 만들면 한 부가 생기고, 실행 대화상자에서 기존 실행을 복사할 수도 있습니다:

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
  preset: auto
  compile: true
export:
  formats: [kohya]
```

## 캡션 생성

캡션 생성 모듈([captioner.py](<moss_lora/captioner.py>))은 WebUI의 **캡션 생성** 탭이나 CLI에서 OpenAI 호환 비전 모델을 호출합니다:

```bash
.venv/bin/python -m moss_lora.captioner \
    --images <DATASET_DIR> \
    --styles tag_caption,official \
    --base-url http://127.0.0.1:8080/v1 \
    --api-key VLM_API_KEY
```

엔드포인트와 인증 정보는 `VLM_BASE_URL`, `VLM_API_KEY`, `VLM_MODEL` 환경 변수로도 전달할 수 있습니다. `--api-key`는 실제 키 문자열이나 키가 저장된 환경 변수 이름을 모두 허용합니다(`cli caption`은 `settings.yaml`의 해당 필드를 읽습니다).

`--styles`에는 하나 또는 두 스타일을 모두 지정할 수 있습니다:

- `tag_caption`(`tag_caption_style`): `<stem>.txt`/`.caption`을 읽고(없으면 기존 sidecar의 `kept_tags`로 대체), 기존 태그를 유지한 채 비전 모델에 자연어 장면 묘사(포즈, 구도, 카메라 앵글, 배경, 조명)를 요청하며 머리색이나 눈동자 색 같은 정체성 특징은 제거합니다. 사용할 수 있는 태그가 없는 이미지는 건너뜁니다.
- `official`(`official_style`): 기존 태그 파일을 무시하고 공식 recaptioner 사양에 따라 `caption_long`, `caption_medium`, `caption_short` 3단계 설명과 워터마크 좌표 및 테두리 방향을 출력합니다.

두 스타일은 `<stem>_moss-image-trainer.json`(`schema: 2`)을 공유하며 원본 `.txt` 파일은 수정하지 않습니다. 단일 스타일을 다시 실행하면 해당 블록만 갱신됩니다. 기본적으로 기존 결과는 건너뛰며, `--overwrite`는 강제 재생성, `--migrate`는 구형 평면 sidecar 업그레이드, `--validate-only`는 기존 파일 검증만 수행합니다:

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

`dataset.moss_caption_source`는 학습 시 읽을 필드(`tag_caption`, `official_long`, `official_medium` 또는 `official_short`)를 선택합니다. `dataset.metadata`를 설정하면 `metadata.jsonl`이 우선 적용되며, 선택한 스타일이 없는 샘플은 `<stem>.txt`/`.caption` 및 `reg_caption`으로 대체됩니다. 캡션 소스를 변경한 후에는 `encode-text`만 다시 실행하면 되며 latent 캐시는 그대로 유지됩니다.

배치 크기는 캐시 설정입니다. `cache.text_batch`와 `cache.vae_batch`로 고정할 수 있고(`0`은 남은 VRAM에서 자동), 실제로 사용한 값은 `train_cache/index.json`의 `batch`에 기록됩니다. VAE latent는 배치와 무관하며(`LatentCodec`는 fp32 합성곱으로 계산), 텍스트 임베딩은 배치에 따라 달라지므로 재현이 필요하면 텍스트 배치를 고정하세요.

## UI 언어

WebUI는 간체 중국어(`zh`), 영어(`en`), 일본어(`ja`), 한국어(`ko`) 사전(`moss_lora/web/static/i18n/<code>.json`)을 내장하고 있으며, 상단 바에서 페이지 새로고침 없이 언어를 전환할 수 있습니다.

**설정 → UI 언어**에서 템플릿을 내보내고 사용자 언어 팩을 가져올 수 있습니다. `$code`(`ru` 또는 `pt-br` 등), `$name`, 번역 문구(`{n}`, `{path}` 등의 플레이스홀더 유지)를 작성하여 업로드하면 `i18n/<code>.json`(`.gitignore`에서 제외됨)에 저장됩니다. 동일한 `$code`는 삭제될 때까지 내장 사전을 덮어쓰며, 누락된 키는 활성 언어 → 간체 중국어 → 원본 키 순서로 대체됩니다.

## ComfyUI로 내보내기

학습 체크포인트(`runs/<name>/checkpoints/stepXXXXXX.safetensors`, `final.safetensors`)와 최종 내보내기 파일 `runs/<name>/<name>.safetensors`는 ComfyUI 호환 kohya 키 형식을 사용합니다. 옵티마이저 및 RNG 상태는 `state.safetensors`(또는 `<ckpt>.state.safetensors`)에 별도로 저장됩니다.

텐서 키는 `lora_unet_`로 시작하고 레이어 경로의 마침표를 밑줄로 치환하며, ComfyUI의 `Load LoRA` 노드가 이를 `diffusion_model.layers.<layer>.proj.<module>.weight`에 매핑합니다:

```text
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_down.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.lora_up.weight
lora_unet_layers_<0..41>_proj_<q|k|v|o|ffn_up_gate|ffn_down>.alpha
```

`.safetensors` 파일을 `<COMFYUI_DIR>/models/loras/`에 넣고 `Load LoRA` 노드를 연결하여 사용합니다. `diffusers` 형식(`transformer.layers.*`) 변환이나 스텝 범위 일괄 내보내기는 WebUI의 **내보내기** 탭이나 CLI에서 수행할 수 있습니다(5B 모델을 로드하지 않는 순수 키 재매핑):

```bash
# 단일 스텝 체크포인트를 kohya와 diffusers 형식으로 동시에 내보내기
.venv/bin/python -m moss_lora.cli export \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --checkpoint 1000 \
    --formats kohya,diffusers \
    --out "workspaces/<WORKSPACE>/exports"

# 스텝 범위를 일괄 내보내고 zip 아카이브로 패키징
.venv/bin/python -m moss_lora.cli export-batch \
    --config "workspaces/<WORKSPACE>/runs/<RUN>/config.yaml" \
    --steps 250-1000:250 \
    --formats kohya \
    --out "workspaces/<WORKSPACE>/exports" \
    --zip "workspaces/<WORKSPACE>/exports/batch.zip"
```

## 테스트

단위 테스트는 GPU 없이 실행되며 sidecar 읽기/쓰기 및 마이그레이션, 4개 국어 사전 정합성, API 키 저장 규칙, 문서 민감 정보 제거 검사를 다룹니다. 선택적 테스트를 통해 로컬 텍스트 인코더에서 우측 패딩 배치 인코딩과 단일 샘플 인코딩의 일치 여부를 검증할 수 있습니다:

```bash
.venv/bin/python tests/test_sidecar.py
.venv/bin/python tests/test_i18n.py
.venv/bin/python tests/test_caption_key.py
.venv/bin/python tests/test_docs_sanitized.py

# 선택 사항: 우측 패딩 배치 인코딩과 단일 항목 인코딩의 일치 검증
MOSS_TE=<MODELS_DIR>/text_encoders/Qwen3.5-9B .venv/bin/python tests/test_text_padding.py
```

프런트엔드 회귀 테스트 하네스 [smoke.mjs](<tests/ui/smoke.mjs>)는 네이티브 Node.js ESM과 Chromium DevTools Protocol(외부 npm 패키지 없음)을 사용하여 1440×1000 해상도에서 4개 언어와 9개 탭을 모두 순회하며 콘솔 오류, 실패한 요청, 가로 오버플로, 잘린 텍스트, 번역되지 않은 원본 키를 검사합니다:

```bash
.venv/bin/python -m moss_lora.cli web --port 7891 &
chromium --headless=new --remote-debugging-port=9401 \
    --user-data-dir=.tmp/chrome --no-sandbox --disable-gpu about:blank &
node tests/ui/smoke.mjs --base http://127.0.0.1:7891 --cdp http://127.0.0.1:9401 \
    --langs zh,en,ja,ko --out .tmp/ui.json
```

## 디렉터리 구조

```text
moss-image-trainer/
├── setup.sh / setup_fa2.py   # .venv 생성, 의존성 설치 및 FlashAttention-2 휠 매칭
├── run.sh                    # 환경 점검 및 WebUI 실행
├── settings.example.yaml     # 머신 수준 구성 템플릿
├── moss_lora/
│   ├── cli.py                # CLI 진입점 (cache / encode-text / train / export / web 등)
│   ├── config.py             # 구성 dataclass, 경로 해석, 검증 및 VRAM 프리셋
│   ├── model.py              # 공식 MOSS-Image DiT 아키텍처 적용 (SDPA / FA2 및 그래디언트 체크포인팅)
│   ├── weights.py / convrot.py        # DiT/VAE 키 매핑 및 ConvRot int8 역양자화
│   ├── data.py / cache.py    # 데이터셋 스캔, 버킷팅 및 이미지 latent 캐싱
│   ├── hf_text.py / encode.py # 로컬 Qwen3.5-9B 로더 및 텍스트 임베딩 캐싱
│   ├── captioner.py          # 캡션 생성 워커 및 sidecar JSON 관리
│   ├── trainer.py            # Flow Matching 학습 루프, 샘플 프리뷰 및 체크포인트 관리
│   ├── quant.py / offload.py # 동결 계층 NF4 양자화 및 CPU 블록 스왑
│   ├── lora_file.py / export_batch.py # 모델 로드 없는 LoRA 형식 변환 및 일괄 내보내기
│   ├── i18n.py / settings.py # 언어 팩 병합 및 머신 수준 설정 입출력
│   └── web/                  # FastAPI 라우트/서비스 계층 및 네이티브 ES module 프런트엔드
└── tests/                    # Python 단위 테스트 및 CDP 프런트엔드 회귀 하네스
```

모듈 경계, 데이터 흐름, 프런트엔드-백엔드 계약에 대한 자세한 내용은 [docs/DESIGN.md](<docs/DESIGN.md>)를 참조하십시오.

## 알려진 제한 사항

- **배치 크기 및 처리량**: 마이크로 배치는 토큰 예산으로 구성되고 이미지는 각자의 해상도를 유지합니다. 한 스텝의 작업량을 정하는 것은 `batch_size`가 아니라 `train.bin_tokens`입니다. FlashAttention-2를 설치하면 기본값이 `padded`(토큰 패딩 + key mask)에서 `packed`(varlen, 마스크 없음)로 바뀝니다.
- **저용량 VRAM 정밀도 트레이드오프**: `model.dtype: float16`과 `vram.fp16_accumulate: true` 조합은 높은 학습률이나 장시간 학습에서 `bfloat16`보다 수치적 불안정성이 발생하기 쉽습니다. NF4 양자화(`quantize_dit: ffn|all`)와 CPU 블록 스왑은 6–12 GB GPU를 대상으로 하며, 비양자화 백본에 비해 미세한 디테일 재현력이 약간 절충됩니다.
- **학습 범위**: LoRA 어댑터는 5B DiT(기본적으로 42개 레이어의 `q`, `k`, `v`, `o`, `ffn_up_gate`, `ffn_down` 및 선택적 `text_embedder`)에만 주입되며, Qwen3.5-9B 텍스트 인코더와 VAE는 동결됩니다.

## 라이선스 및 감사의 글

- 모델 아키텍처([model.py](<moss_lora/model.py>)), 스케줄러([scheduler.py](<moss_lora/scheduler.py>)) 및 가중치 매핑은 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)(Apache-2.0 License)를 기반으로 작성되었습니다. 기본 가중치와 재패키징 가중치는 [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image)를 참조하십시오.
- 내장된 Qwen3.5 토크나이저 및 모델 구성(`moss_lora/assets/`)은 Apache-2.0 라이선스를 따르며, 자세한 내용은 [QWEN_LICENSE.txt](<moss_lora/assets/QWEN_LICENSE.txt>)를 참조하십시오.
