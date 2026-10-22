"""Generate training captions for MOSS-Image LoRA with a local VLM.

One request per image per style, written into a single sidecar next to the image:
``<stem>_moss-image-trainer.json``.  Two independent styles live in that file, so a
dataset can hold both a "keep my tags" caption and the official recaptioner output:

    tag_caption_style  保留原 tag，再补一句场景描述（需要 <stem>.txt 或已有的参考 tag）
    official_style     官方 recaptioner：long / medium / short 三档，无视现成标注

同一张图再生成某个风格时，只替换那个风格的字段，另一个原样保留；不记录版本历史。
原 ``<stem>.txt`` 永远不动，一个目录可以同时喂 SDXL 和 MOSS-Image。

Endpoint comes from the CLI/settings, falling back to the environment::

    VLM_BASE_URL   e.g. http://127.0.0.1:8080/v1
    VLM_API_KEY    bearer token
    VLM_MODEL      model id (default: first model the server reports)

    .venv/bin/python -m moss_lora.captioner --images <dir> --styles tag_caption,official
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SIDECAR_SUFFIX = "_moss-image-trainer.json"
TOOL_NAME = "moss-image-trainer"
TOOL_VERSION = 2
SCHEMA_VERSION = 2

STYLE_TAG = "tag_caption_style"
STYLE_OFFICIAL = "official_style"
STYLES = (STYLE_TAG, STYLE_OFFICIAL)
STYLE_ALIASES = {
    "tag_caption": STYLE_TAG,
    "tags": STYLE_TAG,
    "tag": STYLE_TAG,
    "official": STYLE_OFFICIAL,
    "natural": STYLE_OFFICIAL,
}
SOURCES = ("txt", "none")
# 场景句里不该出现的身份线索：发色/发型、瞳色、体型、肤色、发色形容词
BANNED_IN_SCENE = ("hair", "blonde", "brunette", "skin color", "breast")
BANNED_SCENE_PATTERNS = (
    re.compile(r"\b(red|blue|green|purple|pink|brown|black|white|gold|golden|yellow|amber|violet|lavender|grey|gray)\s+eyes?\b", re.I),
)
BORDER_VALUES = {"left", "right", "up", "down"}
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
WEIGHTED_TAG = re.compile(r"\(([^()]+):\s*\d+(?:\.\d+)?\)")

NATURAL_PROMPT = """You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [
    {"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}
  ],
  "borders": ["none"]
}

Rules:
- caption.long:
    - Describe the composition, key subjects, actions, setting, colors, lighting, and artistic style in detail. Include only information visible in the image.
    - Preserve visible text exactly, excluding watermarks. Do not translate, paraphrase, romanize, or normalize image text.
    - Describe text layout as much as possible: line breaks, ordering, relative position, alignment, size hierarchy, font/style, color, spacing, and how text blocks are arranged.
- caption.medium: compress long into one paragraph.
- caption.short: summarize the main subject, action, or relationship in one sentence.
- Do not describe watermarks in any caption field. Record them only in watermark.
- watermark:
    - Use normalized 0-1000 [y_min, x_min, y_max, x_max] coordinates relative to the full image.
    - Set label to the visible watermark text, or a short description if unreadable.
    - Use [] if there are no watermarks.
- borders:
    - Record evident borders or margins using one or more of "left", "right", "up", and "down".
    - Use ["none"] if there are no evident borders or margins. Do not combine "none" with other directions.
- All three caption fields must be non-empty strings.
- Treat the original caption as a reference. Correct any claims that are not supported by the image.
- Return only the JSON object, without Markdown or explanations."""

TAGS_PROMPT = """You are an image recaptioner for a character-LoRA dataset.

The image belongs to the character described by these tags: {tags}

Return exactly one JSON object whose keys are written in this order: analysis, scene, watermark, borders.

{
  "analysis": "2-4 sentences, written BEFORE anything else: the framing (full-body / upper-body / close-up) and camera angle, the pose and what the subject is doing, the background and lighting, whether any text or signature is visible and roughly where, and which visible details are identity features (hair colour, eye colour, body shape, clothing colour) that must stay out of the scene line",
  "scene": "one line, at most 25 words, describing ONLY pose, framing, camera angle, background, lighting and art style",
  "watermark": [{ "box_2d": [y_min, x_min, y_max, x_max], "label": "text" }],
  "borders": ["none"]
}

Rules:
- analysis comes first and is the reasoning that the other fields follow from; it is kept for auditing, not used as caption text.
- scene must NOT mention hair colour, eye colour, body shape, clothing colours or any identity feature: those are already carried by the tags.
- scene describes what the tags cannot: full-body / upper-body / close-up, facing viewer / side view, standing / sitting / lying, indoor / outdoor / plain background, lighting, rendering style.
- watermark: [] when there is no watermark; otherwise 0-1000 normalised coordinates and the visible text as label.
- borders: "left", "right", "up", "down", or ["none"]; never mix "none" with directions.
- Do not describe watermarks inside scene.
- Nothing after the JSON object."""

SCHEMA_NATURAL = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "object",
            "properties": {
                "long": {"type": "string"},
                "medium": {"type": "string"},
                "short": {"type": "string"},
            },
            "required": ["long", "medium", "short"],
        },
        "watermark": {"type": "array"},
        "borders": {"type": "array"},
    },
    "required": ["caption", "watermark", "borders"],
}

SCHEMA_TAGS = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "scene": {"type": "string"},
        "watermark": {"type": "array"},
        "borders": {"type": "array"},
    },
    "required": ["analysis", "scene", "watermark", "borders"],
}

# -- sidecar / reference ----------------------------------------------------
def sidecar_path(image: str, out_dir: str = "") -> str:
    """``foo.jpg`` -> ``foo_moss-image-trainer.json`` (next to the image by default)."""
    stem = os.path.splitext(os.path.basename(image))[0]
    return os.path.join(out_dir or os.path.dirname(image), stem + SIDECAR_SUFFIX)


def read_reference(image: str) -> str:
    """The dataset's own caption next to the image, if any."""
    stem = os.path.splitext(image)[0]
    for ext in (".txt", ".caption"):
        path = stem + ext
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    return handle.read().strip()
            except OSError:
                return ""
    return ""


def load_tags(tags_dir: str, stem: str) -> list[str]:
    """Split the existing caption into tags (only used by the forced tags mode)."""
    if not tags_dir:
        return []
    path = os.path.join(tags_dir, f"{stem}.txt")
    if not os.path.exists(path):
        return []
    raw = open(path, encoding="utf-8").read()
    tags = []
    for chunk in re.split(r"[,\n]", raw):
        tag = chunk.strip()
        if tag:
            tags.append(tag)
    return tags

    if not isinstance(data, dict):
        return ["top level is not an object"]
    caption = data.get("caption")
    if style == "natural":
        if not isinstance(caption, dict):
            errors.append("caption is not an object")
        else:
            for field in ("long", "medium", "short"):
                value = caption.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"caption.{field} missing or empty")
                elif len(value.split()) < (3 if field == "short" else 8):
                    errors.append(f"caption.{field} suspiciously short")
    elif caption and not isinstance(caption, dict):
        errors.append("caption must be an object or null")
    errors.extend(validate_meta(data))
    return errors


def validate_natural(data: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["top level is not an object"]
    caption = data.get("caption")
    if not isinstance(caption, dict):
        errors.append("caption is not an object")
    else:
        for field in ("long", "medium", "short"):
            value = caption.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"caption.{field} missing or empty")
            elif len(value.split()) < (3 if field == "short" else 8):
                errors.append(f"caption.{field} suspiciously short")
    errors.extend(validate_meta(data))
    return errors


def clean_tags(tags, trigger: list[str]) -> list[str]:
    blocked = {item.lower() for item in trigger}
    out, seen = [], set()
    for tag in tags or []:
        key = str(tag).strip()
        if not key:
            continue
        if key.lower() in blocked or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def compose_caption(trigger: list[str], tags: list[str], scene: str, long_text: str = "") -> str:
    """tag 风格用逗号串；自然语言用句号分段，读起来才像话。"""
    prefix = ", ".join(part for part in trigger if part)
    if long_text:
        body = long_text.strip()
        return f"{prefix}. {body}" if prefix else body
    scene = (scene or "").strip().rstrip(".")
    parts = list(trigger or []) + list(tags or []) + ([scene] if scene else [])
    return ", ".join(part for part in parts if part)


# -- validation -------------------------------------------------------------


def validate_meta(data: dict) -> list[str]:
    errors = []
    borders = data.get("borders")
    if not isinstance(borders, list):
        errors.append("borders is not a list")
    else:
        unknown = [item for item in borders if item not in BORDER_VALUES and item != "none"]
        if unknown:
            errors.append(f"borders has unknown values: {unknown}")
        if "none" in borders and len(borders) > 1:
            errors.append("borders mixes 'none' with directions")
    watermark = data.get("watermark")
    if not isinstance(watermark, list):
        errors.append("watermark is not a list")
    else:
        for index, item in enumerate(watermark):
            if not isinstance(item, dict):
                errors.append(f"watermark[{index}] is not an object")
                continue
            box = item.get("box_2d")
            if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
                errors.append(f"watermark[{index}].box_2d is not 4 numbers")
                continue
            if not all(0 <= value <= 1000 for value in box):
                errors.append(f"watermark[{index}].box_2d outside 0-1000")
            elif not (box[0] < box[2] and box[1] < box[3]):
                errors.append(f"watermark[{index}].box_2d has y_min>=y_max or x_min>=x_max")
            if not isinstance(item.get("label"), str) or not item.get("label").strip():
                errors.append(f"watermark[{index}].label missing")
    return errors


def validate_tags(data: dict, allow_identity: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["top level is not an object"]
    analysis = data.get("analysis")
    if not isinstance(analysis, str) or len(analysis.split()) < 10:
        errors.append("analysis missing or too short (it must be the first key)")
    errors.extend(validate_scene(data.get("scene"), allow_identity))
    errors.extend(validate_meta(data))
    return errors


def validate_scene(scene, allow_identity: bool = False) -> list[str]:
    if not isinstance(scene, str) or not scene.strip():
        return ["scene missing or empty"]
    errors = []
    words = scene.split()
    if len(words) > 30:
        errors.append(f"scene too long ({len(words)} words)")
    if not allow_identity:
        lowered = scene.lower()
        for word in BANNED_IN_SCENE:
            if word in lowered:
                errors.append(f"scene mentions identity feature: {word}")
                break
        else:
            for pattern in BANNED_SCENE_PATTERNS:
                match = pattern.search(scene)
                if match:
                    errors.append(f"scene mentions identity feature: {match.group(0)}")
                    break
    return errors


def validate_tag_style(style: dict, allow_identity: bool = False) -> list[str]:
    errors: list[str] = []
    caption = style.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        errors.append("caption missing or empty")
    errors.extend(validate_scene(style.get("scene"), allow_identity))
    errors.extend(validate_meta(style))
    return errors


def validate_official_style(style: dict) -> list[str]:
    errors: list[str] = []
    for field in ("caption_long", "caption_medium", "caption_short"):
        value = style.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} missing or empty")
    errors.extend(validate_meta(style))
    return errors


def validate_style(style_key: str, style, allow_identity: bool = False) -> list[str]:
    if not isinstance(style, dict):
        return [f"{style_key} is not an object"]
    if style_key == STYLE_TAG:
        return validate_tag_style(style, allow_identity)
    return validate_official_style(style)


def validate_sidecar(data: dict, allow_identity: bool = False,
                     styles=STYLES) -> list[str]:
    """Check a written sidecar: schema version + every style it actually carries."""
    if not isinstance(data, dict):
        return ["top level is not an object"]
    errors: list[str] = []
    if data.get("schema") != SCHEMA_VERSION:
        errors.append(f"schema is {data.get('schema')!r}, expected {SCHEMA_VERSION}")
    present = [key for key in STYLES if isinstance(data.get(key), dict)]
    if not present:
        errors.append(f"no caption style ({', '.join(STYLES)})")
    for key in present:
        if key not in styles and styles is not STYLES:
            continue
        for problem in validate_style(key, data[key], allow_identity):
            errors.append(f"{key}.{problem}")
    return errors


# -- sidecar read / merge / migrate -----------------------------------------
def read_sidecar(image: str, out_dir: str = "") -> dict:
    """Existing sidecar for this image (empty dict when missing or unreadable)."""
    path = sidecar_path(image, out_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def merge_sidecar(image: str, out_dir: str, style: str, payload: dict, *,
                  role: str) -> dict:
    """Write ``payload`` into ``sidecar[style]``, keeping every other style as-is."""
    data = read_sidecar(image, out_dir)
    if not isinstance(data, dict):
        data = {}
    data["schema"] = SCHEMA_VERSION
    data["tool"] = TOOL_NAME
    data["version"] = TOOL_VERSION
    data.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    data["role"] = role
    data[style] = payload
    return data


def reference_tags(image: str, out_dir: str = "") -> str:
    """Tags for the tag_caption style: the dataset's own ``.txt`` first, then the
    tags this tool kept last time."""
    reference = read_reference(image)
    if reference.strip():
        return reference
    data = read_sidecar(image, out_dir)
    style = data.get(STYLE_TAG) if isinstance(data.get(STYLE_TAG), dict) else {}
    kept = style.get("kept_tags")
    if isinstance(kept, list) and kept:
        return ", ".join(str(item) for item in kept if str(item).strip())
    stored = style.get("reference")
    return str(stored or "").strip()


def create_sidecar(image: str, out_dir: str, style: str, payload: dict, *, role: str,
                   tools: dict) -> dict:
    tools = dict(tools or {})
    tools.update({"tool": TOOL_NAME, "version": TOOL_VERSION})
    payload = dict(payload)
    payload.setdefault("generated", {}).update(tools)
    return merge_sidecar(image, out_dir, style, payload, role=role)


def write_sidecar(image: str, out_dir: str, data: dict) -> str:
    path = sidecar_path(image, out_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def migrate_sidecar(data: dict) -> tuple[dict, str]:
    """Old flat sidecar -> schema 2.  Returns ``(data, status)``.

    status is ``migrated`` / ``current`` / ``unknown``.  The old file's values are
    copied verbatim; nothing is guessed or regenerated.
    """
    if not isinstance(data, dict):
        return data, "unknown"
    if data.get("schema") == SCHEMA_VERSION:
        return data, "current"

    def generated(flat: dict) -> dict:
        out = {
            key: flat.get(key)
            for key in ("model", "temperature", "top_p", "thinking", "mode", "route")
            if flat.get(key) is not None
        }
        if flat.get("created"):
            out.setdefault("at", flat["created"])
        return {key: value for key, value in out.items() if value not in (None, "")}

    out: dict = {
        "schema": SCHEMA_VERSION,
        "tool": data.get("tool") or TOOL_NAME,
        "version": TOOL_VERSION,
        "created": data.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "role": data.get("role") or "train",
    }
    migrated = False
    if isinstance(data.get(STYLE_TAG), dict):
        out[STYLE_TAG] = data[STYLE_TAG]
        migrated = True
    elif isinstance(data.get("caption"), str) and data["caption"].strip():
        out[STYLE_TAG] = {
            "caption": data["caption"].strip(),
            "kept_tags": data.get("kept_tags") or [],
            "scene": data.get("scene") or "",
            "reference": data.get("reference") or "",
            "trigger": data.get("trigger") or "",
            "analysis": data.get("analysis") or "",
            "watermark": data.get("watermark") or [],
            "borders": data.get("borders") or ["none"],
            "generated": generated(data),
        }
        migrated = True
    if isinstance(data.get(STYLE_OFFICIAL), dict):
        out[STYLE_OFFICIAL] = data[STYLE_OFFICIAL]
        migrated = True
    elif any(isinstance(data.get(key), str) and data[key].strip()
             for key in ("caption_long", "caption_medium", "caption_short")):
        out[STYLE_OFFICIAL] = {
            "caption_long": data.get("caption_long") or "",
            "caption_medium": data.get("caption_medium") or "",
            "caption_short": data.get("caption_short") or "",
            "analysis": data.get("analysis") or "",
            "watermark": data.get("watermark") or [],
            "borders": data.get("borders") or ["none"],
            "generated": generated(data),
        }
        migrated = True
    return (out, "migrated") if migrated else (data, "unknown")


def last_json_object(text: str) -> str | None:
    """The final balanced {...} block: thinking models print prose before it."""
    end = text.rfind("}")
    if end < 0:
        return None
    depth = 0
    for index in range(end, -1, -1):
        char = text[index]
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                return text[index:end + 1]
    return None


# -- backend ----------------------------------------------------------------
def _tcp_ok(url: str, timeout: float = 3.0) -> bool:
    """先花 3 秒确认端口通不通：有些环境对没监听的端口直接丢 SYN，
    urlopen 会一直等到超时（几分钟），比连不上还难排查。"""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def open_url(request, timeout: float):
    if not _tcp_ok(request.full_url, timeout=min(3.0, timeout)):
        raise urllib.error.URLError(f"connect timeout: {request.full_url}")
    return urllib.request.urlopen(request, timeout=timeout)


def probe_structured(base: str, key: str) -> dict:
    for path in ("/status", "/props"):
        try:
            req = urllib.request.Request(f"{base}{path}", headers={"Authorization": f"Bearer {key}"})
            with open_url(req, 5) as response:
                data = json.loads(response.read().decode())
        except Exception:
            continue
        info = data.get("structured_output") if isinstance(data, dict) else None
        if isinstance(info, dict):
            return {
                "constrained": bool(info.get("constrained_decoding")),
                "method": info.get("method") or "",
                "formats": info.get("formats") or [],
            }
    return {"constrained": False, "method": "unknown", "formats": []}


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


def api_base() -> str:
    base = env("VLM_BASE_URL")
    if not base:
        raise SystemExit("VLM_BASE_URL is not set (use --base-url or the trainer settings)")
    return base.rstrip("/")


def api_key() -> str:
    key = env("VLM_API_KEY")
    if not key:
        raise SystemExit("VLM_API_KEY is not set (use --api-key or the trainer settings)")
    return key


def model_id() -> str:
    explicit = env("VLM_MODEL")
    if explicit:
        return explicit
    req = urllib.request.Request(f"{api_base()}/models", headers={"Authorization": f"Bearer {api_key()}"})
    with open_url(req, 10) as response:
        payload = json.loads(response.read().decode())
    models = payload.get("data") or []
    if not models:
        raise SystemExit("server reports no models")
    return models[0]["id"]


def service_alive(timeout: float = 5.0) -> bool:
    """Quick /models ping: 服务挂了就别让每张图各等 600 秒。"""
    try:
        request = urllib.request.Request(f"{api_base()}/models",
                                         headers={"Authorization": f"Bearer {api_key()}"})
        with open_url(request, timeout) as response:
            response.read(1)
        return True
    except Exception:
        return False


def encode_image(path: str) -> str:
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{data}"


def call_model(path: str, model: str, prompt: str, *, validator, response_format=None,
               temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 16384,
               thinking: bool = True) -> dict:
    body = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": encode_image(path)}},
            ],
        }],
    }
    if thinking:
        body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    if response_format is not None:
        body["response_format"] = response_format
    req = urllib.request.Request(
        f"{api_base()}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    with open_url(req, 600) as response:
        payload = json.loads(response.read().decode())
    text = payload["choices"][0]["message"]["content"] or ""
    block = last_json_object(text) or ""
    analysis = text[: text.rfind(block)].strip() if block else ""
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ValueError(f"no parsable JSON in reply: {exc}: {text[-160:]}") from exc
    if analysis:
        data["_analysis"] = analysis
    boxes = data.get("watermark")
    if isinstance(boxes, list):
        for item in boxes:
            box = item.get("box_2d") if isinstance(item, dict) else None
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                y0, x0, y1, x1 = box
                y0, y1 = min(y0, y1), max(y0, y1)
                x0, x1 = min(x0, x1), max(x0, x1)
                item["box_2d"] = [y0, x0, y1, x1]
    if isinstance(boxes, list):
        data["watermark"] = [
            item for item in boxes
            if not (isinstance(item, dict) and isinstance(item.get("box_2d"), list)
                    and len(item["box_2d"]) == 4
                    and (item["box_2d"][0] == item["box_2d"][2] or item["box_2d"][1] == item["box_2d"][3]))
        ]
    problems = validator(data)
    if problems:
        raise ValueError("schema: " + "; ".join(problems))
    return data


# -- prompts ----------------------------------------------------------------
def build_prompt(style: str, template: str, reference: str) -> str:
    """tag_caption 把原 tag 交给模型保留；official 无视现成标注，直接描述图片。"""
    if style == STYLE_TAG:
        return template.replace("{tags}", reference or "(none)")
    return template


def default_prompt(style: str) -> str:
    return TAGS_PROMPT if style == STYLE_TAG else NATURAL_PROMPT


def schema_for(style: str):
    return (SCHEMA_TAGS, validate_tags) if style == STYLE_TAG else (SCHEMA_NATURAL, validate_natural)


def parse_styles(raw: str) -> list[str]:
    wanted: list[str] = []
    for token in re.split(r"[,\s]+", str(raw or "")):
        key = token.strip().lower()
        if not key:
            continue
        style = STYLE_ALIASES.get(key, key)
        if style not in STYLES:
            raise SystemExit(f"unknown style: {token} (choose tag_caption / official)")
        if style not in wanted:
            wanted.append(style)
    return wanted or list(STYLES)


# -- worker -----------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MOSS-Image caption worker")
    parser.add_argument("--images", required=True, help="image directory")
    parser.add_argument("--out", default="", help="sidecar directory (default: next to each image)")
    parser.add_argument("--styles", default="tag_caption,official",
                        help="comma list: tag_caption / official (default: both)")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate styles that already exist (default: keep them)")
    parser.add_argument("--role", default="train", choices=("train", "reg"),
                        help=argparse.SUPPRESS)  # reg = 正则集：不加触发词、允许身份词
    parser.add_argument("--base-url", default="", help="OpenAI-compatible endpoint (else VLM_BASE_URL)")
    parser.add_argument("--api-key", default="",
                        help="bearer token, or the name of an env var holding it (else VLM_API_KEY)")
    parser.add_argument("--model", default="", help="model id (default: first the server reports)")
    parser.add_argument("--trigger", default="", help="trigger words prepended to every caption")
    parser.add_argument("--prompt-tags", default="", help="template for tag_caption (empty = built-in)")
    parser.add_argument("--prompt-natural", default="", help="template for official (empty = built-in)")
    parser.add_argument("--structured", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--no-thinking", action="store_true", help="disable the model's thinking mode")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt instead of calling the model")
    parser.add_argument("--validate-only", action="store_true", help="re-check existing sidecars")
    parser.add_argument("--migrate", action="store_true",
                        help="convert old flat sidecars to the style-keyed schema")
    args = parser.parse_args(argv)

    if args.base_url:
        os.environ["VLM_BASE_URL"] = args.base_url
    if args.api_key:
        # 既接受密钥本身，也接受环境变量名
        os.environ["VLM_API_KEY"] = os.environ.get(args.api_key, "").strip() or args.api_key
    if args.model:
        os.environ["VLM_MODEL"] = args.model
    is_reg = args.role == "reg"
    # 正则集是别的角色：不加触发词，否则等于教 LoRA「这些也是该角色」
    trigger = [] if is_reg else [item.strip() for item in args.trigger.split(",") if item.strip()]
    styles = parse_styles(args.styles)
    out_dir = os.path.abspath(os.path.expanduser(args.out)) if args.out else ""
    images_dir = os.path.abspath(os.path.expanduser(args.images))
    files = sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.lower().endswith(IMAGE_EXT)
    )[: args.limit or None]

    templates = {
        STYLE_TAG: _template(args.prompt_tags, STYLE_TAG),
        STYLE_OFFICIAL: _template(args.prompt_natural, STYLE_OFFICIAL),
    }

    if args.migrate:
        return _migrate(files, out_dir, is_reg)

    if args.validate_only:
        checked = bad = 0
        for path in files:
            sidecar = sidecar_path(path, out_dir)
            if not os.path.isfile(sidecar):
                continue
            checked += 1
            try:
                with open(sidecar, encoding="utf-8") as handle:
                    data = json.load(handle)
            except json.JSONDecodeError as exc:
                print(f"BAD  {os.path.basename(sidecar)}: invalid json ({exc})")
                bad += 1
                continue
            problems = validate_sidecar(data, is_reg, styles)
            if problems:
                print(f"BAD  {os.path.basename(sidecar)}: {'; '.join(problems)}")
                bad += 1
        print(f"validated {checked} sidecars, {bad} invalid")
        return 0 if bad == 0 else 1

    structured = False
    model = "dry-run"
    if not args.dry_run:
        model = model_id()
        if args.structured != "off":
            probe = probe_structured(api_base(), api_key())
            structured = True if args.structured == "on" else probe["constrained"]
            print(f"structured: {'constrained decoding' if structured else 'fallback (prompt + validate)'} "
                  f"[method={probe['method']}, constrained={probe['constrained']}]", flush=True)
    print(f"model: {model} · styles: {', '.join(styles)} · role: {args.role} · images: {len(files)} · "
          f"workers: {args.workers}", flush=True)

    started = time.time()
    done = 0
    counts = {STYLE_TAG: 0, STYLE_OFFICIAL: 0}
    skipped_reference = 0

    def generated() -> dict:
        return {
            "model": model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking": not args.no_thinking,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def tag_payload(data: dict, reference: str) -> dict:
        tags_raw = [item.strip() for item in re.split(r"[,\n]", reference) if item.strip()]
        kept = clean_tags(tags_raw, trigger)
        scene = str(data.get("scene") or "").strip()
        return {
            "caption": compose_caption(trigger, kept, scene),
            "kept_tags": kept,
            "scene": scene,
            "reference": reference,
            "trigger": ", ".join(trigger),
            "analysis": str(data.get("analysis") or data.get("_analysis") or "").strip(),
            "watermark": data.get("watermark") or [],
            "borders": data.get("borders") or ["none"],
            "generated": generated(),
        }

    def official_payload(data: dict) -> dict:
        tiers = data["caption"]
        return {
            "caption_long": compose_caption(trigger, [], "", long_text=tiers["long"].strip()),
            "caption_medium": compose_caption(trigger, [], "", long_text=tiers["medium"].strip()),
            "caption_short": compose_caption(trigger, [], "", long_text=tiers["short"].strip()),
            "trigger": ", ".join(trigger),
            "analysis": str(data.get("analysis") or data.get("_analysis") or "").strip(),
            "watermark": data.get("watermark") or [],
            "borders": data.get("borders") or ["none"],
            "generated": generated(),
        }

    def work(path: str):
        nonlocal skipped_reference
        existing = read_sidecar(path, out_dir)
        todo = [style for style in styles
                if args.overwrite or not isinstance(existing.get(style), dict)]
        if not todo:
            return "skip"
        reference = reference_tags(path, out_dir)
        if STYLE_TAG in todo and not reference.strip():
            # tag_caption 的意义就是保留原 tag：没有 .txt、也没有上次保留的 tag，就跳过
            todo.remove(STYLE_TAG)
            skipped_reference += 1
            if not todo:
                return "skip"
        done_styles = []
        for style in todo:
            prompt = build_prompt(style, templates[style], reference)
            if args.dry_run:
                print(f"--- style: {style}\n{prompt}")
                continue
            response_format = None
            if structured:
                schema, _ = schema_for(style)
                response_format = {"type": "json_schema",
                                   "json_schema": {"name": "caption", "schema": schema, "strict": True}}
            validator = (lambda data: validate_tags(data, is_reg)) if style == STYLE_TAG else validate_natural
            last = None
            for attempt in range(args.retries):
                try:
                    data = call_model(path, model, prompt, validator=validator,
                                      response_format=response_format,
                                      temperature=args.temperature, top_p=args.top_p,
                                      max_tokens=args.max_tokens, thinking=not args.no_thinking)
                    payload = tag_payload(data, reference) if style == STYLE_TAG else official_payload(data)
                    main_text = payload.get("caption") or payload.get("caption_long") or ""
                    if not str(main_text).strip():
                        raise ValueError("composed caption is empty")
                    write_sidecar(path, out_dir,
                                  merge_sidecar(path, out_dir, style, payload, role=args.role))
                    done_styles.append(style)
                    counts[style] = counts.get(style, 0) + 1
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError,
                        json.JSONDecodeError) as exc:
                    last = exc
                    if isinstance(exc, urllib.error.URLError) and not service_alive():
                        # 服务没了：重试只会让每张图各卡几分钟，直接报错退出这一张
                        raise RuntimeError(
                            f"{os.path.basename(path)} [{style}]: 视觉服务连不上（{api_base()}）") from exc
                    time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"{os.path.basename(path)} [{style}]: {last}")
        return "dry" if args.dry_run else ("ok" if done_styles else "skip")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, path): path for path in files}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                status = future.result()
            except RuntimeError as exc:
                print(f"FAIL {exc}", flush=True)
                continue
            done += 1
            rate = done / max(time.time() - started, 1e-6)
            print(f"{done}/{len(files)}  {status}  {os.path.basename(path)}  {rate:.2f} img/s", flush=True)
    if not args.dry_run:
        print(f"styles: tag_caption {counts[STYLE_TAG]} · official {counts[STYLE_OFFICIAL]} · "
              f"skipped without reference: {skipped_reference}", flush=True)
    return 0


def _template(path: str, style: str) -> str:
    """Template from a file path, falling back to the built-in default."""
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    if path and ("\n" in path or "{tags}" in path or len(path) > 200):
        return path  # 已经是一段提示词文本
    return default_prompt(style)


def _migrate(files, out_dir: str, is_reg: bool) -> int:
    migrated = current = unknown = 0
    for path in files:
        sidecar = sidecar_path(path, out_dir)
        if not os.path.isfile(sidecar):
            continue
        try:
            with open(sidecar, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"BAD  {os.path.basename(sidecar)}: {exc}")
            unknown += 1
            continue
        data, status = migrate_sidecar(data)
        if status == "migrated":
            write_sidecar(path, out_dir, data)
            migrated += 1
        elif status == "current":
            current += 1
        else:
            print(f"UNKNOWN {os.path.basename(sidecar)}")
            unknown += 1
        problems = validate_sidecar(data, is_reg)
        if problems:
            print(f"BAD  {os.path.basename(sidecar)}: {'; '.join(problems)}")
    print(f"migrated {migrated} · already schema 2: {current} · unknown: {unknown}", flush=True)
    return 0 if unknown == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
