"""UI language packs.

内置四套语言随代码走（``moss_lora/web/static/i18n/<code>.json``），用户自己的包放在
项目根 ``i18n/<code>.json``（gitignore，只在本机生效）：

* 用户包可以覆盖内置语言（zh/en/ja/ko），删掉就还原内置；
* 也可以新增任意 ``[a-z][a-z0-9-]{1,7}`` 代码的语言，删掉即移除；
* 任何 key 缺失都会回退到中文，再回退到 key 本身。

包格式（模板就是它，导出后照着填）：

```json
{
  "$code": "ja",
  "$name": "日本語",
  "nav.dashboard": "中控台",
  "dash.train.summary": "当前配置 {path} · {steps} 步"
}
```

``$`` 开头的是元信息，其余每一项都是「key → 该语言的文字」。
"""

from __future__ import annotations

import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTIN_DIR = os.path.join(PROJECT_ROOT, "moss_lora", "web", "static", "i18n")
USER_DIR = os.path.join(PROJECT_ROOT, "i18n")

DEFAULT_CODE = "zh"
BUILTIN_CODES = ("zh", "en", "ja", "ko")
CODE_RE = re.compile(r"^[a-z][a-z0-9-]{1,7}$")
LANGUAGE_NAMES = {"zh": "简体中文", "en": "English", "ja": "日本語", "ko": "한국어"}


def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


class LanguageError(ValueError):
    """A language code or pack the loader rejected, as a code the UI translates.

    It subclasses ``ValueError`` so callers that only care about failure keep
    working, while the web layer can hand the code straight to ``fail()``.
    """

    def __init__(self, code: str, **params):
        super().__init__(code)
        self.code = code
        self.params = params


def normalize_code(value: str) -> str:
    code = str(value or "").strip().lower()
    if not CODE_RE.match(code):
        raise LanguageError("err.lang.code_invalid")
    return code


def entries(pack: dict) -> dict:
    """A pack minus its ``$`` metadata."""
    return {key: value for key, value in (pack or {}).items()
            if not str(key).startswith("$") and isinstance(value, str)}


def user_path(code: str) -> str:
    return os.path.join(USER_DIR, f"{code}.json")


def builtin_path(code: str) -> str:
    return os.path.join(BUILTIN_DIR, f"{code}.json")


def load_builtin(code: str) -> dict:
    return _read_json(builtin_path(code))


def load_user(code: str) -> dict:
    return _read_json(user_path(code))


def pack_languages() -> dict:
    """All language files we can see, builtin first then user-only ones."""
    found: dict[str, dict] = {}
    for code in BUILTIN_CODES:
        pack = load_builtin(code)
        found[code] = {
            "code": code,
            "name": str(pack.get("$name") or LANGUAGE_NAMES.get(code, code)),
            "builtin": True,
            "custom": False,
        }
    if os.path.isdir(USER_DIR):
        for name in sorted(os.listdir(USER_DIR)):
            if not name.endswith(".json"):
                continue
            code = name[:-5]
            if not CODE_RE.match(code):
                continue
            pack = load_user(code)
            entry = found.setdefault(code, {
                "code": code,
                "name": str(pack.get("$name") or code),
                "builtin": code in BUILTIN_CODES,
                "custom": code not in BUILTIN_CODES,
            })
            if not entry.get("name") or entry["name"] == code:
                entry["name"] = str(pack.get("$name") or code)
            entry["builtin"] = code in BUILTIN_CODES
            entry["custom"] = code not in BUILTIN_CODES
    for code, entry in found.items():
        builtin_keys = set(entries(load_builtin(code)))
        user = entries(load_user(code))
        entry["overridden"] = bool(user)
        entry["keys"] = len(user) or len(builtin_keys)
        if user and builtin_keys:
            entry["missing"] = len(builtin_keys - set(user))
        elif not user:
            entry["missing"] = 0
        else:
            entry["missing"] = 0
    return dict(sorted(found.items(), key=lambda item: (not item[1]["builtin"], item[0])))


def merged(code: str) -> dict:
    """English/Japanese/Korean/user packs layered on top of Chinese."""
    code = str(code or DEFAULT_CODE).strip().lower()
    base = entries(load_builtin(DEFAULT_CODE)) or entries(load_user(DEFAULT_CODE))
    if code == DEFAULT_CODE:
        return dict(base)
    layer = entries(load_builtin(code))
    layer.update(entries(load_user(code)))
    merged_pack = dict(base)
    merged_pack.update(layer)
    return merged_pack


def template(base: str = DEFAULT_CODE) -> dict:
    """Everything a translator needs: metadata + every key with its base text."""
    payload = merged(base)
    out = {"$code": "", "$name": "", "$base": base}
    out.update(dict(sorted(payload.items())))
    return out


def save_pack(code: str, payload: dict) -> dict:
    """Write ``i18n/<code>.json`` from a template-shaped payload (``$`` keys kept)."""
    code = normalize_code(code)
    incoming = payload if isinstance(payload, dict) else {}
    name = str(incoming.get("$name") or LANGUAGE_NAMES.get(code, code)).strip() or code
    values = entries(incoming)
    if not values:
        raise LanguageError("err.lang.pack_empty")
    data = {"$code": code, "$name": name}
    data.update(dict(sorted(values.items())))
    os.makedirs(USER_DIR, exist_ok=True)
    path = user_path(code)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return {"code": code, "name": name, "path": path, "keys": len(values),
            "builtin": code in BUILTIN_CODES}


def delete_pack(code: str) -> dict:
    """Remove the user pack; a builtin language falls back to its shipped text."""
    code = normalize_code(code)
    path = user_path(code)
    existed = os.path.isfile(path)
    if existed:
        os.remove(path)
    return {"code": code, "removed": existed, "builtin": code in BUILTIN_CODES,
            "restored": existed and code in BUILTIN_CODES,
            "deleted_language": existed and code not in BUILTIN_CODES}


def resolve(code: str) -> str:
    """A usable language code: falls back to Chinese when the pack is gone."""
    try:
        wanted = str(code or DEFAULT_CODE).strip().lower()
    except Exception:
        return DEFAULT_CODE
    if wanted in pack_languages():
        return wanted
    return DEFAULT_CODE
