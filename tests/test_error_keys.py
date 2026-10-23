"""错误提示的键必须齐：代码里用到的每个错误码都要有 manifest 和四语言译文。

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/test_error_keys.py

翻译分支交付前这个测试会故意失败：新增的 ``err.*`` 键还没写进 i18n 包。
除键是否齐全外，它还检查每条译文的 ``{placeholder}`` 集合与中文完全一致。
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from moss_lora.web.errors import CODES  # noqa: E402

I18N_DIR = os.path.join(ROOT, "moss_lora", "web", "static", "i18n")
MANIFEST = os.path.join(ROOT, ".tmp", "err-manifest.json")
LANGUAGES = ("zh", "en", "ja", "ko")
NAME_KINDS = ("workspace", "run", "name", "config")
CODE_RE = re.compile(r"^err\.[a-z0-9_.]+$")
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def used_codes() -> set[str]:
    """Every ``err.*`` string literal the web layer can emit."""
    codes: set[str] = set()
    for dirpath, _dirs, names in os.walk(os.path.join(ROOT, "moss_lora", "web")):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(names):
            if not name.endswith(".py") or name == "errors.py":
                continue  # errors.py holds the registry itself, not a raise site
            path = os.path.join(dirpath, name)
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                text = None
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    text = node.value
                elif isinstance(node, ast.JoinedStr):
                    # f"err.name.empty.{kind}" -> prefix ends with "."
                    parts = []
                    for value in node.values:
                        parts.append(value.value if isinstance(value, ast.Constant) else "\0")
                    text = "".join(parts)
                if not text or not text.startswith("err."):
                    continue
                if "\0" in text:
                    prefix = text.split("\0", 1)[0]
                    if prefix.startswith("err.name."):
                        for kind in NAME_KINDS:
                            codes.add(prefix + kind)
                    continue
                if text.endswith("."):
                    if text.startswith("err.name."):
                        for kind in NAME_KINDS:
                            codes.add(text + kind)
                    continue
                if CODE_RE.match(text):
                    codes.add(text)
    return codes


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def check_codes_known(codes: set[str]) -> list[str]:
    unknown = sorted(code for code in codes if code not in CODES)
    assert not unknown, f"代码里用了未登记的错误码：{unknown}"
    return sorted(codes)


def check_manifest(codes: list[str]) -> None:
    if not os.path.isfile(MANIFEST):
        print(f"skip manifest check: {os.path.relpath(MANIFEST, ROOT)} 不存在")
        return
    manifest = load_json(MANIFEST)
    missing = [code for code in codes if code not in manifest]
    assert not missing, f"manifest 缺这些码：{missing}"
    for code in codes:
        entry = manifest[code]
        for key in ("zh", "params", "note", "sites"):
            assert key in entry, f"{code} 缺少 {key}"
        fields = set(PLACEHOLDER_RE.findall(entry["zh"]))
        assert fields == set(entry["params"]), f"{code} 的占位符与 params 不一致"


def check_translations(codes: list[str]) -> list[str]:
    packs = {code: load_json(os.path.join(I18N_DIR, f"{code}.json")) for code in LANGUAGES}
    problems = []
    for code in codes:
        key = code
        zh = packs["zh"].get(key)
        if not zh:
            problems.append(f"zh 缺 {key}")
            continue
        for language in LANGUAGES[1:]:
            value = packs[language].get(key)
            if not value:
                problems.append(f"{language} 缺 {key}")
                continue
            if set(PLACEHOLDER_RE.findall(value)) != set(PLACEHOLDER_RE.findall(zh)):
                problems.append(f"{language} 的 {key} 占位符与 zh 不一致")
    return problems


def check_registry_matches_pack() -> None:
    """The Chinese template in ``CODES`` is the source every pack was translated
    from, so it must be the same string the zh pack serves. Letting the two drift
    makes the server's fallback message disagree with the UI's own language."""
    zh = load_json(os.path.join(I18N_DIR, "zh.json"))
    drift = sorted(
        f"{code}: 注册表 {template!r} ≠ 语言包 {zh.get(code)!r}"
        for code, template in CODES.items()
        if zh.get(code) != template
    )
    assert not drift, "中文模板与 zh 语言包不一致：\n  " + "\n  ".join(drift)


def main() -> None:
    codes = check_codes_known(used_codes())
    check_manifest(codes)
    check_registry_matches_pack()
    print(f"错误码 {len(codes)} 个，manifest 与代码一致")
    problems = check_translations(codes)
    if problems:
        print("译文未就绪（翻译分支交付前属正常）：")
        for problem in problems[:20]:
            print("  -", problem)
        if len(problems) > 20:
            print(f"  … 另有 {len(problems) - 20} 条")
        raise SystemExit(1)
    print("四语言译文齐全，占位符一致")


if __name__ == "__main__":
    main()
