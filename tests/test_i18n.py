"""语言包自检：四套内置语言的 key 必须和中文一致，占位符不能丢。

    .venv/bin/python tests/test_i18n.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora import i18n as pack  # noqa: E402

PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}", re.I)


def main() -> int:
    zh = pack.entries(pack.load_builtin("zh"))
    if not zh:
        print("skip: moss_lora/web/static/i18n/zh.json 还不存在")
        return 0
    problems: list[str] = []
    zh_places = {key: set(PLACEHOLDER.findall(value)) for key, value in zh.items()}
    for code in pack.BUILTIN_CODES:
        entries = pack.entries(pack.load_builtin(code))
        if not entries:
            problems.append(f"{code}: 语言包缺失或为空")
            continue
        missing = sorted(set(zh) - set(entries))
        extra = sorted(set(entries) - set(zh))
        if missing:
            problems.append(f"{code}: 缺 {len(missing)} 条，例如 {missing[:5]}")
        if extra:
            problems.append(f"{code}: 多出 {len(extra)} 条，例如 {extra[:5]}")
        empty = [key for key, value in entries.items() if not str(value).strip()]
        if empty:
            problems.append(f"{code}: 空值 {len(empty)} 条，例如 {empty[:5]}")
        for key, value in entries.items():
            lost = zh_places.get(key, set()) - set(PLACEHOLDER.findall(value))
            if lost:
                problems.append(f"{code}: {key} 丢了占位符 {sorted(lost)}")
    names = pack.pack_languages()
    for code in pack.BUILTIN_CODES:
        if code not in names:
            problems.append(f"{code}: 没有登记到语言列表")
    if problems:
        print("\n".join(f"- {line}" for line in problems))
        return 1
    print(f"i18n packs ok: {len(zh)} keys x {len(pack.BUILTIN_CODES)} languages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
