"""Sidecar schema 2：两种风格共用一个文件、只更新对应风格、旧扁平格式能迁移。

    .venv/bin/python tests/test_sidecar.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.captioner import (  # noqa: E402
    SCHEMA_VERSION,
    STYLE_OFFICIAL,
    STYLE_TAG,
    merge_sidecar,
    migrate_sidecar,
    read_sidecar,
    reference_tags,
    sidecar_path,
    validate_sidecar,
    write_sidecar,
)

TAG = {
    "caption": "mychar, 1girl, park, a girl standing in a park",
    "kept_tags": ["1girl", "park"],
    "scene": "a girl standing in a park, full-body, evening light",
    "reference": "1girl, park",
    "trigger": "mychar",
    "analysis": "framing, pose, background",
    "watermark": [],
    "borders": ["none"],
    "generated": {"model": "fake"},
}
OFFICIAL = {
    "caption_long": "A girl stands in a park at dusk with soft light behind her.",
    "caption_medium": "A girl standing in a park at dusk, soft light behind her.",
    "caption_short": "a girl in a park at dusk",
    "trigger": "mychar",
    "analysis": "framing",
    "watermark": [],
    "borders": ["none"],
    "generated": {"model": "fake"},
}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        image = os.path.join(tmp, "pic.png")
        open(image, "wb").close()

        # 1. 只有 tag 风格时写出来就是只有它
        write_sidecar(image, "", merge_sidecar(image, "", STYLE_TAG, dict(TAG), role="train"))
        data = read_sidecar(image)
        assert data["schema"] == SCHEMA_VERSION and STYLE_TAG in data, data
        assert STYLE_OFFICIAL not in data
        assert not validate_sidecar(data), validate_sidecar(data)

        # 2. 再生成官方风格：tag 风格原样保留
        write_sidecar(image, "", merge_sidecar(image, "", STYLE_OFFICIAL, dict(OFFICIAL), role="train"))
        data = read_sidecar(image)
        assert data[STYLE_TAG]["caption"] == TAG["caption"], data
        assert STYLE_OFFICIAL in data and not validate_sidecar(data), validate_sidecar(data)

        # 3. 覆盖同一风格：只换这一块
        new_tag = dict(TAG, caption="mychar, 1girl, park, a rewritten scene")
        write_sidecar(image, "", merge_sidecar(image, "", STYLE_TAG, new_tag, role="train"))
        data = read_sidecar(image)
        assert data[STYLE_TAG]["caption"] == new_tag["caption"]
        assert data[STYLE_OFFICIAL]["caption_short"] == OFFICIAL["caption_short"]

        # 4. 参考 tag：.txt 优先，没有就用上次保留的 tag
        assert reference_tags(image) == "1girl, park"
        txt = os.path.splitext(image)[0] + ".txt"
        with open(txt, "w", encoding="utf-8") as handle:
            handle.write("1girl, beach\n")
        assert reference_tags(image) == "1girl, beach"
        os.remove(txt)

        # 5. 旧扁平格式迁移：值原样搬进对应风格
        legacy_tag = os.path.join(tmp, "legacy_tag.png")
        open(legacy_tag, "wb").close()
        flat = {
            "caption": "mychar, 1girl, old caption",
            "kept_tags": ["1girl"],
            "scene": "an old scene line",
            "route": "tags",
            "role": "train",
            "trigger": "mychar",
            "reference": "1girl",
            "watermark": [],
            "borders": ["none"],
            "model": "old-model",
            "temperature": 1.0,
            "thinking": True,
            "created": "2026-10-01T00:00:00+0900",
        }
        with open(sidecar_path(legacy_tag), "w", encoding="utf-8") as handle:
            json.dump(flat, handle)
        migrated, status = migrate_sidecar(read_sidecar(legacy_tag))
        assert status == "migrated" and STYLE_TAG in migrated, (status, migrated)
        assert migrated[STYLE_TAG]["caption"] == flat["caption"]
        assert migrated[STYLE_TAG]["generated"]["model"] == "old-model"
        assert migrated["role"] == "train"
        assert not validate_sidecar(migrated), validate_sidecar(migrated)
        assert migrate_sidecar(migrated)[1] == "current"

        # 官方风格的旧扁平文件
        legacy_official = {
            "caption_long": "long text here for the test case",
            "caption_medium": "medium text",
            "caption_short": "short",
            "route": "natural",
            "role": "reg",
            "watermark": [],
            "borders": ["none"],
        }
        migrated2, status2 = migrate_sidecar(legacy_official)
        assert status2 == "migrated" and STYLE_OFFICIAL in migrated2, (status2, migrated2)
        assert migrated2[STYLE_OFFICIAL]["caption_long"] == legacy_official["caption_long"]
        assert migrated2["role"] == "reg"

        # 认不出来的文件不瞎猜
        assert migrate_sidecar({"foo": "bar"})[1] == "unknown"

    print("sidecar schema 2 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
