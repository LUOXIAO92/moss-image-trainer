"""打标密钥的三条规则，不联网也能跑：

* 填环境变量名 → 存名字，跑任务时从环境里取
* 直接填密钥 → 只活在进程内存里，不写设置文件
* 密钥被服务拒绝 → 报错、什么都不保存

    .venv/bin/python tests/test_caption_key.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 连不上的地址：端口 1 会立刻 connection refused，不会卡 8 秒
DEAD = {"base_url": "http://127.0.0.1:1/v1", "model": "m"}


def main() -> int:
    os.environ["TEST_VLM_KEY"] = "key_from_env"
    from moss_lora.web import deps
    from moss_lora.web.services import caption as A
    from moss_lora.web.services import settings_svc

    # 拆包后：密钥规划在 services/settings_svc.py，运行时的那把钥匙存在 deps 里。
    A._plan_api_key = settings_svc._plan_api_key

    # 环境变量名：存名字，运行时解出真值
    error, _, stored, literal = A._plan_api_key(dict(DEAD), "TEST_VLM_KEY")
    assert not error and stored == "TEST_VLM_KEY" and literal == "", (error, stored, literal)
    deps._LITERAL_API_KEY = ""
    key, source = A.resolve_api_key({"api_key": "TEST_VLM_KEY"})
    assert (key, source) == ("key_from_env", "env"), (key, source)

    # 直接填密钥：不落盘，只进内存
    error, note, stored, literal = A._plan_api_key(dict(DEAD), "key_literal_123")
    assert not error and stored == "" and literal == "key_literal_123", (error, stored, literal)
    assert "不写进设置文件" in note, note
    deps._LITERAL_API_KEY = literal
    assert A.resolve_api_key({"api_key": ""}) == ("key_literal_123", "literal")

    # 空串 = 清空；不带这个键 = 保持原样（由 merge 保证，这里只验清空）
    assert A._plan_api_key(dict(DEAD), "") == (None, "", "", "")
    deps._LITERAL_API_KEY = ""

    # 旧版 settings.yaml 里直接存的密钥仍然可用（长得不像变量名）
    deps._LITERAL_API_KEY = ""
    key, source = A.resolve_api_key({"api_key": "key_legacy_from_file"})
    assert (key, source) == ("key_legacy_from_file", "legacy"), (key, source)

    # 存的是变量名、但进程里没这个变量：不能拿名字当密钥使
    deps._LITERAL_API_KEY = ""
    key, source = A.resolve_api_key({"api_key": "SOME_UNSET_VLM_KEY"})
    assert (key, source) == ("", "missing"), (key, source)

    print("caption key rules passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
