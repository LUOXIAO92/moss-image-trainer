"""兼容入口：Web 后端已拆到 ``server.py`` + ``routers/`` + ``services/``。

保留本模块只是为了让 ``moss_lora.cli web``（``from .web.app import main``）和
按旧路径 import 的脚本继续工作；新代码请直接 import ``moss_lora.web.server``。
"""
from __future__ import annotations

from .server import app, create_app, main

__all__ = ["app", "create_app", "main"]
