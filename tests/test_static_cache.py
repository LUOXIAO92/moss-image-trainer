"""前端静态资源不得返回 304 的回归测试。

前端没有构建步骤，模块按 URL 缓存。一旦浏览器拿到 304，它就用旧模块去配新
HTML；如果这期间某个模块多了一个导出，整张模块图链接失败，页面白成一片。
所以静态资源必须每次都 200 且不许存。

    .venv/bin/python tests/test_static_cache.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moss_lora.web.server import STATIC_CACHE_CONTROL, create_app  # noqa: E402

# 一个远在未来的时间戳：老行为（StaticFiles 自己比对 mtime）会据此回 304。
FUTURE = "Wed, 21 Oct 2099 07:28:00 GMT"


def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def check_html_uncached(http):
    response = await http.get("/")
    assert response.status_code == 200, response.status_code
    assert response.headers.get("cache-control") == STATIC_CACHE_CONTROL, dict(response.headers)
    assert "etag" not in response.headers, "静态资源不该给浏览器任何校验器"
    assert "last-modified" not in response.headers, "静态资源不该给浏览器任何校验器"


async def check_conditional_request_still_200(http):
    for path in ("/js/main.js", "/js/pages/cache.js", "/pages/train.html"):
        plain = await http.get(path)
        assert plain.status_code == 200, (path, plain.status_code)
        assert plain.content, path

        revalidated = await http.get(path, headers={
            "if-modified-since": FUTURE,
            "if-none-match": '"stale-tag"',
        })
        assert revalidated.status_code == 200, f"{path} 回了 {revalidated.status_code}，浏览器会留住旧模块"
        assert revalidated.content == plain.content, path
        assert revalidated.headers.get("cache-control") == STATIC_CACHE_CONTROL, path


async def check_api_routes_survive_the_mount(http):
    """静态挂载在 "/" 上，API 必须仍然先匹配。"""
    response = await http.get("/api/i18n")
    assert response.status_code == 200, response.status_code
    body = response.json()
    assert isinstance(body, (dict, list)) and body, body


async def check_missing_module_is_404(http):
    assert (await http.get("/js/does-not-exist.js")).status_code == 404


async def main():
    async with client() as http:
        await check_html_uncached(http)
        await check_conditional_request_still_200(http)
        await check_api_routes_survive_the_mount(http)
        await check_missing_module_is_404(http)
    print("static cache checks passed")


if __name__ == "__main__":
    asyncio.run(main())
