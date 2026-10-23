# moved from app.py: L119, L2452-2455, L2702-2719
"""Application assembly: routes, static mount and the ``main()`` entry point.

Router import order below is the first-appearance order of the routes in the
old single-module app; within each router the declaration order is unchanged,
which is what keeps wildcard routes (``/api/i18n/{code}``) ahead of the
literal siblings they could otherwise swallow.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from .deps import STATIC
from .routers import (config, dataset, estimate, export, files, i18n, jobs, samples,
                      settings, system, validate, workspaces)

# The front end ships as loose files with no build step and no asset hashes, so
# the browser caches modules by URL alone. A conditional request then answers
# 304 with whatever that URL held last time, and a page load can pair the HTML
# just written with a module graph from an older edit. When a module gained an
# export in the meantime the graph fails to link and the app renders an empty
# page. Serving every static file as uncached keeps a reload truthful.
STATIC_CACHE_CONTROL = "no-store"


class UncachedStaticFiles(StaticFiles):
    """Static files that never answer 304."""

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
        response.headers["cache-control"] = STATIC_CACHE_CONTROL
        # Validators are useless without a stored copy, and their presence
        # invites an intermediary to revalidate on the browser's behalf.
        for header in ("etag", "last-modified"):
            if header in response.headers:
                del response.headers[header]
        return response


def _flatten_included_routes(app: FastAPI) -> None:
    """Expand lazy ``include_router`` markers into the app's own route list.

    FastAPI 0.146 keeps an included router as a single deferred marker and
    resolves it while matching.  The pre-split app registered every route flat
    on the app, so expanding the markers here keeps ``app.routes`` -- and the
    exact first-match ordering -- identical to the old single-module app.
    """
    routes = app.router.routes
    index = 0
    while index < len(routes):
        route = routes[index]
        child = getattr(route, "original_router", None)
        if child is None or not hasattr(child, "routes"):
            index += 1
            continue
        routes[index:index + 1] = list(child.routes)


def create_app() -> FastAPI:
    app = FastAPI(title="MOSS-Image LoRA trainer")
    app.include_router(i18n.router)
    app.include_router(system.router)
    app.include_router(settings.router)
    app.include_router(workspaces.router)
    app.include_router(dataset.router)
    app.include_router(config.router)
    app.include_router(jobs.router)
    app.include_router(export.router)
    app.include_router(files.router)
    app.include_router(validate.router)
    app.include_router(samples.router)
    # Appended last: the routes above keep the first-appearance order, and this
    # literal path cannot shadow anything.
    app.include_router(estimate.router)
    _flatten_included_routes(app)

    @app.get("/favicon.ico")
    def favicon():
        """Browsers still probe this path even when the page declares a data icon."""
        return Response(status_code=204)

    if os.path.isdir(STATIC):
        app.mount("/", UncachedStaticFiles(directory=STATIC, html=True), name="static")
    return app


app = create_app()


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
