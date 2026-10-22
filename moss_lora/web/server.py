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

from .deps import STATIC
from .routers import (config, dataset, export, files, i18n, jobs, samples, settings,
                      system, validate, workspaces)


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
    _flatten_included_routes(app)

    @app.get("/favicon.ico")
    def favicon():
        """Browsers still probe this path even when the page declares a data icon."""
        return Response(status_code=204)

    if os.path.isdir(STATIC):
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
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
