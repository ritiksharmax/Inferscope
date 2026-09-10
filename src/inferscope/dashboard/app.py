"""A local dashboard for a trace file.

    python -m inferscope.dashboard --db traces.db

Reads the SQLite file the tracer writes, so it serves finished recordings and
live ones the same way -- point it at a database an engine is still writing to
and it will keep up. Needs the ``dashboard`` extra.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inferscope.dashboard.data import build_payload, request_detail
from inferscope.trace import Trace

HERE = Path(__file__).parent
INDEX = HERE / "index.html"


def create_app(db_path: str | Path) -> Any:
    """Build the FastAPI app for a trace file.

    The trace is re-read per request rather than cached: these are local files
    of a few MB, and staleness on a live trace would be much more annoying than
    the millisecond it costs.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    path = Path(db_path)
    app = FastAPI(title="inferscope", docs_url=None, redoc_url=None)

    def load() -> Trace:
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"no trace at {path}")
        return Trace.from_sqlite(path)

    @app.get("/")
    def index() -> Any:
        return FileResponse(INDEX)

    @app.get("/api/payload")
    def payload() -> Any:
        return JSONResponse(build_payload(load()))

    @app.get("/api/request/{request_id}")
    def detail(request_id: str) -> Any:
        try:
            return JSONResponse(request_detail(load(), request_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="SQLite trace to serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    print(f"\n  inferscope dashboard -> http://{args.host}:{args.port}\n")
    uvicorn.run(create_app(args.db), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
