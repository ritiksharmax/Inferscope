"""Dashboard: the payload the page renders, and that the page can render it."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from inferscope import Tracer
from inferscope.dashboard.data import build_payload, request_detail
from inferscope.trace import Trace
from inferscope_lab.pathologies import SCENARIOS, run_workload

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "src" / "inferscope" / "dashboard" / "index.html"
RENDER_TEST = Path(__file__).parent / "js" / "render-test.mjs"


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("trace") / "traces.db"
    tracer = Tracer(f"sqlite://{path}")
    run_workload(SCENARIOS["prefill-starvation"](), tracer)
    tracer.close()
    return path


def test_payload_has_every_panel(db: Path) -> None:
    p = build_payload(Trace.from_sqlite(db))
    assert set(p) >= {"trace", "window", "timeline", "diagnosis", "ttft", "tpot", "requests"}
    assert p["diagnosis"]["pathology"] == "prefill-starvation"
    assert p["requests"]
    assert len(p["timeline"]["t"]) == len(p["timeline"]["batch"])
    assert json.dumps(p), "payload must be JSON-serialisable"


def test_requests_are_ordered_slowest_first(db: Path) -> None:
    rows = build_payload(Trace.from_sqlite(db))["requests"]
    totals = [r["total_ms"] for r in rows]
    assert totals == sorted(totals, reverse=True)


def test_request_breakdowns_do_not_exceed_their_total(db: Path) -> None:
    for r in build_payload(Trace.from_sqlite(db))["requests"]:
        assert sum(r["parts"].values()) <= r["total_ms"] * 1.01


def test_histograms_count_every_sample(db: Path) -> None:
    trace = Trace.from_sqlite(db)
    p = build_payload(trace)
    from inferscope.metrics import per_request

    tpots = [x for m in per_request(trace).values() for x in m.tpot_samples]
    assert sum(p["tpot"]["counts"]) == len(tpots), "a log histogram must not drop samples"


def test_empty_trace_yields_a_renderable_payload() -> None:
    p = build_payload(Trace(events=[], names={}))
    assert p["diagnosis"]["pathology"] == "no-data"
    assert p["requests"] == []
    assert json.dumps(p)


def test_request_detail_round_trips(db: Path) -> None:
    trace = Trace.from_sqlite(db)
    worst = build_payload(trace)["diagnosis"]["worst_request"]
    d = request_detail(trace, worst)
    assert d["id"] == worst
    assert d["verdict"]
    assert sum(d["components"].values()) <= d["total_ms"] * 1.01


def test_trace_carries_a_wall_clock_anchor(db: Path) -> None:
    assert Trace.from_sqlite(db).epoch_offset_ns != 0


def test_api_serves_the_page_and_the_data(db: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from inferscope.dashboard.app import create_app

    client = TestClient(create_app(db))
    assert client.get("/").status_code == 200
    payload = client.get("/api/payload")
    assert payload.status_code == 200
    worst = payload.json()["diagnosis"]["worst_request"]
    assert client.get(f"/api/request/{worst}").status_code == 200
    assert client.get("/api/request/not-a-request").status_code == 404


def test_api_reports_a_missing_trace_file(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from inferscope.dashboard.app import create_app

    client = TestClient(create_app(tmp_path / "nope.db"), raise_server_exceptions=False)
    assert client.get("/api/payload").status_code == 404


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_page_actually_renders_every_scenario(tmp_path: Path) -> None:
    """Run the page's own JS against real payloads under a minimal DOM shim.

    Static checks cannot catch a chart that divides by zero on an empty series
    or writes NaN into an SVG coordinate; the shim rejects both.
    """
    for name in SCENARIOS:
        tracer = Tracer(f"sqlite://{tmp_path / name}.db")
        run_workload(SCENARIOS[name](), tracer)
        tracer.close()
        payload = build_payload(Trace.from_sqlite(tmp_path / f"{name}.db"))
        (tmp_path / f"payload-{name}.json").write_text(json.dumps(payload))

    script = INDEX.read_text().split("<script>")[1].split("</script>")[0]
    (tmp_path / "dash.js").write_text(script)

    result = subprocess.run(
        ["node", str(RENDER_TEST), str(tmp_path / "dash.js"), str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("OK") == len(SCENARIOS)
