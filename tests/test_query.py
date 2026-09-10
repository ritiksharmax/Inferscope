"""The four questions, checked against workloads with known answers.

This is the point of the lab: a query that "detects" a pathology is only
credible if it is run against a scenario that provably has one, and stays quiet
on one that does not.
"""

from __future__ import annotations

import pytest

from inferscope import MemorySink, Tracer
from inferscope.query import (
    batch_context,
    diagnose,
    explain_latency,
    kv_pressure,
    latency_regime,
)
from inferscope.trace import Trace
from inferscope_lab.pathologies import SCENARIOS, run_workload

#: What each reference scenario must be diagnosed as.
EXPECTED = {
    "healthy": "healthy",
    "prefill-starvation": "prefill-starvation",
    "kv-thrash": "kv-pressure",
    "batch-starvation": "admission-starvation",
}


def traced(scenario: str, decode_mode: str = "aggregate") -> Trace:
    tracer = Tracer(MemorySink(), decode_mode=decode_mode, autostart=False)
    run_workload(SCENARIOS[scenario](), tracer)
    tracer.flush()
    trace = Trace.from_sink(tracer.sink)
    tracer.close()
    return trace


# -- the exit criterion ----------------------------------------------------


@pytest.mark.parametrize("scenario,expected", sorted(EXPECTED.items()))
@pytest.mark.parametrize("mode", ["aggregate", "full"])
def test_diagnose_names_the_injected_pathology(
    scenario: str, expected: str, mode: str
) -> None:
    d = diagnose(traced(scenario, mode))
    assert d.pathology == expected, (
        f"{scenario} in {mode} mode was diagnosed as {d.pathology}: {d.summary}"
    )
    assert d.evidence
    assert d.summary


# -- question 1: queueing or compute? --------------------------------------


def test_explain_latency_blames_the_scheduler_under_starvation() -> None:
    trace = traced("prefill-starvation")
    d = diagnose(trace)
    b = explain_latency(trace, d.worst_request)
    assert b.share("decode_wait") + b.share("stalled") > 0.2
    assert "scheduler" in b.verdict
    assert sum(b.components.values()) <= b.total_ns * 1.01


def test_explain_latency_blames_queueing_under_admission_starvation() -> None:
    trace = traced("batch-starvation")
    slowest = max(
        (m for m in __import__("inferscope.metrics", fromlist=["per_request"])
         .per_request(trace).values() if m.completed),
        key=lambda m: m.queue_ns,
    )
    b = explain_latency(trace, slowest.request_id)
    assert b.dominant == "queued"
    assert "queued" in b.verdict


def test_explain_latency_blames_kv_when_preempted() -> None:
    from inferscope.metrics import per_request

    trace = traced("kv-thrash")
    preempted = [m for m in per_request(trace).values() if m.preemptions]
    assert preempted
    b = explain_latency(trace, preempted[0].request_id)
    assert "KV pressure" in b.verdict


def test_explain_latency_is_boring_on_a_healthy_run() -> None:
    trace = traced("healthy")
    d = diagnose(trace)
    b = explain_latency(trace, d.worst_request)
    assert b.dominant == "decoding"
    assert "computing" in b.verdict


def test_explain_latency_rejects_an_unknown_request() -> None:
    with pytest.raises(KeyError):
        explain_latency(traced("healthy"), "not-a-request")


# -- question 2: what was it batched with? ---------------------------------


def test_batch_context_finds_the_prefill_it_waited_behind() -> None:
    trace = traced("prefill-starvation")
    d = diagnose(trace)
    ctx = batch_context(trace, d.worst_request)
    assert ctx.iterations > 0
    assert ctx.co_residents, "a starved request still shares its batches"
    assert ctx.prefill_tokens_during > 1000, (
        "the long-prompt burst should be visible in its lifetime"
    )
    assert "prefill" in ctx.verdict


def test_batch_context_reports_undersized_batches() -> None:
    trace = traced("batch-starvation")
    d = diagnose(trace)
    ctx = batch_context(trace, d.worst_request)
    assert ctx.mean_batch_size < 2
    assert "alone" in ctx.verdict


# -- question 3: KV pressure -----------------------------------------------


def test_kv_pressure_detects_thrashing() -> None:
    kv = kv_pressure(traced("kv-thrash"))
    assert kv.preemptions > 0
    assert kv.evictions > 0
    assert kv.recomputed_tokens > 0
    assert kv.goodput_ratio < 0.95
    assert "too small" in kv.verdict


def test_kv_pressure_is_quiet_when_the_pool_is_ample() -> None:
    kv = kv_pressure(traced("healthy"))
    assert kv.preemptions == 0
    assert kv.goodput_ratio == 1.0
    assert "not the constraint" in kv.verdict


# -- question 4: TTFT or TPOT? ---------------------------------------------


@pytest.mark.parametrize("mode", ["aggregate", "full"])
def test_latency_regime_finds_the_decode_tail(mode: str) -> None:
    r = latency_regime(traced("prefill-starvation", mode))
    assert r.regime == "tpot-tail"
    assert r.tail_time_share > 0.10
    assert r.tpot_max_ns > 10e6


def test_latency_regime_calls_admission_starvation_ttft_bound() -> None:
    r = latency_regime(traced("batch-starvation"))
    assert r.regime == "ttft-bound"
    assert r.ttft_share > 0.5


def test_latency_regime_is_quiet_on_a_healthy_run() -> None:
    r = latency_regime(traced("healthy"))
    assert r.regime == "tpot-bound"
    assert r.tail_time_share < 0.05


@pytest.mark.parametrize("mode", ["aggregate", "full"])
def test_tail_share_is_stable_across_decode_modes(mode: str) -> None:
    """The statistic exists because p99 was not stable.

    Stalls are ~1% of decode steps, which puts a 99th percentile exactly on the
    boundary: the same scenario scored 39x and 1.3x on different runs. Their
    share of elapsed *time* is ~30% either way.
    """
    r = latency_regime(traced("prefill-starvation", mode))
    assert 0.20 < r.tail_time_share < 0.45


# -- reporting -------------------------------------------------------------


@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_every_result_formats_without_error(scenario: str) -> None:
    trace = traced(scenario)
    d = diagnose(trace)
    assert d.format()
    assert latency_regime(trace).format()
    assert kv_pressure(trace).format()
    assert explain_latency(trace, d.worst_request).format()
    assert batch_context(trace, d.worst_request).format()


def test_diagnose_rejects_an_empty_trace() -> None:
    with pytest.raises(ValueError):
        diagnose(Trace(events=[], names={}))
