"""inferscope -- OpenTelemetry for LLM inference internals."""

from inferscope.events import Event, EventKind
from inferscope.sinks import MemorySink, NullSink, Sink, SQLiteSink
from inferscope.tracer import RequestSpan, Tracer

__version__ = "0.1.0"
__all__ = [
    "Tracer",
    "RequestSpan",
    "EventKind",
    "Event",
    "Sink",
    "MemorySink",
    "NullSink",
    "SQLiteSink",
]
