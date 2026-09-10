"""Event sinks. See ``inferscope.sinks.base.Sink`` for the interface."""

from inferscope.sinks.base import Sink
from inferscope.sinks.memory import MemorySink
from inferscope.sinks.null import NullSink
from inferscope.sinks.sqlite import SQLiteSink

__all__ = ["Sink", "MemorySink", "NullSink", "SQLiteSink"]
