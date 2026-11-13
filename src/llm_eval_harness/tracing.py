"""Minimal span tracer: nested spans with timings and attributes, exported as JSONL."""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_span", default=None)


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    kind: str  # case | agent | llm | judge
    start: float  # seconds since the tracer was created
    duration: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id, "parent_id": self.parent_id, "name": self.name, "kind": self.kind,
            "start": round(self.start, 4), "duration": round(self.duration, 4), "attrs": self.attrs,
        }


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, kind: str, **attrs: Any) -> Iterator[Span]:
        sp = Span(uuid.uuid4().hex[:12], _current.get(), name, kind, time.perf_counter() - self._t0, attrs=dict(attrs))
        token = _current.set(sp.span_id)
        begin = time.perf_counter()
        try:
            yield sp
        except BaseException as exc:
            sp.attrs["error"] = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            sp.duration = time.perf_counter() - begin
            _current.reset(token)
            self.spans.append(sp)

    def children_of(self, span_id: str, kind: str | None = None) -> list[Span]:
        return [s for s in self.spans if s.parent_id == span_id and (kind is None or s.kind == kind)]

    def descendants_of(self, span_id: str, kind: str | None = None) -> list[Span]:
        out: list[Span] = []
        frontier = [span_id]
        while frontier:
            nxt = []
            for s in self.spans:
                if s.parent_id in frontier:
                    nxt.append(s.span_id)
                    if kind is None or s.kind == kind:
                        out.append(s)
            frontier = nxt
        return out

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for sp in sorted(self.spans, key=lambda s: s.start):
                fh.write(json.dumps(sp.to_dict()) + "\n")
