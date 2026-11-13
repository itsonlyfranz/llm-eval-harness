"""Thin Ollama HTTP client that records every call as a trace span."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .tracing import Tracer

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class OllamaChat:
    """Deterministic (temperature 0, fixed seed) chat completions against a local server."""

    def __init__(
        self,
        model: str,
        tracer: Tracer,
        host: str = DEFAULT_HOST,
        num_predict: int | None = None,
        timeout: float = 600.0,
        kind: str = "llm",
    ) -> None:
        self.model, self.tracer, self.host = model, tracer, host.rstrip("/")
        self.num_predict, self.timeout, self.kind = num_predict, timeout, kind

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        purpose: str = "chat",
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        options: dict[str, Any] = {"temperature": 0, "seed": 0}
        limits = [n for n in (self.num_predict, max_tokens) if n]
        if limits:
            options["num_predict"] = min(limits)
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "options": options,
        }
        if json_mode:
            body["format"] = "json"
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with self.tracer.span(purpose, self.kind, model=self.model) as sp:
            begin = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise OllamaError(f"ollama request failed: {exc}") from exc
            latency = time.perf_counter() - begin
            comp = Completion(
                text=data.get("message", {}).get("content", ""),
                prompt_tokens=int(data.get("prompt_eval_count", 0)),
                completion_tokens=int(data.get("eval_count", 0)),
                latency_s=latency,
            )
            sp.attrs.update(
                prompt_tokens=comp.prompt_tokens, completion_tokens=comp.completion_tokens,
                done_reason=data.get("done_reason"), load_s=round(data.get("load_duration", 0) / 1e9, 3),
            )
        return comp
