"""Adapters that satisfy each target agent's one-method LLM protocol with a traced local model."""

from __future__ import annotations

from collections.abc import Sequence

from .ollama import OllamaChat


class StructuredOutputLLM:
    """structured-output-agent: complete(*, system, messages) -> str."""

    def __init__(self, chat: OllamaChat) -> None:
        self.chat = chat

    def complete(self, *, system: str, messages: Sequence[dict[str, str]]) -> str:
        return self.chat.chat(system, [dict(m) for m in messages], purpose="complete").text


class RagLLM:
    """rag-citation-agent: complete(system, user, max_tokens) -> str."""

    def __init__(self, chat: OllamaChat) -> None:
        self.chat = chat

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        return self.chat.chat(system, [{"role": "user", "content": user}], purpose="complete", max_tokens=max_tokens).text


class ReactLLM:
    """react-planning-agent: complete(*, purpose, system, prompt, max_tokens) -> str."""

    def __init__(self, chat: OllamaChat) -> None:
        self.chat = chat

    def complete(self, *, purpose: str, system: str, prompt: str, max_tokens: int) -> str:
        return self.chat.chat(system, [{"role": "user", "content": prompt}], purpose=purpose, max_tokens=max_tokens).text
