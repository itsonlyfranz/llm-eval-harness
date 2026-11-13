"""Runs each target agent on a case and applies deterministic checks. Target repos are imported read-only."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import RagLLM, ReactLLM, StructuredOutputLLM
from .ollama import OllamaChat
from .suites import Case

REPO_DIRS = {
    "structured-output": ("structured-output-agent", "src"),
    "rag-citation": ("rag-citation-agent", "src"),
    "react-planning": ("react-planning-agent", "src"),
}


@dataclass
class Outcome:
    output: str  # what the judge sees
    checks: dict[str, bool]
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def check_score(self) -> float:
        return sum(self.checks.values()) / len(self.checks) if self.checks else 0.0


def add_repo_to_path(agents_root: Path, target: str) -> Path:
    repo, sub = REPO_DIRS[target]
    path = agents_root / repo / sub
    if not path.is_dir():
        raise FileNotFoundError(f"{path} not found; point --agents-root (or AGENT_REPOS) at the folder holding {repo}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return agents_root / repo


def agents_root_from(arg: str | None) -> Path:
    root = arg or os.environ.get("AGENT_REPOS")
    if not root:
        raise SystemExit("set --agents-root or AGENT_REPOS to the folder that contains the agent repositories")
    return Path(root).expanduser()


class Target:
    name: str

    def __init__(self, agents_root: Path, chat: OllamaChat) -> None:
        self.repo = add_repo_to_path(agents_root, self.name)
        self.chat = chat

    def run(self, case: Case) -> Outcome:  # pragma: no cover - interface
        raise NotImplementedError


class StructuredOutputTarget(Target):
    name = "structured-output"

    def run(self, case: Case) -> Outcome:
        from structured_output_agent.demo import Invoice
        from structured_output_agent.errors import StructuredOutputError
        from structured_output_agent.runner import StructuredRunner

        exp = case.expect
        runner = StructuredRunner(StructuredOutputLLM(self.chat), max_retries=3)
        try:
            res = runner.run(exp["text"], Invoice, system="You extract invoice data from text. Currency must be an ISO 4217 code.")
        except StructuredOutputError as exc:  # RetriesExhausted derives from it
            return Outcome(f"(no valid output: {exc})", {"valid_schema": False, "vendor": False, "total": False, "currency": False},
                           {"attempts": 4}, error=str(exc)[:200])
        v = res.value
        checks = {
            "valid_schema": True,
            "vendor": v.vendor.strip().lower().rstrip(".") == exp["vendor"].lower().rstrip("."),
            "total": abs(v.total - exp["total"]) < 0.01,
            "currency": v.currency == exp["currency"],
        }
        return Outcome(v.model_dump_json(), checks, {"attempts": res.attempts})


class RagTarget(Target):
    name = "rag-citation"

    def __init__(self, agents_root: Path, chat: OllamaChat) -> None:
        super().__init__(agents_root, chat)
        from rag_citation_agent.config import Settings
        from rag_citation_agent.index import KnowledgeBase, load_documents
        from rag_citation_agent.search import SnapshotSearch

        self.settings = Settings(model=chat.model)
        self.kb = KnowledgeBase()
        self.kb.add(load_documents([self.repo / "data" / "knowledge_base"]))
        self.search = SnapshotSearch.from_file(self.repo / "data" / "web_snapshot" / "results.json")

    def run(self, case: Case) -> Outcome:
        from rag_citation_agent.agent import RAGAgent
        from rag_citation_agent.models import AnswerStatus

        exp = case.expect
        answer = RAGAgent(self.kb, RagLLM(self.chat), self.search, self.settings).ask(exp["question"])
        answered = answer.status == AnswerStatus.ANSWERED
        text = answer.text.lower()
        checks = {"abstention_correct": answered == exp["answerable"]}
        for needle in exp["must_contain"]:
            checks[f"mentions:{needle}"] = needle.lower() in text
        detail = {
            "status": answer.status.value, "confidence": answer.confidence.overall, "used_fallback": answer.used_fallback,
            "claims": len(answer.claims), "removed_claims": len(answer.removed_claims), "flags": answer.flags,
        }
        return Outcome(answer.text, checks, detail)


class ReactTarget(Target):
    name = "react-planning"

    def run(self, case: Case) -> Outcome:
        from react_agent.agent import ReActAgent
        from react_agent.models import AgentConfig

        exp = case.expect
        cfg = AgentConfig(max_iterations=6, deadline_seconds=300.0, max_tokens=512)
        res = ReActAgent(ReactLLM(self.chat), config=cfg).run(exp["task"])
        must = exp["must_contain"]
        if must is None:  # unanswerable: must not be a clean answer containing a guessed number
            checks = {"no_guess": res.degraded or not any(ch.isdigit() for ch in res.answer)}
        else:
            checks = {"correct_answer": must in res.answer.replace(",", ""), "not_degraded": not res.degraded}
        detail = {"stop_reason": res.stop_reason.value, "iterations": res.iterations, "revisions": res.revisions,
                  "tool_calls": sum(1 for s in res.steps if s.action), "warnings": res.warnings}
        return Outcome(res.answer, checks, detail)


TARGET_CLASSES = {c.name: c for c in (StructuredOutputTarget, RagTarget, ReactTarget)}
