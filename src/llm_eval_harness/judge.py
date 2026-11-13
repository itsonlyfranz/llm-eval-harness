"""LLM-as-judge: scores an agent output 1-5 against a rubric using the local judge model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .ollama import OllamaChat

JUDGE_SYSTEM = """You are a strict evaluator of an AI agent's output.
Score the output from 1 to 5 against the rubric:
5 = fully correct and complete; 4 = correct with a minor omission; 3 = partly correct;
2 = mostly wrong or unsupported; 1 = wrong, empty, or fabricated.
Judge only against the reference and the rubric. Do not reward length or confidence.
Reply with one JSON object: {"score": <integer 1-5>, "reason": "<one short sentence>"}"""


@dataclass
class Verdict:
    score: int | None  # None = judge reply could not be parsed
    reason: str
    raw: str


def parse_verdict(raw: str) -> Verdict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data["score"])
            if 1 <= score <= 5:
                return Verdict(score, str(data.get("reason", ""))[:300], raw)
        except (ValueError, KeyError, TypeError):
            pass
    return Verdict(None, "unparseable judge reply", raw)


class Judge:
    def __init__(self, chat: OllamaChat) -> None:
        self.chat = chat

    def score(self, *, task: str, rubric: str, reference: str, output: str) -> Verdict:
        prompt = (
            f"Task given to the agent:\n{task}\n\nRubric:\n{rubric}\n\n"
            f"Reference (ground truth):\n{reference}\n\nAgent output:\n{output}\n"
        )
        comp = self.chat.chat(JUDGE_SYSTEM, [{"role": "user", "content": prompt}], purpose="judge", max_tokens=200, json_mode=True)
        return parse_verdict(comp.text)
