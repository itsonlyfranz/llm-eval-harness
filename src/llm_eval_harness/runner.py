"""Runs suites against targets, judges each output, and writes a run record plus a trace file."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Callable

from .judge import Judge
from .ollama import OllamaChat, OllamaError
from .stats import mean, percentile
from .suites import SUITES, Case
from .targets import TARGET_CLASSES
from .tracing import Tracer

# Reference API rates (USD per million tokens). The models run locally, so nothing is billed;
# these only convert measured token counts into an *estimate* of what the same traffic would cost.
# They are an assumption of this harness, not a quote from any provider: override with --price-in/--price-out.
DEFAULT_PRICE_IN = 1.0
DEFAULT_PRICE_OUT = 5.0


def estimate_cost(prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float) -> float:
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def summarize(cases: list[dict[str, Any]], price_in: float, price_out: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in sorted({c["target"] for c in cases}):
        rows = [c for c in cases if c["target"] == target]
        judged = [c["judge_score"] for c in rows if c["judge_score"] is not None]
        lat = [c["latency_s"] for c in rows]
        pin = sum(c["prompt_tokens"] for c in rows)
        pout = sum(c["completion_tokens"] for c in rows)
        out[target] = {
            "cases": len(rows),
            "check_pass_rate": mean([1.0 if c["check_score"] == 1.0 else 0.0 for c in rows]),
            "check_score_mean": mean([c["check_score"] for c in rows]),
            "judge_mean": mean(judged),
            "judge_unparsed": len(rows) - len(judged),
            "latency_mean_s": mean(lat), "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
            "llm_calls": sum(c["llm_calls"] for c in rows),
            "prompt_tokens": pin, "completion_tokens": pout,
            "est_cost_usd": estimate_cost(pin, pout, price_in, price_out),
            "judge_latency_total_s": sum(c["judge_latency_s"] for c in rows),
            "errors": sum(1 for c in rows if c["error"]),
        }
    return out


def run_suites(
    *,
    label: str,
    targets: list[str],
    agents_root: Path,
    agent_model: str,
    judge_model: str,
    host: str,
    out_dir: Path,
    num_predict: int | None = None,
    price_in: float = DEFAULT_PRICE_IN,
    price_out: float = DEFAULT_PRICE_OUT,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    tracer = Tracer()
    agent_chat = OllamaChat(agent_model, tracer, host=host, num_predict=num_predict, kind="llm")
    judge = Judge(OllamaChat(judge_model, tracer, host=host, kind="judge"))
    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{label}"
    cases_out: list[dict[str, Any]] = []

    for tname in targets:
        target = TARGET_CLASSES[tname](agents_root, agent_chat)
        suite: list[Case] = SUITES[tname][:limit] if limit else SUITES[tname]
        for case in suite:
            with tracer.span(case.case_id, "case", target=tname) as case_span:
                with tracer.span("agent", "agent", target=tname) as agent_span:
                    t0 = time.perf_counter()
                    try:
                        outcome = target.run(case)
                    except OllamaError:
                        raise  # infrastructure failure: do not record a fake score
                    except Exception as exc:  # agent crash is a real result for that case
                        from .targets import Outcome

                        outcome = Outcome(f"(agent crashed: {type(exc).__name__})", {"completed": False}, error=f"{type(exc).__name__}: {exc}"[:200])
                    latency = time.perf_counter() - t0
                llm_spans = tracer.descendants_of(agent_span.span_id, "llm")
                verdict = judge.score(task=case.task, rubric=case.rubric, reference=case.reference, output=outcome.output)
                judge_spans = tracer.descendants_of(case_span.span_id, "judge")
            row = {
                "target": tname, "case_id": case.case_id, "task": case.task, "reference": case.reference,
                "output": outcome.output, "checks": outcome.checks, "check_score": outcome.check_score,
                "judge_score": verdict.score, "judge_reason": verdict.reason, "detail": outcome.detail, "error": outcome.error,
                "latency_s": latency, "llm_calls": len(llm_spans),
                "prompt_tokens": sum(s.attrs.get("prompt_tokens", 0) for s in llm_spans),
                "completion_tokens": sum(s.attrs.get("completion_tokens", 0) for s in llm_spans),
                "judge_latency_s": sum(s.duration for s in judge_spans),
                "judge_prompt_tokens": sum(s.attrs.get("prompt_tokens", 0) for s in judge_spans),
                "judge_completion_tokens": sum(s.attrs.get("completion_tokens", 0) for s in judge_spans),
                "span_id": case_span.span_id,
            }
            cases_out.append(row)
            log(f"[{label}] {case.case_id:<28} checks={row['check_score']:.2f} judge={verdict.score} {latency:6.1f}s calls={row['llm_calls']}")

    record = {
        "run_id": run_id, "label": label, "started_at": started.isoformat(), "agent_model": agent_model,
        "judge_model": judge_model, "num_predict": num_predict, "price_in_per_mtok": price_in, "price_out_per_mtok": price_out,
        "wall_clock_s": sum(c["latency_s"] + c["judge_latency_s"] for c in cases_out),
        "cases": cases_out, "summary": summarize(cases_out, price_in, price_out),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "runs" / f"{run_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tracer.write_jsonl(out_dir / "traces" / f"{run_id}.jsonl")
    return record
