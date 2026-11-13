"""Regression tracking: compare a run against a baseline, case by case and per target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# A case regresses when its deterministic check score drops, or its judge score drops by at least this much.
JUDGE_DROP = 2
# A target regresses when a mean metric moves past these thresholds.
TARGET_CHECK_DROP = 0.10  # absolute drop in mean check score
TARGET_JUDGE_DROP = 0.5  # absolute drop in mean judge score
TARGET_LATENCY_RISE = 0.5  # relative rise in median latency


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))]
    return sorted(runs, key=lambda r: r["started_at"])


def find_run(runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    hits = [r for r in runs if r["label"] == key or r["run_id"] == key]
    if not hits:
        raise KeyError(f"no run with label or id {key!r}; have {[r['label'] for r in runs]}")
    return hits[-1]


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = {(c["target"], c["case_id"]): c for c in baseline["cases"]}
    cases, regressed, improved = [], [], []
    for c in candidate["cases"]:
        b = base.get((c["target"], c["case_id"]))
        if b is None:
            continue
        d_check = c["check_score"] - b["check_score"]
        d_judge = None if None in (c["judge_score"], b["judge_score"]) else c["judge_score"] - b["judge_score"]
        reasons = []
        if d_check < 0:
            reasons.append(f"check score {b['check_score']:.2f} -> {c['check_score']:.2f}")
        if d_judge is not None and d_judge <= -JUDGE_DROP:
            reasons.append(f"judge {b['judge_score']} -> {c['judge_score']}")
        row = {"target": c["target"], "case_id": c["case_id"], "d_check": d_check, "d_judge": d_judge, "reasons": reasons}
        cases.append(row)
        if reasons:
            regressed.append(row)
        elif d_check > 0 or (d_judge is not None and d_judge >= JUDGE_DROP):
            improved.append(row)

    targets = {}
    for t, cs in candidate["summary"].items():
        bs = baseline["summary"].get(t)
        if not bs:
            continue
        flags = []
        d_check = cs["check_score_mean"] - bs["check_score_mean"]
        if d_check <= -TARGET_CHECK_DROP:
            flags.append(f"mean check score {bs['check_score_mean']:.2f} -> {cs['check_score_mean']:.2f}")
        d_judge = None
        if cs["judge_mean"] is not None and bs["judge_mean"] is not None:
            d_judge = cs["judge_mean"] - bs["judge_mean"]
            if d_judge <= -TARGET_JUDGE_DROP:
                flags.append(f"mean judge {bs['judge_mean']:.2f} -> {cs['judge_mean']:.2f}")
        rise = (cs["latency_p50_s"] / bs["latency_p50_s"] - 1) if bs["latency_p50_s"] else None
        if rise is not None and rise >= TARGET_LATENCY_RISE:
            flags.append(f"median latency {bs['latency_p50_s']:.1f}s -> {cs['latency_p50_s']:.1f}s")
        targets[t] = {"d_check": d_check, "d_judge": d_judge, "latency_p50_change": rise, "regressed": bool(flags), "flags": flags}

    return {
        "baseline": baseline["label"], "candidate": candidate["label"], "cases_compared": len(cases),
        "regressed_cases": regressed, "improved_cases": improved, "targets": targets,
        "regressed": any(t["regressed"] for t in targets.values()) or bool(regressed),
    }


def render_text(report: dict[str, Any]) -> str:
    lines = [f"{report['candidate']} vs baseline {report['baseline']}: {report['cases_compared']} cases compared"]
    for t, r in report["targets"].items():
        dj = "n/a" if r["d_judge"] is None else f"{r['d_judge']:+.2f}"
        lat = "n/a" if r["latency_p50_change"] is None else f"{r['latency_p50_change']:+.0%}"
        status = "REGRESSION: " + "; ".join(r["flags"]) if r["regressed"] else "ok"
        lines.append(f"  {t:<18} d_check={r['d_check']:+.2f} d_judge={dj} p50_latency={lat}  {status}")
    lines.append(f"  regressed cases: {len(report['regressed_cases'])}, improved cases: {len(report['improved_cases'])}")
    for c in report["regressed_cases"]:
        lines.append(f"    - {c['case_id']}: {'; '.join(c['reasons'])}")
    return "\n".join(lines)
