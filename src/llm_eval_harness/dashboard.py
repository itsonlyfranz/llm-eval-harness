"""Static HTML dashboard (no JavaScript, no external assets) built from run records and traces."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .regress import compare, find_run, load_runs


def _f(x: float | None, nd: int = 2, suffix: str = "") -> str:
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"


def _bar(value: float | None, maxv: float, cls: str = "") -> str:
    if value is None or maxv <= 0:
        return ""
    return f'<span class="bar {cls}" style="width:{max(1.0, 100 * value / maxv):.1f}%"></span>'


CSS = """
:root{--bg:#fbfaf8;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--line:#e3e1dc;--acc:#2f6f5e;--acc2:#b4601f;--bad:#b3261e;--ok:#2f6f5e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--fg:#ececea;--mut:#9a9a9f;--card:#1d1d1b;--line:#33332f;--acc:#6fbfa8;--acc2:#e0955a;--bad:#f2867f;--ok:#6fbfa8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
p.sub,.note{color:var(--mut);font-size:13px;margin:4px 0}section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}.cell{position:relative;min-width:120px}.bar{display:block;height:6px;border-radius:3px;background:var(--acc);margin-top:3px}.bar.l{background:var(--acc2)}
.bad{color:var(--bad);font-weight:600}.ok{color:var(--ok)}.wf{position:relative;height:18px;margin:2px 0;background:transparent}.wf span{position:absolute;height:14px;border-radius:2px;top:2px;min-width:2px}
.wf .case{background:var(--line)}.wf .agent{background:var(--mut);opacity:.5}.wf .llm{background:var(--acc)}.wf .judge{background:var(--acc2)}
.lbl{display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;font-size:12px}.lbl div:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
"""


def build(out_dir: Path, baseline_label: str = "baseline") -> str:
    runs = load_runs(out_dir / "runs")
    if not runs:
        raise SystemExit(f"no runs found in {out_dir / 'runs'}")
    latest = runs[-1]
    targets = sorted({t for r in runs for t in r["summary"]})
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>LLM Eval Dashboard</title><style>{CSS}</style></head><body><main>"]
    parts.append("<h1>LLM eval dashboard</h1>")
    parts.append(f"<p class=sub>{len(runs)} runs · agent model <b>{escape(latest['agent_model'])}</b> · judge model <b>{escape(latest['judge_model'])}</b> "
                 "(same local model, so judge scores carry self-preference bias) · all numbers measured on a local Ollama server</p>")

    # Run comparison
    parts.append("<h2>Runs</h2><section><table><tr><th>run</th><th>target</th><th>cases</th><th>checks passed</th><th>judge mean (1-5)</th>"
                 "<th>p50 s</th><th>p95 s</th><th>tokens in/out</th><th>est. cost USD</th></tr>")
    max_judge = 5.0
    for r in runs:
        for t in targets:
            s = r["summary"].get(t)
            if not s:
                continue
            parts.append(f"<tr><td>{escape(r['label'])}</td><td>{t}</td><td class=n>{s['cases']}</td>"
                         f"<td class=n>{_f(s['check_pass_rate'], 0) if s['check_pass_rate'] in (0, 1) else _f(100 * s['check_pass_rate'], 0, '%')}</td>"
                         f"<td class=cell>{_f(s['judge_mean'])}{_bar(s['judge_mean'], max_judge)}</td>"
                         f"<td class=n>{_f(s['latency_p50_s'], 1)}</td><td class=n>{_f(s['latency_p95_s'], 1)}</td>"
                         f"<td class=n>{s['prompt_tokens']:,}/{s['completion_tokens']:,}</td><td class=n>${s['est_cost_usd']:.4f}</td></tr>")
    parts.append("</table>")
    parts.append(f"<p class=note>Estimated cost converts measured tokens at an assumed reference rate of ${latest['price_in_per_mtok']}/M input and "
                 f"${latest['price_out_per_mtok']}/M output tokens. Nothing is billed: the models run locally.</p></section>")

    # Latency / cost bars for the latest run
    parts.append(f"<h2>Latency and cost, run “{escape(latest['label'])}”</h2><section><table><tr><th>target</th><th>p50 latency</th><th>p95 latency</th><th>LLM calls</th><th>est. cost</th></tr>")
    maxp95 = max((latest["summary"][t]["latency_p95_s"] or 0) for t in latest["summary"])
    maxcost = max(latest["summary"][t]["est_cost_usd"] for t in latest["summary"])
    for t, s in latest["summary"].items():
        parts.append(f"<tr><td>{t}</td><td class=cell>{_f(s['latency_p50_s'], 1, ' s')}{_bar(s['latency_p50_s'], maxp95)}</td>"
                     f"<td class=cell>{_f(s['latency_p95_s'], 1, ' s')}{_bar(s['latency_p95_s'], maxp95)}</td>"
                     f"<td class=n>{s['llm_calls']}</td><td class=cell>${s['est_cost_usd']:.4f}{_bar(s['est_cost_usd'], maxcost, 'l')}</td></tr>")
    parts.append("</table></section>")

    # Regression tracking
    base = next((r for r in runs if r["label"] == baseline_label), None)
    parts.append("<h2>Regression tracking</h2><section>")
    if base is None:
        parts.append(f"<p class=note>No run labelled “{escape(baseline_label)}”.</p>")
    else:
        parts.append(f"<p class=note>Each run compared with “{escape(baseline_label)}”. Case regression: check score drops, or judge score drops by 2 or more. "
                     "Target regression: mean check score down 0.10, mean judge down 0.5, or median latency up 50%.</p>"
                     "<table><tr><th>run</th><th>target</th><th>Δ check</th><th>Δ judge</th><th>Δ p50 latency</th><th>verdict</th></tr>")
        for r in runs:
            if r is base:
                continue
            rep = compare(base, r)
            for t, x in rep["targets"].items():
                verdict = f"<span class=bad>regression: {escape('; '.join(x['flags']))}</span>" if x["regressed"] else "<span class=ok>no regression</span>"
                lat = "n/a" if x["latency_p50_change"] is None else f"{x['latency_p50_change']:+.0%}"
                dj = "n/a" if x["d_judge"] is None else f"{x['d_judge']:+.2f}"
                parts.append(f"<tr><td>{escape(r['label'])}</td><td>{t}</td><td class=n>{x['d_check']:+.2f}</td><td class=n>{dj}</td><td class=n>{lat}</td><td>{verdict}</td></tr>")
        parts.append("</table>")
    parts.append("</section>")

    # Trace waterfall for slowest case of the latest run
    slow = max(latest["cases"], key=lambda c: c["latency_s"])
    trace_path = out_dir / "traces" / f"{latest['run_id']}.jsonl"
    parts.append(f"<h2>Trace: slowest case, {escape(slow['case_id'])} ({slow['latency_s']:.1f} s)</h2><section>")
    if trace_path.is_file():
        spans = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        by_id = {s["span_id"]: s for s in spans}
        case_span = by_id[slow["span_id"]]
        kids = [s for s in spans if s["start"] >= case_span["start"] - 1e-6 and s["start"] <= case_span["start"] + case_span["duration"] + 1e-6]
        total = case_span["duration"] or 1
        for s in kids:
            left = 100 * (s["start"] - case_span["start"]) / total
            width = 100 * s["duration"] / total
            tok = f" · {s['attrs'].get('prompt_tokens', 0)}→{s['attrs'].get('completion_tokens', 0)} tok" if s["kind"] in ("llm", "judge") else ""
            parts.append(f"<div class=lbl><div>{escape(s['kind'])}: {escape(s['name'])}</div><div class=wf title='{s['duration']:.2f}s{tok}'>"
                         f"<span class={s['kind']} style='left:{left:.2f}%;width:{width:.2f}%'></span></div></div>")
        parts.append("<p class=note>Bars are spans on one time axis: grey = case, dark grey = agent run, green = LLM calls, orange = judge call.</p>")
    else:
        parts.append("<p class=note>Trace file not found.</p>")
    parts.append("</section></main></body></html>")
    return "".join(parts)
