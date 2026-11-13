"""Command line: run | compare | dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dashboard import build
from .regress import compare, find_run, load_runs, render_text
from .runner import DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT, run_suites
from .suites import SUITES
from .targets import agents_root_from

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://127.0.0.1:11434"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llm-eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run suites, judge outputs, store a run record and trace")
    r.add_argument("--label", required=True, help="name for this run, e.g. baseline")
    r.add_argument("--targets", nargs="+", default=list(SUITES), choices=list(SUITES))
    r.add_argument("--agents-root", help="folder containing the agent repositories (or set AGENT_REPOS)")
    r.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model that powers the agents")
    r.add_argument("--judge-model", default=DEFAULT_MODEL)
    r.add_argument("--host", default=DEFAULT_HOST)
    r.add_argument("--num-predict", type=int, help="cap agent output tokens per call (used to inject a regression)")
    r.add_argument("--limit", type=int, help="only the first N cases per target")
    r.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN, help="assumed USD per million input tokens")
    r.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT, help="assumed USD per million output tokens")
    r.add_argument("--out", type=Path, default=Path("results"))

    c = sub.add_parser("compare", help="compare a run with a baseline; exit 1 on regression")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--out", type=Path, default=Path("results"))

    d = sub.add_parser("dashboard", help="write results/dashboard.html")
    d.add_argument("--out", type=Path, default=Path("results"))
    d.add_argument("--baseline", default="baseline")

    a = p.parse_args(argv)
    if a.cmd == "run":
        rec = run_suites(label=a.label, targets=a.targets, agents_root=agents_root_from(a.agents_root), agent_model=a.model,
                         judge_model=a.judge_model, host=a.host, out_dir=a.out, num_predict=a.num_predict,
                         price_in=a.price_in, price_out=a.price_out, limit=a.limit)
        print(f"wrote {a.out / 'runs' / (rec['run_id'] + '.json')}")
        return 0
    if a.cmd == "compare":
        runs = load_runs(a.out / "runs")
        report = compare(find_run(runs, a.baseline), find_run(runs, a.candidate))
        print(render_text(report))
        return 1 if report["regressed"] else 0
    html = build(a.out, a.baseline)
    path = a.out / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
