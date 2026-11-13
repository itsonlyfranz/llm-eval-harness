# llm-eval-harness

Evaluation and observability harness for LLM agents. It runs three existing agent projects against a local Ollama model, scores them with deterministic checks and a local LLM judge, records a trace of every model call, tracks regressions between runs, and builds a static cost/latency dashboard.

Every number below comes from the run records in `results/runs/` and was produced on one machine (Apple silicon, 32 GB, Ollama 0.40.14). Test data is synthetic or taken from the agents' own bundled sample data.

## What is measured

| Target agent | Suite | Deterministic check |
|---|---|---|
| structured-output-agent | 8 invoice-extraction cases (symbols, subtotals vs totals, thousands separators) | schema valid, vendor / total / currency match |
| rag-citation-agent | 10 questions over a sample knowledge base (7 from the knowledge base, 1 needing the search fallback, 2 unanswerable) | abstains exactly when expected, expected facts present |
| react-planning-agent | 8 calculator / lookup tasks, one unanswerable | expected value in the final answer, run not degraded |

The agent repositories are imported read-only; nothing in them is modified. Each agent's one-method LLM interface is satisfied by an adapter (`adapters.py`) that calls Ollama at temperature 0 with a fixed seed and records a span per call (`tracing.py`).

Judge: `qwen3:8b` (Alibaba, 5.2 GB, Q4_K_M) served by a local Ollama, models stored in `~/models`. It scores each output 1-5 against a reference answer and rubric. The same model also powers the agents, so the judge is not independent of the thing it grades.

Cost: the models run locally, so nothing is billed. "Estimated cost" converts measured token counts at an assumed reference rate of $1.00 per million input tokens and $5.00 per million output tokens. That rate is an assumption of this harness, not a provider quote; change it with `--price-in` / `--price-out`.

## Results

Run `baseline`, then `baseline-repeat` (identical settings, to measure noise), then `truncated` (agent output capped at 40 tokens per call, a deliberately injected regression to check that the detector fires).

| Run | Target | Checks fully passed | Judge mean (1-5) | p50 / p95 latency (s) | LLM calls | Tokens in / out | Est. cost (USD) |
|---|---|---|---|---|---|---|---|
| baseline | structured-output | 8/8 | 5.00 | 1.10 / 1.25 | 8 | 1,419 / 209 | 0.0025 |
| baseline | rag-citation | 10/10 | 5.00 | 3.55 / 9.57 | 10 | 5,518 / 864 | 0.0098 |
| baseline | react-planning | 7/8 | 4.62 | 15.0 / 21.1 | 54 | 14,397 / 2,620 | 0.0275 |
| baseline-repeat | structured-output | 8/8 | 5.00 | 1.09 / 1.59 | 8 | 1,419 / 209 | 0.0025 |
| baseline-repeat | rag-citation | 10/10 | 5.00 | 3.70 / 9.45 | 10 | 5,518 / 864 | 0.0098 |
| baseline-repeat | react-planning | 7/8 | 4.62 | 13.2 / 18.8 | 54 | 14,397 / 2,620 | 0.0275 |
| truncated | structured-output | 8/8 | 5.00 | 1.09 / 1.31 | 8 | 1,419 / 209 | 0.0025 |
| truncated | rag-citation | 2/10 | 3.30 | 7.77 / 8.04 | 30 | 15,764 / 1,144 | 0.0215 |
| truncated | react-planning | 1/8 | 1.62 | 6.23 / 12.0 | 38 | 7,369 / 1,421 | 0.0145 |

Costs are the `est_cost_usd` values from the run files, shown to 4 decimals. Full per-run wall clock (agent plus judge): 208 s, 197 s, 165 s.

### Findings

- **Reproducibility.** `baseline` and `baseline-repeat` give identical scores and token counts on all 26 cases; only latency moves (median changes of +4%, -12%, -1%). No regression is reported between them.
- **Regression detection works on an injected fault.** Comparing `truncated` with `baseline` flags rag-citation (mean check 1.00 to 0.20, judge 5.00 to 3.30) and react-planning (check 0.88 to 0.12, judge 4.62 to 1.62), with 14 regressed cases. structured-output is unaffected, since its replies are short JSON.
- **Truncation can raise cost and lower latency.** For rag-citation it tripled LLM calls (10 to 30) through retries, raising median latency 119% and estimated cost about 2x. For react-planning it made the agent give up early, so latency fell 58% while quality collapsed. Latency and cost alone would have shown "faster" there, which is why quality metrics are tracked alongside them.
- **One real baseline failure.** `re-04-lookup-add` fails in both baseline runs: the agent stops with `tool_failures` after three lookups and reports it cannot compute the combined population. Judge score 2, check 0.
- **The judge is not reliable on its own.** Deterministic checks and the judge agree (check fully passed vs judge >= 4) on 26/26 cases in both baseline runs but only 23/26 in `truncated`. In those 3 cases (`rag-03-refund`, `rag-04-sso`, `rag-08-regions`) the agent wrongly abstained although the reference contains the answer, and the judge scored 5, saying the abstention was correct. Treat judge scores as a secondary signal; the deterministic checks caught these.
- **Small suites.** 8 to 10 cases per target and a single model mean small differences are not statistically meaningful. The perfect baseline scores show the suites are easy for this model; they are useful for catching large regressions, not for ranking models.

## Reproduce

```bash
# 1. Local judge/agent model (no sudo; models go to ~/models)
OLLAMA_MODELS=~/models ollama serve &
OLLAMA_MODELS=~/models ollama pull qwen3:8b

# 2. Install the harness and the agents' dependencies
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 3. Point at the folder holding structured-output-agent, rag-citation-agent, react-planning-agent
export AGENT_REPOS=/path/to/agent/repos

# 4. Everything reported above: three runs, two comparisons, the dashboard
scripts/run_all.sh

# Individual commands
.venv/bin/llm-eval run --label baseline
.venv/bin/llm-eval run --label candidate --num-predict 40
.venv/bin/llm-eval compare baseline candidate   # exit code 1 on regression
.venv/bin/llm-eval dashboard                    # writes results/dashboard.html
.venv/bin/python -m pytest
```

## Outputs

- `results/runs/<run>.json`: per-case output, checks, judge score and reason, latency, tokens; per-target summary.
- `results/traces/<run>.jsonl`: one span per case, agent run, model call and judge call, with timings and token counts.
- `results/dashboard.html`: self-contained page (no JavaScript, no external assets) with run comparison, latency and cost bars, regression verdicts and a trace waterfall for the slowest case.

## Regression rules

A case regresses when its check score drops or its judge score drops by 2 or more. A target regresses when mean check score falls by 0.10, mean judge score by 0.5, or median latency rises by 50%. Thresholds live in `regress.py`.
