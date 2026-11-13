from llm_eval_harness.judge import parse_verdict
from llm_eval_harness.regress import compare
from llm_eval_harness.runner import estimate_cost, summarize
from llm_eval_harness.stats import mean, percentile
from llm_eval_harness.tracing import Tracer


def test_parse_verdict_accepts_wrapped_json_and_rejects_out_of_range():
    assert parse_verdict('noise {"score": 4, "reason": "ok"} tail').score == 4
    assert parse_verdict('{"score": 9, "reason": "x"}').score is None
    assert parse_verdict("not json").score is None


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 50) is None and mean([]) is None


def test_tracer_nests_spans_and_records_errors():
    t = Tracer()
    with t.span("case", "case") as case:
        with t.span("call", "llm", prompt_tokens=3):
            pass
        try:
            with t.span("bad", "llm"):
                raise ValueError("boom")
        except ValueError:
            pass
    assert [s.name for s in t.children_of(case.span_id)] == ["call", "bad"]
    assert "ValueError" in t.spans[1].attrs["error"]
    assert len(t.descendants_of(case.span_id, "llm")) == 2


def test_estimate_cost():
    assert estimate_cost(1_000_000, 1_000_000, 1.0, 5.0) == 6.0


def _case(cid, check, judge, lat=1.0):
    return {"target": "t", "case_id": cid, "check_score": check, "judge_score": judge, "latency_s": lat, "llm_calls": 1,
            "prompt_tokens": 10, "completion_tokens": 5, "judge_latency_s": 0.1, "error": None}


def _run(label, cases):
    return {"label": label, "cases": cases, "summary": summarize(cases, 1.0, 5.0)}


def test_compare_flags_case_and_target_regression():
    base = _run("base", [_case("a", 1.0, 5), _case("b", 1.0, 4)])
    worse = _run("worse", [_case("a", 0.5, 5), _case("b", 1.0, 1, lat=3.0)])
    rep = compare(base, worse)
    assert rep["regressed"] and {c["case_id"] for c in rep["regressed_cases"]} == {"a", "b"}
    assert rep["targets"]["t"]["regressed"]


def test_compare_identical_runs_is_clean():
    base = _run("base", [_case("a", 1.0, 5)])
    assert not compare(base, _run("again", [_case("a", 1.0, 5)]))["regressed"]
