"""Unit tests for the deterministic analytics, router and grounding check."""
import math

import pytest

from atlas import grounding
from atlas.analytics import TOOLS, run_tool
from atlas.analytics import enterprise as E
from atlas.analytics import university as U
from atlas.analytics.common import money, score_options
from atlas.router import route

ALL = [(p, t) for p in TOOLS for t in TOOLS[p] if TOOLS[p][t][0] is not None]  # developer tools run via atlas/dev.py


@pytest.mark.parametrize("persona,tool", ALL)
def test_contract(persona, tool):
    r = run_tool(persona, tool)
    for k in ("values", "method", "sources", "chart", "headline", "facts", "template"):
        assert k in r, k
    if r["chart"]["type"] != "none":
        assert r["chart"]["labels"] and all(len(s["data"]) == len(r["chart"]["labels"]) for s in r["chart"]["series"])
    # the deterministic template must itself be fully grounded
    assert grounding.check(r["template"], r)["ok"], grounding.check(r["template"], r)["unmatched"]


@pytest.mark.parametrize("persona,tool", ALL)
def test_deterministic(persona, tool):
    a, b = run_tool(persona, tool), run_tool(persona, tool)
    assert a["headline"]["value"] == b["headline"]["value"]


def test_conversion_rates_monotonic():
    c = E.conversion_rates()["All"]
    assert 0 < c["Prospect"] < c["Qualify"] < c["Propose"] < c["Commit"] < 1


def test_forecast_percentiles_ordered():
    v = E.forecast_quarter()["values"]
    assert v["revenue_p10"] <= v["revenue_p50"] <= v["revenue_p90"]
    assert v["profit_p10"] <= v["profit_p50"] <= v["profit_p90"]


def test_what_if_lower_win_rate_lowers_profit():
    v = E.what_if("Commit", -10)["values"]
    assert v["scenario_profit"] < v["base_profit"]
    assert math.isclose(v["profit_change"], v["revenue_change"] * E.margin_model()["gross_margin"], rel_tol=1e-6)


def test_region_gap_math():
    for r in E.region_vs_plan()["values"]["regions"]:
        assert math.isclose(r["gap"], r["projected"] - r["plan"])


def test_hiring_nonnegative():
    v = E.hiring_plan()["values"]
    assert v["hires"] >= 0 and v["hires_if_q2"] >= v["hires"]


def test_risk_probabilities_bounded():
    for d in E.deal_risk(5)["values"]["deals"]:
        assert 0 < d["risk"] < 1


def test_scoring_bounds_and_sensitivity():
    rows = [{"a": 1, "b": 10}, {"a": 2, "b": 5}, {"a": 3, "b": 1}]
    scores, contrib, norm, wins = score_options(rows, [("a", "A", 50, True, "num"), ("b", "B", 50, False, "num")], salt=1)
    assert all(0 <= s <= 100 for s in scores)
    assert math.isclose(wins.sum(), 1.0)
    assert scores.argmax() == 2


def test_sponsor_shares_sum_to_one():
    v = U.sponsor_concentration()["values"]
    assert math.isclose(sum(v["shares"].values()), 1.0, rel_tol=1e-9)


def test_money_format():
    assert money(644_707_194.98) == "$644.7M"
    assert money(2_480_000_000) == "$2.48B"
    assert money(-4_259_486) == "-$4.3M"


def test_grounding_rejects_invented_numbers():
    r = E.forecast_quarter()
    assert grounding.check("Revenue is $644.7M with a 13.3% margin.", r)["ok"]
    assert not grounding.check("Revenue is $700.0M.", r)["ok"]
    assert not grounding.check("Margin is 21.5%.", r)["ok"]
    assert grounding.check("Q4 2026 looks solid.", r)["ok"]  # years / quarter labels are not claims


GOLDEN = [
    ("enterprise", "What's our projected Q4 revenue and profit?", "forecast_quarter"),
    ("enterprise", "Which five deals are most at risk this quarter, and why?", "deal_risk"),
    ("enterprise", "Should we build a new sales center in Phoenix, Denver, or Albuquerque?", "score_sites"),
    ("enterprise", "What happens to Q4 profit if our Commit stage win rate drops ten points?", "what_if"),
    ("enterprise", "Which region is furthest behind plan, and by how much?", "region_vs_plan"),
    ("enterprise", "How many reps do we need to hire to hit next year's target?", "hiring_plan"),
    ("enterprise", "How does our pipeline coverage compare to this time last year?", "pipeline_coverage"),
    ("enterprise", "Give me a thirty second board summary.", "board_summary"),
    ("university", "Based on our grant pipeline, what's projected research funding for FY27?", "grant_forecast"),
    ("university", "Which sponsors are we most dependent on, and what's the risk?", "sponsor_concentration"),
    ("university", "Where should we open our next research center?", "score_research_sites"),
    ("university", "What's our projected fall enrollment yield?", "enrollment_forecast"),
]


@pytest.mark.parametrize("persona,q,tool", GOLDEN)
def test_fallback_router_golden(persona, q, tool):
    assert route(q, persona)[0] == tool


def test_router_args():
    assert route("What if Commit win rate drops ten points?", "enterprise")[1] == {"stage": "Commit", "delta_pts": -10.0}
    assert route("Phoenix or Denver for a sales center?", "enterprise")[1] == {"cities": ["Phoenix", "Denver"]}


def test_off_topic_routes_to_capabilities():
    assert route("What's the weather like in Paris?", "enterprise")[0] == "capabilities"


def test_follow_up_detection():
    from atlas.brain import is_follow_up
    assert is_follow_up("Why?") and is_follow_up("What about the West?") and is_follow_up("Tell me more")
    assert not is_follow_up("Which region is furthest behind plan, and by how much?")
