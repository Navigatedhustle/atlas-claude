"""Alta Vista University (university persona) analytics. Pure Python + numpy, deterministic."""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np

from atlas.analytics.common import (TODAY, item, money, num, p10_50_90, pct, q, q1, result, rng,
                                    score_options)
from atlas.config import MC_SIMS

UNI_STAGES = ["Drafting", "Submitted", "Under Review", "Awarded pending"]
SPONSORS = ["NIH", "NSF", "DOE", "DoD", "Industry", "Foundations"]
# Year-over-year volatility of each sponsor's award volume (from the sponsor history table below)
FY_NOW = TODAY.year + (1 if TODAY.month >= 7 else 0)   # FY runs July..June; Oct 2026 is FY27
FY_START = dt.date(FY_NOW - 1, 7, 1)
FY_END = dt.date(FY_NOW, 6, 30)


def fy_label(fy: int) -> str:
    return f"FY{fy % 100:02d}"


@lru_cache(maxsize=1)
def stage_success() -> dict:
    """P(awarded | proposal reached stage), by sponsor and overall, from 3 years of proposals."""
    rows = q("SELECT sponsor, max_stage, outcome, COUNT(*) n FROM proposal_history GROUP BY 1,2,3")
    out = {}
    for sp in SPONSORS + ["All"]:
        reached = np.zeros(4)
        won = np.zeros(4)
        for r in rows:
            if sp != "All" and r["sponsor"] != sp:
                continue
            i = UNI_STAGES.index(r["max_stage"])
            reached[: i + 1] += r["n"]
            if r["outcome"] == "awarded":
                won[: i + 1] += r["n"]
        out[sp] = {s: float(won[i] / max(reached[i], 1)) for i, s in enumerate(UNI_STAGES)}
        out[sp]["_n"] = int(reached[0])
    return out


def active_funding(fy: int, by: str | None = None) -> dict | float:
    """Annual funding in a fiscal year from awards active that year (already awarded)."""
    rows = q("SELECT sponsor, college, annual_amount FROM grant_awards WHERE start_fy <= ? AND start_fy + duration_years > ?",
             fy, fy)
    if by is None:
        return float(sum(r["annual_amount"] for r in rows))
    out: dict = {}
    for r in rows:
        out[r[by]] = out.get(r[by], 0.0) + r["annual_amount"]
    return out


@lru_cache(maxsize=1)
def _simulate():
    succ = stage_success()
    pipe = q("SELECT * FROM grant_pipeline WHERE start_fy = ?", FY_NOW)
    g = rng(11)
    p = np.array([succ[r["sponsor"]][r["stage"]] for r in pipe])
    # Funding counts only for the part of FY27 after the award starts
    frac = np.array([max(0.0, (FY_END - dt.date.fromisoformat(r["expected_start"])).days / 365.0) for r in pipe])
    amt = np.array([r["annual_amount"] for r in pipe]) * frac
    wins = g.random((MC_SIMS, len(pipe))) < p
    new = wins * amt
    continuing = active_funding(FY_NOW, "sponsor")
    by_sponsor = {}
    for sp in SPONSORS:
        m = np.array([r["sponsor"] == sp for r in pipe])
        by_sponsor[sp] = continuing.get(sp, 0.0) + new[:, m].sum(axis=1)
    total = sum(by_sponsor.values())
    return total, by_sponsor, pipe, p, amt


def grant_forecast() -> dict:
    total, by_sponsor, pipe, p, amt = _simulate()
    pr = p10_50_90(total)
    cont = active_funding(FY_NOW)
    expected_new = float((p * amt).sum())
    hist = {fy: active_funding(fy) for fy in range(FY_NOW - 4, FY_NOW)}
    prev = hist[FY_NOW - 1]
    growth = pr["P50"] / prev - 1
    succ = stage_success()["All"]
    lbl = fy_label(FY_NOW)
    by_college = active_funding(FY_NOW, "college")
    for r, pp, a in zip(pipe, p, amt):
        by_college[r["college"]] = by_college.get(r["college"], 0.0) + float(pp * a)
    colleges = sorted(by_college.items(), key=lambda kv: -kv[1])
    facts = [f"Projected {lbl} research funding is {money(pr['P50'])} at the median.",
             f"The 80% range is {money(pr['P10'])} to {money(pr['P90'])}.",
             f"{money(cont)} is already secured from continuing awards; the open pipeline adds an expected {money(expected_new)}.",
             f"Total {lbl} funding (median) is {pct(abs(growth))} {'above' if growth >= 0 else 'below'} total {fy_label(FY_NOW - 1)} funding of {money(prev)}.",
             f"Proposals under review win {pct(succ['Under Review'], 0)} of the time; submitted proposals win {pct(succ['Submitted'], 0)}.",
             f"By college, {colleges[0][0]} brings in the most ({money(colleges[0][1])} expected in {lbl}), then {colleges[1][0]} ({money(colleges[1][1])}) and {colleges[2][0]} ({money(colleges[2][1])})."]
    template = (f"{lbl} research funding is projected at {money(pr['P50'])}, with an 80% range of {money(pr['P10'])} to "
                f"{money(pr['P90'])}. {money(cont)} is already secured from continuing awards, and the pipeline adds about "
                f"{money(expected_new)}. That is {pct(abs(growth))} {'up' if growth >= 0 else 'down'} on {fy_label(FY_NOW - 1)}.")
    return result(
        tool="grant_forecast", title=f"{lbl} research funding forecast",
        headline={"label": f"Projected {lbl} research funding (P50)", "value": pr["P50"], "fmt": "money",
                  "sub": f"Secured {money(cont)} · pipeline +{money(expected_new)} · {'+' if growth >= 0 else '-'}{pct(abs(growth))} YoY"},
        values={"funding_p10": pr["P10"], "funding_p50": pr["P50"], "funding_p90": pr["P90"], "continuing": cont,
                "expected_new": expected_new, "prior_year": prev, "growth": growth, "pipeline_proposals": len(pipe),
                "success_under_review": succ["Under Review"], "success_submitted": succ["Submitted"]},
        facts=facts,
        method={"summary": "Continuing multi-year awards plus a Monte Carlo over every open proposal starting this fiscal year.",
                "formula": f"{lbl} = Σ active award annual amounts + Σ (awarded? × annual amount × share of {lbl} remaining)",
                "steps": [f"Stage success rates from {stage_success()['All']['_n']:,} past proposals: " +
                          ", ".join(f"{s} {pct(succ[s], 0)}" for s in UNI_STAGES),
                          "Rates are sponsor-specific (NIH, NSF, DOE, DoD, Industry, Foundations)",
                          f"{MC_SIMS:,} simulations, fixed seed",
                          "Fiscal year runs 1 July to 30 June"]},
        inputs=[item("Continuing awards", cont, "money"), item("Open proposals starting " + lbl, len(pipe)),
                item("Expected new funding", expected_new, "money"), item(f"{fy_label(FY_NOW - 1)} funding", prev, "money")],
        percentiles={"label": "Funding", "fmt": "money", **pr},
        sources=["grant_awards (1,780 awards)", f"proposal_history ({stage_success()['All']['_n']:,} proposals)",
                 f"grant_pipeline ({len(pipe)} proposals starting {lbl})"],
        chart={"type": "bar", "title": "Research funding by fiscal year", "fmt": "money",
               "labels": [fy_label(fy) for fy in hist] + [lbl + " (fcst)"],
               "series": [{"name": "Funding", "data": list(hist.values()) + [pr["P50"]]}], "highlight_last": True,
               "band": {"low": pr["P10"], "high": pr["P90"]}},
        template=template,
        decisions=[{"key": "growth", "type": "noul",
                    "question": f"Will {lbl} research funding grow over {fy_label(FY_NOW - 1)}?",
                    "state": _state_text(facts)}],
        followups=["Which sponsors are we most dependent on, and what's the risk?"],
    )


def sponsor_concentration() -> dict:
    total, by_sponsor, _, _, _ = _simulate()
    med = {sp: float(np.median(v)) for sp, v in by_sponsor.items()}
    tot = sum(med.values())
    shares = {sp: v / tot for sp, v in med.items()}
    order = sorted(SPONSORS, key=lambda s: -shares[s])
    hhi = sum((s * 100) ** 2 for s in shares.values())
    top, second = order[0], order[1]
    # Sponsor volatility = std of year-over-year change in that sponsor's active funding (FY23..FY27)
    vol = {}
    for sp in SPONSORS:
        series = np.array([active_funding(fy, "sponsor").get(sp, 0.0) for fy in range(FY_NOW - 4, FY_NOW + 1)])
        ch = series[1:] / np.maximum(series[:-1], 1) - 1
        vol[sp] = float(max(np.std(ch, ddof=1), 0.05))
    cut = 0.15
    impact = med[top] * cut
    lbl = fy_label(FY_NOW)
    facts = [f"We are most dependent on {top}, at {pct(shares[top])} of projected {lbl} research funding ({money(med[top])}).",
             f"{second} is next at {pct(shares[second])}; together they are {pct(shares[top] + shares[second])} of funding.",
             f"A 15% cut in {top} funding would cost {money(impact)} a year.",
             f"The concentration index (HHI) is {hhi:,.0f}; above 2,500 is considered highly concentrated."]
    template = (f"We are most dependent on {top}, at {pct(shares[top])} of {lbl} funding, about {money(med[top])}. "
                f"{second} adds {pct(shares[second])}, so two sponsors carry {pct(shares[top] + shares[second])}. "
                f"A 15% cut at {top} would cost {money(impact)} a year. The concentration index is {hhi:,.0f}.")
    return result(
        tool="sponsor_concentration", title=f"Sponsor dependence · {lbl}",
        headline={"label": f"{top} share of {lbl} research funding", "value": shares[top], "fmt": "pct",
                  "sub": f"{money(med[top])} · 15% cut = {money(impact)} · HHI {hhi:,.0f}"},
        values={"shares": shares, "amounts": med, "top_sponsor": top, "top_share": shares[top], "second": second,
                "second_share": shares[second], "top_two_share": shares[top] + shares[second], "hhi": hhi,
                "cut_scenario": cut, "cut_impact": impact, "volatility": vol, "total": tot},
        facts=facts,
        method={"summary": "Projected funding by sponsor (median of the funding simulation), share of total, and a Herfindahl-Hirschman concentration index.",
                "formula": "Shareᵢ = Fundingᵢ ÷ Σ Funding;  HHI = Σ (Shareᵢ × 100)²;  Cut impact = Funding_top × 15%",
                "steps": ["Funding = continuing awards + simulated new awards, by sponsor",
                          "HHI < 1,500 unconcentrated · 1,500–2,500 moderate · > 2,500 high",
                          "Sponsor volatility (std of YoY change): " + ", ".join(f"{s} {pct(vol[s], 0)}" for s in order)]},
        inputs=[item(f"{sp} funding", med[sp], "money") for sp in order],
        sources=["grant_awards", "grant_pipeline", "proposal_history"],
        chart={"type": "hbar", "title": f"{lbl} projected funding by sponsor", "fmt": "money",
               "labels": order, "series": [{"name": "Funding", "data": [med[s] for s in order]}]},
        template=template,
        decisions=[{"key": "concentrated", "type": "noul",
                    "question": "Is this research portfolio dangerously dependent on a single sponsor?",
                    "state": _state_text([f"{sp}: {pct(shares[sp])} of funding ({money(med[sp])}), year-over-year volatility {pct(vol[sp], 0)}" for sp in order]
                                         + [f"HHI {hhi:,.0f}"])}],
    )


def score_research_sites() -> dict:
    rows = [dict(r) for r in q("SELECT * FROM research_sites")]
    criteria = [("aligned_pipeline", "Aligned grant pipeline", 30, True, "money"),
                ("capital_cost", "Capital cost", 25, False, "money"),
                ("faculty_nearby", "Faculty within 15 minutes", 20, True, "num"),
                ("industry_partners", "Industry partners nearby", 15, True, "num"),
                ("months_to_open", "Months to open", 10, False, "num")]
    scores, contrib, norm, wins = score_options(rows, criteria, salt=13)
    order = np.argsort(-scores)
    bi = order[0]
    best, second = rows[bi], rows[order[1]]
    drivers = [criteria[j][1] for j in np.argsort(-contrib[bi])[:2]]
    weakest = criteria[int(np.argmin(norm[bi]))][1]
    close = scores[bi] - scores[order[1]] < 3 or wins[bi] < 0.55
    facts = [(f"It is a close call: {best['site']} edges {second['site']}, {scores[bi]:.0f} to {scores[order[1]]:.0f} out of 100." if close else
              f"The evidence favors {best['site']}, scoring {scores[bi]:.0f} out of 100 versus {scores[order[1]]:.0f} for {second['site']}."),
             f"It wins {pct(wins[bi], 0)} of 2,000 weight-sensitivity scenarios.",
             f"Its strengths are {drivers[0].lower()} and {drivers[1].lower()}; the tradeoff is {weakest.lower()}.",
             f"Capital cost is {money(best['capital_cost'])} with {money(best['aligned_pipeline'])} of aligned proposals in the pipeline."]
    template = (f"{'It is a close call, but the evidence leans to' if close else 'The evidence favors'} {best['site']}. It scores {scores[bi]:.0f} of 100 versus {scores[order[1]]:.0f} for "
                f"{second['site']} and wins {pct(wins[bi], 0)} of sensitivity scenarios. Strengths: {drivers[0].lower()} and "
                f"{drivers[1].lower()}. Tradeoff: {weakest.lower()}. Capital cost is {money(best['capital_cost'])}.")
    return result(
        tool="score_research_sites", title="Next research center: decision brief",
        headline={"label": "Close call · evidence leans to" if close else "Evidence favors", "value": best["site"], "fmt": "text",
                  "sub": f"Score {scores[bi]:.0f}/100 · wins {pct(wins[bi], 0)} of sensitivity runs"},
        values={"ranked": [{"site": rows[i]["site"], "score": float(scores[i]), "win_share": float(wins[i])} for i in order],
                "best": best["site"], "best_score": float(scores[bi]), "runner_up": second["site"],
                "runner_up_score": float(scores[order[1]]), "best_win_share": float(wins[bi]),
                "capital_cost": best["capital_cost"], "aligned_pipeline": best["aligned_pipeline"]},
        facts=facts,
        method={"summary": "Weighted multi-criteria score (min-max normalised) with 2,000 random re-weightings to test robustness.",
                "formula": "Score = Σ wᵢ × normalised(criterionᵢ) × 100",
                "steps": [", ".join(f"{c[1]} {c[2]}%" for c in criteria),
                          "Aligned pipeline = open proposals from the colleges the site would serve",
                          f"Drivers: {drivers[0]}, {drivers[1]} · tradeoff: {weakest}"]},
        inputs=[item(r["site"], float(scores[i]), "score") for i, r in enumerate(rows)],
        sources=["research_sites", "grant_pipeline"],
        chart={"type": "hbar", "title": "Site score (0–100)", "fmt": "score",
               "labels": [rows[i]["site"] for i in order], "series": [{"name": "Score", "data": [float(scores[i]) for i in order]}]},
        table={"columns": ["Site", "Score", "Wins", "Capital", "Faculty", "Aligned pipeline", "Months"],
               "rows": [[rows[i]["site"], f"{scores[i]:.0f}", pct(wins[i], 0), money(rows[i]["capital_cost"]),
                         num(rows[i]["faculty_nearby"]), money(rows[i]["aligned_pipeline"]), str(rows[i]["months_to_open"])]
                        for i in order]},
        template=template,
        decisions=[{"key": "site", "type": "choice", "question": "Which site does the evidence favor for the next research center?",
                    "options": {r["site"]: f"Open the research center at {r['site']}" for r in rows},
                    "state": _state_text([f"{r['site']}: capital cost {money(r['capital_cost'])}, {r['faculty_nearby']} faculty nearby, "
                                          f"aligned pipeline {money(r['aligned_pipeline'])}, {r['industry_partners']} industry partners, "
                                          f"{r['months_to_open']} months to open" for r in rows]
                                         + ["Priorities: aligned grant pipeline and capital cost matter most."])}],
    )


def enrollment_forecast() -> dict:
    fun = q("SELECT * FROM enrollment_funnel ORDER BY fall_year")
    plan = {r["segment"]: r for r in q("SELECT * FROM enrollment_plan")}
    segs = list(plan)
    g = rng(17)
    rows = []
    total = np.zeros(MC_SIMS)
    admits_total = 0
    for s in segs:
        hist = [r for r in fun if r["segment"] == s]
        yields = np.array([r["enrolled"] / r["admits"] for r in hist])
        years = np.array([r["fall_year"] for r in hist], dtype=float)
        slope, icpt = np.polyfit(years, yields, 1)
        fit = icpt + slope * 2027
        resid = yields - (icpt + slope * years)
        sd = float(max(np.sqrt((resid ** 2).sum() / max(len(yields) - 2, 1)), 0.008))
        sim = plan[s]["planned_admits"] * np.clip(fit + sd * g.standard_normal(MC_SIMS), 0, 1)
        total += sim
        admits_total += plan[s]["planned_admits"]
        rows.append({"segment": s, "admits": plan[s]["planned_admits"], "yield_2026": float(yields[-1]),
                     "yield_2027": float(fit), "trend_per_year": float(slope), "enrolled_p50": float(np.median(sim)),
                     "target": plan[s]["enrollment_target"]})
    pr = p10_50_90(total)
    y = pr["P50"] / admits_total
    y_last = sum(r["enrolled"] for r in fun if r["fall_year"] == 2026) / sum(r["admits"] for r in fun if r["fall_year"] == 2026)
    target = sum(plan[s]["enrollment_target"] for s in segs)
    worst = min(rows, key=lambda r: r["trend_per_year"])
    facts = [f"Projected Fall 2027 yield is {pct(y)}, versus {pct(y_last)} for Fall 2026.",
             f"That means about {num(pr['P50'])} new students from {num(admits_total)} planned admits; the 80% range is {num(pr['P10'])} to {num(pr['P90'])}.",
             f"The enrollment target is {num(target)}, so the median projection is {num(abs(pr['P50'] - target))} {'above' if pr['P50'] >= target else 'below'} target.",
             f"{worst['segment']} yield is falling fastest, about {abs(worst['trend_per_year']) * 100:.1f} points a year, to {pct(worst['yield_2027'])}.",
             "Projected yield by segment, highest first: " + "; ".join(f"{r['segment']} {pct(r['yield_2027'])}" for r in sorted(rows, key=lambda r: -r["yield_2027"])) + "."]
    template = (f"Fall 2027 yield is projected at {pct(y)}, versus {pct(y_last)} last fall. That is about {num(pr['P50'])} "
                f"new students from {num(admits_total)} admits, {num(abs(pr['P50'] - target))} {'above' if pr['P50'] >= target else 'below'} "
                f"the target of {num(target)}. Watch {worst['segment']} yield, falling about "
                f"{abs(worst['trend_per_year']) * 100:.1f} points a year.")
    return result(
        tool="enrollment_forecast", title="Fall 2027 enrollment yield",
        headline={"label": "Projected Fall 2027 yield", "value": y, "fmt": "pct",
                  "sub": f"{num(pr['P50'])} enrolled (P50) vs target {num(target)} · Fall 2026 {pct(y_last)}"},
        values={"yield": y, "yield_last": y_last, "enrolled_p10": pr["P10"], "enrolled_p50": pr["P50"],
                "enrolled_p90": pr["P90"], "admits": admits_total, "target": target, "gap": pr["P50"] - target,
                "segments": rows, "worst_segment": worst["segment"], "worst_trend": worst["trend_per_year"]},
        facts=facts,
        method={"summary": "Per-segment yield trend fitted on three admission cycles, applied to planned admits, with Monte Carlo on the residual spread.",
                "formula": "Enrolled = Σ planned admitsₛ × (trend-fit yieldₛ + ε);  Yield = Enrolled ÷ Admits",
                "steps": [f"{r['segment']}: {pct(r['yield_2026'])} → {pct(r['yield_2027'])}" for r in rows] + [f"{MC_SIMS:,} simulations"]},
        inputs=[item(f"{r['segment']} admits", r["admits"]) for r in rows],
        percentiles={"label": "New students", "fmt": "num", **pr},
        sources=["enrollment_funnel (Fall 2024–2026)", "enrollment_plan (Fall 2027)"],
        chart={"type": "bar", "title": "Yield by segment: Fall 2026 vs Fall 2027 projection", "fmt": "pct",
               "labels": [r["segment"] for r in rows],
               "series": [{"name": "Fall 2026", "data": [r["yield_2026"] for r in rows], "muted": True},
                          {"name": "Fall 2027 (proj.)", "data": [r["yield_2027"] for r in rows]}]},
        template=template,
        decisions=[{"key": "meet", "type": "noul", "question": "Will Fall 2027 new enrollment meet its target?",
                    "state": _state_text(facts)},
                   {"key": "focus", "type": "choice", "question": "Which segment should recruitment focus on first?",
                    "options": {r["segment"]: r["segment"] for r in rows},
                    "state": _state_text([f"{r['segment']}: yield {pct(r['yield_2026'])} last fall, projected {pct(r['yield_2027'])}, "
                                          f"trend {r['trend_per_year'] * 100:+.1f} points a year, {num(r['admits'])} planned admits" for r in rows])}],
    )


def board_summary_university() -> dict:
    gf, sc, en = grant_forecast(), sponsor_concentration(), enrollment_forecast()
    facts = [gf["facts"][0], sc["facts"][0], en["facts"][0]]
    template = (f"Here is the board summary. {fy_label(FY_NOW)} research funding is projected at {money(gf['values']['funding_p50'])}. "
                f"{sc['values']['top_sponsor']} carries {pct(sc['values']['top_share'])} of it. Fall 2027 yield is projected at "
                f"{pct(en['values']['yield'])}, about {num(en['values']['enrolled_p50'])} new students.")
    return result(
        tool="board_summary_university", title="Board summary",
        headline=gf["headline"],
        values={"funding_p50": gf["values"]["funding_p50"], "top_sponsor": sc["values"]["top_sponsor"],
                "top_share": sc["values"]["top_share"], "yield": en["values"]["yield"],
                "enrolled_p50": en["values"]["enrolled_p50"]},
        facts=facts, method={"summary": "Grant forecast, sponsor concentration and enrollment forecast run together.",
                             "formula": "See underlying tools", "steps": ["grant_forecast", "sponsor_concentration", "enrollment_forecast"]},
        inputs=gf["inputs"][:3], percentiles=gf.get("percentiles"),
        sources=sorted(set(gf["sources"] + sc["sources"] + en["sources"])), chart=gf["chart"], template=template,
        decisions=gf.get("decisions", []) + sc.get("decisions", []),
    )


def _state_text(lines: list[str]) -> str:
    return "Alta Vista University (fictional). " + " ".join(l if l.endswith(".") else l + "." for l in lines)
