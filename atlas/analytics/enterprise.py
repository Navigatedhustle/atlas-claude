"""Meridian Dynamics (enterprise persona) analytics. Pure Python + numpy, deterministic."""
from __future__ import annotations

import datetime as dt
import math
from functools import lru_cache

import numpy as np

from atlas.analytics.common import (TODAY, item, money, num, p10_50_90, pct, q, q1, result, rng,
                                    score_options)
from atlas.config import MC_SIMS

STAGES = ["Prospect", "Qualify", "Propose", "Commit"]
REGIONS = ["West", "Central", "East", "International"]
CUR_Y, CUR_Q = TODAY.year, (TODAY.month - 1) // 3 + 1
Q_START = dt.date(CUR_Y, 3 * (CUR_Q - 1) + 1, 1)
Q_END = dt.date(CUR_Y, 12, 31) if CUR_Q == 4 else dt.date(CUR_Y, 3 * CUR_Q + 1, 1) - dt.timedelta(days=1)
QLABEL = f"Q{CUR_Q} {CUR_Y}"


# ----------------------------------------------------------------------------- model inputs

@lru_cache(maxsize=1)
def conversion_rates() -> dict:
    """P(won | deal reached stage), by region, from 3 years of closed deals."""
    rows = q("SELECT region, max_stage, outcome, COUNT(*) n FROM deals_history GROUP BY 1,2,3")
    out = {}
    for region in REGIONS + ["All"]:
        reached = np.zeros(4)
        won = np.zeros(4)
        for r in rows:
            if region != "All" and r["region"] != region:
                continue
            idx = STAGES.index(r["max_stage"])
            reached[: idx + 1] += r["n"]
            if r["outcome"] == "won":
                won[: idx + 1] += r["n"]
        out[region] = {s: float(won[i] / reached[i]) for i, s in enumerate(STAGES)}
        out[region]["_n"] = int(reached[0])
    return out


@lru_cache(maxsize=1)
def slip_rate() -> float:
    """Share of won deals (last 4 quarters) that closed in a later quarter than originally forecast."""
    rows = q("SELECT original_close_date o, closed_date c FROM deals_history WHERE outcome='won' AND closed_date >= ?",
             (TODAY - dt.timedelta(days=365)).isoformat())
    def qi(s):
        d = dt.date.fromisoformat(s)
        return d.year * 4 + (d.month - 1) // 3
    slipped = sum(1 for r in rows if qi(r["c"]) > qi(r["o"]))
    return slipped / max(len(rows), 1)


def open_deals(quarter_only: bool = True) -> list:
    if quarter_only:
        return q("SELECT * FROM open_pipeline WHERE close_date BETWEEN ? AND ? ORDER BY amount DESC",
                 TODAY.isoformat(), Q_END.isoformat())
    return q("SELECT * FROM open_pipeline ORDER BY amount DESC")


def qtd_won(region: str | None = None, start: dt.date = Q_START, end: dt.date = TODAY) -> float:
    sql = "SELECT COALESCE(SUM(amount),0) FROM deals_history WHERE outcome='won' AND closed_date >= ? AND closed_date < ?"
    args = [start.isoformat(), end.isoformat()]
    if region:
        sql += " AND region = ?"
        args.append(region)
    return float(q1(sql, *args))


@lru_cache(maxsize=1)
def recurring_model() -> dict:
    """Next-quarter recurring revenue per region = last quarter x (1 + mean q/q growth of last 4 quarters)."""
    out = {}
    for region in REGIONS:
        rows = q("SELECT recurring_revenue r FROM revenue_quarterly WHERE region=? ORDER BY year, quarter", region)
        rec = np.array([r["r"] for r in rows])
        g = rec[1:] / rec[:-1] - 1
        g4 = g[-4:]
        out[region] = {"last": float(rec[-1]), "growth": float(g4.mean()), "sd": float(max(g4.std(ddof=1), 0.002)),
                       "next": float(rec[-1] * (1 + g4.mean()))}
    return out


@lru_cache(maxsize=1)
def margin_model() -> dict:
    rows = q("SELECT * FROM costs_quarterly WHERE kind='actual' ORDER BY year DESC, quarter DESC LIMIT 4")
    revs = []
    cogs = []
    for r in rows:
        revs.append(q1("SELECT SUM(revenue) FROM revenue_quarterly WHERE year=? AND quarter=?", r["year"], r["quarter"]))
        cogs.append(r["cogs"])
    cogs_pct = float(sum(cogs) / sum(revs))
    b = q("SELECT * FROM costs_quarterly WHERE kind='budget' AND year=? AND quarter=?", CUR_Y, CUR_Q)[0]
    opex = {"Sales": b["opex_sales"], "Marketing": b["opex_marketing"], "R&D": b["opex_rnd"], "G&A": b["opex_ga"]}
    return {"cogs_pct": cogs_pct, "gross_margin": 1 - cogs_pct, "opex": opex, "opex_total": float(sum(opex.values()))}


def _simulate(stage_delta: dict | None = None, sims: int = MC_SIMS, salt: int = 1):
    """Monte Carlo of current-quarter revenue by region. Returns (total samples, per-region samples, deal p's)."""
    conv = conversion_rates()
    slip = slip_rate()
    rec = recurring_model()
    deals = open_deals()
    g = rng(salt)
    amounts = np.array([d["amount"] for d in deals])
    p = np.array([conv[d["region"]][d["stage"]] for d in deals]) * (1 - slip)
    if stage_delta:
        for i, d in enumerate(deals):
            if d["stage"] in stage_delta:
                p[i] = max(0.0, (conv[d["region"]][d["stage"]] + stage_delta[d["stage"]]) * (1 - slip))
    u = g.random((sims, len(deals)))
    wins = u < p  # same random draws for every scenario (common random numbers)
    region_idx = np.array([REGIONS.index(d["region"]) for d in deals])
    per_region = {}
    total = np.zeros(sims)
    z = g.standard_normal((sims, len(REGIONS)))
    for j, region in enumerate(REGIONS):
        m = region_idx == j
        new = (wins[:, m] * amounts[m]).sum(axis=1)
        recurring = rec[region]["next"] * (1 + rec[region]["sd"] * z[:, j])
        s = recurring + qtd_won(region) + new
        per_region[region] = s
        total += s
    return total, per_region, p, deals


@lru_cache(maxsize=8)
def _cached_sim(stage_delta_key: tuple = ()):
    return _simulate(dict(stage_delta_key) if stage_delta_key else None)


def profit_from_revenue(rev: np.ndarray | float):
    mm = margin_model()
    return rev * mm["gross_margin"] - mm["opex_total"]


def forecast_by_region() -> list[dict]:
    """Public helper: the same Monte Carlo as forecast_quarter, split by region (P10/P50/P90 revenue, USD)."""
    _, per_region, _, _ = _cached_sim()
    out = []
    for region, s in per_region.items():
        p = p10_50_90(s)
        out.append({"region": region, "p10": p["P10"], "p50": p["P50"], "p90": p["P90"]})
    return out


# ----------------------------------------------------------------------------- tools

def forecast_quarter() -> dict:
    total, per_region, p, deals = _cached_sim()
    profit = profit_from_revenue(total)
    pr, pp = p10_50_90(total), p10_50_90(profit)
    conv, slip, rec, mm = conversion_rates()["All"], slip_rate(), recurring_model(), margin_model()
    rec_total = sum(v["next"] for v in rec.values())
    qtd = qtd_won()
    amounts = np.array([d["amount"] for d in deals])
    weighted = float((amounts * p).sum())
    var = amounts ** 2 * p * (1 - p)
    top = np.argsort(-var)[:5]
    drivers = [{"account": deals[i]["account_name"], "stage": deals[i]["stage"], "amount": float(deals[i]["amount"]),
                "p_win": float(p[i]), "variance_share": float(var[i] / var.sum())} for i in top]
    hist = q("SELECT year, quarter, SUM(revenue) rev FROM revenue_quarterly GROUP BY 1,2 ORDER BY 1,2")[-7:]
    labels = [f"Q{h['quarter']} {h['year']}" for h in hist] + [QLABEL + " (fcst)"]
    data = [float(h["rev"]) for h in hist] + [pr["P50"]]
    last_year = q1("SELECT SUM(revenue) FROM revenue_quarterly WHERE year=? AND quarter=?", CUR_Y - 1, CUR_Q)
    prev = q("SELECT year, quarter, SUM(revenue) rev FROM revenue_quarterly GROUP BY 1,2 ORDER BY 1 DESC, 2 DESC LIMIT 1")[0]
    yoy = pr["P50"] / last_year - 1
    margin = pp["P50"] / pr["P50"]
    facts = [
        f"Projected {QLABEL} revenue is {money(pr['P50'])} at the median (P50).",
        f"The 80% range runs from {money(pr['P10'])} (P10) to {money(pr['P90'])} (P90).",
        f"Projected operating profit is {money(pp['P50'])}, an operating margin of {pct(margin)}.",
        f"Projected {QLABEL} revenue is {pct(abs(yoy))} {'above' if yoy >= 0 else 'below'} Q{CUR_Q} {CUR_Y - 1} revenue of {money(last_year)} (same quarter last year).",
        f"Last quarter (Q{prev['quarter']} {prev['year']}) actual revenue was {money(prev['rev'])}.",
        f"By amount, the largest open deal closing this quarter is {deals[0]['account_name']} at {money(deals[0]['amount'])}, in the {deals[0]['stage']} stage.",
        f"The deal adding the most uncertainty to the forecast (not the largest) is {drivers[0]['account']}, {money(drivers[0]['amount'])} in the {drivers[0]['stage']} stage.",
        f"The projection is about {money(rec_total)} of recurring revenue, plus {money(qtd)} already closed this quarter, plus about {money(weighted)} expected from open deals.",
        f"{len(deals)} open deals worth {money(float(amounts.sum()))} are scheduled to close this quarter.",
        f"Gross margin is {pct(mm['gross_margin'])}, the average of the last four quarters of actual results.",
    ]
    template = (f"{QLABEL} revenue is projected at {money(pr['P50'])}, with an 80% range of {money(pr['P10'])} to "
                f"{money(pr['P90'])}. Operating profit lands near {money(pp['P50'])}, a {pct(margin)} margin. "
                f"The deal adding the most uncertainty is {drivers[0]['account']} at {money(drivers[0]['amount'])}.")
    return result(
        tool="forecast_quarter", title=f"{QLABEL} revenue and profit forecast",
        headline={"label": f"Projected {QLABEL} revenue (P50)", "value": pr["P50"], "fmt": "money",
                  "sub": f"Operating profit {money(pp['P50'])} · margin {pct(margin)}"},
        values={"revenue_p10": pr["P10"], "revenue_p50": pr["P50"], "revenue_p90": pr["P90"],
                "profit_p10": pp["P10"], "profit_p50": pp["P50"], "profit_p90": pp["P90"],
                "operating_margin": margin, "yoy_growth": yoy, "last_year_revenue": float(last_year),
                "recurring_base": rec_total, "closed_qtd": qtd, "weighted_pipeline": weighted,
                "slip_rate": slip, "gross_margin": mm["gross_margin"], "opex_budget": mm["opex_total"],
                "open_deals": len(deals), "drivers": drivers},
        facts=facts,
        method={"summary": "Recurring base + closed-won this quarter + Monte Carlo over every open deal, "
                           "using win rates learned from 3 years of closed deals.",
                "formula": "Revenue = Recurring + ClosedQTD + Σ(deal won? × amount); "
                           "P(won) = stage win rate × (1 − slip rate); Profit = Revenue × GM − Opex budget",
                "steps": [f"Stage win rates from {conversion_rates()['All']['_n']:,} closed deals: " +
                          ", ".join(f"{s} {pct(conv[s], 0)}" for s in STAGES),
                          f"Slip rate (won deals that closed a quarter late): {pct(slip)}",
                          f"{MC_SIMS:,} simulations, seed fixed, recurring base varied by its own q/q volatility",
                          f"Gross margin {pct(mm['gross_margin'])} (trailing 4 quarters), opex = approved {QLABEL} budget {money(mm['opex_total'])}"]},
        inputs=[item("Recurring revenue base (this quarter)", rec_total, "money"), item("Closed-won so far this quarter", qtd, "money"),
                item("Open deals closing this quarter", len(deals), "num"),
                item("Weighted pipeline", weighted, "money"),
                item("Commit win rate", conv["Commit"], "pct"), item("Slip rate", slip, "pct"),
                item("Gross margin", mm["gross_margin"], "pct"), item("Opex budget", mm["opex_total"], "money")],
        percentiles={"label": "Revenue", "fmt": "money", **pr, "profit": pp},
        sources=["revenue_quarterly (12 quarters x 4 regions)", f"deals_history ({conversion_rates()['All']['_n']:,} closed deals)",
                 f"open_pipeline ({len(deals)} deals closing {QLABEL})", "costs_quarterly (actuals + budget)"],
        chart={"type": "bar", "title": "Quarterly revenue", "fmt": "money", "labels": labels,
               "series": [{"name": "Revenue", "data": data}], "highlight_last": True,
               "band": {"low": pr["P10"], "high": pr["P90"]}},
        template=template,
        decisions=[{"key": "meet_ly", "type": "noul",
                    "question": f"Based on this evidence, will {QLABEL} revenue exceed {money(last_year)} (same quarter last year)?",
                    "state": _state_text(facts + [f"Recurring base {money(rec_total)}", f"Weighted pipeline {money(weighted)}"])}],
        followups=["What happens to Q4 profit if our Commit stage win rate drops ten points?",
                   "Which five deals are most at risk this quarter, and why?"],
    )


def what_if(stage: str = "Commit", delta_pts: float = -10.0) -> dict:
    stage = stage.title() if stage and stage.title() in STAGES else "Commit"
    delta = float(delta_pts) / 100.0
    base_total, _, _, deals = _cached_sim()
    scen_total, _, _, _ = _cached_sim(((stage, delta),))
    b_rev, s_rev = float(np.median(base_total)), float(np.median(scen_total))
    b_pro, s_pro = float(profit_from_revenue(b_rev)), float(profit_from_revenue(s_rev))
    d_rev, d_pro = s_rev - b_rev, s_pro - b_pro
    conv = conversion_rates()["All"]
    stage_deals = [d for d in deals if d["stage"] == stage]
    stage_amt = sum(d["amount"] for d in stage_deals)
    s_p10 = p10_50_90(profit_from_revenue(scen_total))
    direction = "drops" if delta < 0 else "rises"
    facts = [
        f"If the {stage} win rate {direction} {abs(delta_pts):.0f} points, from {pct(conv[stage], 0)} to {pct(conv[stage] + delta, 0)}, {QLABEL} profit moves from {money(b_pro)} to {money(s_pro)}.",
        f"That is a profit {'drop' if d_pro < 0 else 'gain'} of {money(abs(d_pro))} on a revenue {'drop' if d_rev < 0 else 'gain'} of {money(abs(d_rev))}.",
        f"{len(stage_deals)} deals worth {money(stage_amt)} sit in {stage} this quarter.",
        f"In the downside case (P10) profit would be {money(s_p10['P10'])}.",
    ]
    template = (f"If the {stage} win rate {direction} {abs(delta_pts):.0f} points, {QLABEL} profit moves from "
                f"{money(b_pro)} to {money(s_pro)}, {'a drop' if d_pro < 0 else 'a gain'} of {money(abs(d_pro))}. Revenue {'falls' if d_rev < 0 else 'rises'} {money(abs(d_rev))}, "
                f"driven by {len(stage_deals)} {stage} deals worth {money(stage_amt)}.")
    return result(
        tool="what_if", title=f"What if {stage} win rate {direction} {abs(delta_pts):.0f} pts?",
        headline={"label": f"{QLABEL} profit impact", "value": d_pro, "fmt": "money",
                  "sub": f"Profit {money(b_pro)} → {money(s_pro)} · revenue {money(d_rev)}"},
        values={"stage": stage, "delta_pts": float(delta_pts), "base_win_rate": conv[stage],
                "scenario_win_rate": conv[stage] + delta, "base_revenue": b_rev, "scenario_revenue": s_rev,
                "base_profit": b_pro, "scenario_profit": s_pro, "profit_change": d_pro, "revenue_change": d_rev,
                "stage_deals": len(stage_deals), "stage_amount": stage_amt, "scenario_profit_p10": s_p10["P10"]},
        facts=facts,
        method={"summary": "Re-run the same 10,000-path Monte Carlo with the stage win rate shifted; "
                           "identical random draws, so the difference is purely the scenario.",
                "formula": "ΔProfit = ΔRevenue × Gross margin (opex is fixed for the quarter)",
                "steps": [f"Baseline {stage} win rate {pct(conv[stage])} → scenario {pct(conv[stage] + delta)}",
                          f"Gross margin {pct(margin_model()['gross_margin'])} applied to the revenue change",
                          "Common random numbers across baseline and scenario"]},
        inputs=[item(f"{stage} deals this quarter", len(stage_deals)), item(f"{stage} amount", stage_amt, "money"),
                item("Baseline win rate", conv[stage], "pct"), item("Scenario win rate", conv[stage] + delta, "pct"),
                item("Gross margin", margin_model()["gross_margin"], "pct")],
        percentiles={"label": "Scenario profit", "fmt": "money", **s_p10},
        sources=["deals_history (stage win rates)", f"open_pipeline ({len(deals)} deals)", "costs_quarterly"],
        chart={"type": "bar", "title": f"{QLABEL} operating profit", "fmt": "money",
               "labels": ["Baseline", f"{stage} {delta_pts:+.0f} pts"],
               "series": [{"name": "Operating profit", "data": [b_pro, s_pro]}], "highlight_last": True},
        template=template,
        decisions=[{"key": "exposed", "type": "noul",
                    "question": f"Is {QLABEL} profit materially exposed to {stage}-stage execution?",
                    "state": _state_text(facts + [f"Baseline operating profit {money(b_pro)}"])}],
    )


@lru_cache(maxsize=1)
def risk_model() -> dict:
    """Logistic regression of P(lost) for deals that reached Propose/Commit, fit by IRLS on closed history."""
    rows = q("SELECT * FROM deals_history WHERE max_stage IN ('Propose','Commit')")
    X, y = _risk_features(rows)
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p)
        reg = 2.0 * np.eye(X.shape[1])
        reg[0, 0] = 0
        H = X.T @ (X * W[:, None]) + reg
        beta = beta + np.linalg.solve(H, X.T @ (y - p) - reg @ beta)
    p = 1 / (1 + np.exp(-(X @ beta)))
    mean_x = X.mean(axis=0)
    # Calibration check: mean predicted vs actual loss rate in deciles
    order = np.argsort(p)
    dec = np.array_split(order, 10)
    calib = [(float(p[d].mean()), float(y[d].mean())) for d in dec]
    auc = _auc(p, y)
    return {"beta": beta, "n": len(rows), "calibration": calib, "auc": auc, "base_loss": float(y.mean()), "mean_x": mean_x}


RISK_FEATURES = ["intercept", "no_exec_sponsor", "competitor", "discount_over_12", "close_date_pushes",
                 "idle_over_10"]


def _risk_features(rows):
    X = np.array([[1.0, 1 - r["exec_sponsor"], r["competitor"], max(r["discount_pct"] - 12, 0),
                   min(r["close_date_pushes"], 3), min(max(r["days_since_activity"] - 10, 0), 30) / 10]
                  for r in rows])
    y = np.array([1.0 if ("outcome" in r.keys() and r["outcome"] == "lost") else 0.0 for r in rows])
    return X, y


def _auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def deal_risk(n: int = 5) -> dict:
    n = int(max(1, min(int(n or 5), 10)))
    m = risk_model()
    deals = [d for d in open_deals() if d["stage"] in ("Propose", "Commit")]
    X, _ = _risk_features(deals)
    conv = conversion_rates()
    base = np.array([1 - conv[d["region"]][d["stage"]] for d in deals])
    adj = (X[:, 1:] - m["mean_x"][1:]) @ m["beta"][1:]
    risk = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + adj)))
    at_risk_amt = risk * np.array([d["amount"] for d in deals])
    order = np.argsort(-at_risk_amt)[:n]
    reasons_txt = {
        "no_exec_sponsor": lambda d: "no executive sponsor",
        "competitor": lambda d: "competitor engaged",
        "discount_over_12": lambda d: f"{d['discount_pct']:.0f}% discount requested",
        "close_date_pushes": lambda d: f"close date pushed {d['close_date_pushes']} time{'s' if d['close_date_pushes'] != 1 else ''}",
        "idle_over_10": lambda d: f"no activity in {d['days_since_activity']} days",
    }
    top = []
    for i in order:
        d = deals[i]
        contrib = (X[i] - m["mean_x"]) * m["beta"]
        why = [RISK_FEATURES[j] for j in np.argsort(-contrib) if j > 0 and contrib[j] > 0.1][:3] or ["no_exec_sponsor"][:0]
        top.append({"account": d["account_name"], "region": d["region"], "stage": d["stage"], "amount": float(d["amount"]),
                    "risk": float(risk[i]), "amount_at_risk": float(at_risk_amt[i]), "owner": d["rep_name"],
                    "close_date": d["close_date"], "reasons": [reasons_txt[w](d) for w in why],
                    "days_since_activity": d["days_since_activity"], "pushes": d["close_date_pushes"]})
    total_risk = sum(t["amount_at_risk"] for t in top)
    total_amt = sum(t["amount"] for t in top)
    facts = [f"The {n} most at-risk deals total {money(total_amt)}, with {money(total_risk)} expected to be lost."]
    for k, t in enumerate(top):
        facts.append(f"{k + 1}. {t['account']}, {money(t['amount'])} in {t['stage']}, {pct(t['risk'], 0)} loss risk: "
                     + "; ".join(t["reasons"]) + ".")
    t0 = top[0]
    template = (f"The {n} riskiest deals add up to {money(total_amt)}. Top of the list is {t0['account']}, "
                f"{money(t0['amount'])} in {t0['stage']} with {pct(t0['risk'], 0)} loss risk: "
                f"{', '.join(t0['reasons'])}. Next are {', '.join(t['account'] for t in top[1:4])}"
                f"{' and ' + top[4]['account'] if len(top) > 4 else ''}.")
    decisions = [{"key": f"risk_{k}", "type": "noul", "label": t["account"],
                  "question": "Is this deal at risk of being lost or slipping out of the quarter?",
                  "state": _state_text([f"Account: {t['account']}", f"Stage: {t['stage']}", f"Amount: {money(t['amount'])}",
                                        f"Close date: {t['close_date']} (today is {TODAY.isoformat()})",
                                        f"Days since last customer activity: {t['days_since_activity']}",
                                        f"Close date pushed {t['pushes']} times",
                                        "Signals: " + "; ".join(t["reasons"])])}
                 for k, t in enumerate(top)]
    return result(
        tool="deal_risk", title=f"Top {n} deals at risk · {QLABEL}",
        headline={"label": f"Value of top {n} at-risk deals", "value": total_amt, "fmt": "money",
                  "sub": f"Expected loss {money(total_risk)} · model AUC {m['auc']:.2f}"},
        values={"deals": top, "total_amount": total_amt, "expected_loss": total_risk, "model_auc": m["auc"],
                "training_deals": m["n"], "base_loss_rate": m["base_loss"]},
        facts=facts,
        method={"summary": f"Logistic regression trained on {m['n']:,} closed late-stage deals; ranked by amount × P(loss).",
                "formula": "logit P(loss) = logit(1 − stage win rate) + β·(x − x̄), x = [no sponsor, competitor, discount>12%, pushes, idle days]; "
                           "rank = amount × P(loss)",
                "steps": [f"Coefficients: " + ", ".join(f"{f} {b:+.2f}" for f, b in zip(RISK_FEATURES[1:], m["beta"][1:])),
                          f"In-sample AUC {m['auc']:.2f} on deals that reached Propose; base loss rate {pct(m['base_loss'])}",
                          "Stage-specific win rates set the baseline; deal signals shift it up or down",
                          "Reasons = the features adding most to each deal's log-odds"]},
        inputs=[item("Late-stage deals scored", len(deals)), item("Training deals", m["n"]),
                item("Model AUC", m["auc"], "dec"), item("Base loss rate", m["base_loss"], "pct")],
        sources=[f"deals_history ({m['n']:,} Propose/Commit outcomes)", f"open_pipeline ({len(deals)} late-stage deals)"],
        chart={"type": "hbar", "title": "Amount at risk (amount × P(loss))", "fmt": "money",
               "labels": [t["account"] for t in top],
               "series": [{"name": "Amount at risk", "data": [t["amount_at_risk"] for t in top]}]},
        table={"columns": ["Account", "Stage", "Amount", "Risk", "Why"],
               "rows": [[t["account"], t["stage"], money(t["amount"]), pct(t["risk"], 0), "; ".join(t["reasons"])] for t in top]},
        template=template, decisions=decisions,
    )


def region_vs_plan() -> dict:
    _, per_region, _, _ = _cached_sim()
    rows = []
    for region in REGIONS:
        ytd = float(q1("SELECT SUM(revenue) FROM revenue_quarterly WHERE year=? AND region=?", CUR_Y, region))
        q4 = float(np.median(per_region[region]))
        plan = float(q1("SELECT revenue_plan FROM plan_targets WHERE year=? AND region=?", CUR_Y, region))
        proj = ytd + q4
        rows.append({"region": region, "ytd": ytd, "q4_p50": q4, "projected": proj, "plan": plan,
                     "gap": proj - plan, "gap_pct": proj / plan - 1})
    worst = min(rows, key=lambda r: r["gap_pct"])
    tot_plan = sum(r["plan"] for r in rows)
    tot_proj = sum(r["projected"] for r in rows)
    facts = [f"{worst['region']} is furthest behind plan: projected {money(worst['projected'])} for FY{CUR_Y} against a plan of "
             f"{money(worst['plan'])}, a gap of {money(abs(worst['gap']))} or {pct(abs(worst['gap_pct']))}.",
             f"All FY{CUR_Y} figures are projections: Q1 to Q3 actual revenue plus the Q4 forecast."]
    for r in sorted(rows, key=lambda r: r["gap_pct"]):
        if r is not worst:
            facts.append(f"{r['region']}: projected {money(r['projected'])} vs plan {money(r['plan'])}, "
                         f"{pct(abs(r['gap_pct']))} {'ahead of' if r['gap'] >= 0 else 'behind'} plan.")
    facts.append(f"Company total: {money(tot_proj)} projected vs {money(tot_plan)} plan.")
    template = (f"{worst['region']} is furthest behind. It is projected at {money(worst['projected'])} for {CUR_Y} "
                f"against a {money(worst['plan'])} plan, short by {money(abs(worst['gap']))}, or {pct(abs(worst['gap_pct']))}. "
                f"Company-wide we project {money(tot_proj)} against {money(tot_plan)}.")
    return result(
        tool="region_vs_plan", title=f"FY{CUR_Y} revenue vs plan by region",
        headline={"label": f"{worst['region']} gap to FY{CUR_Y} plan", "value": worst["gap"], "fmt": "money",
                  "sub": f"{pct(worst['gap_pct'])} vs plan · projected {money(worst['projected'])}"},
        values={"regions": rows, "worst_region": worst["region"], "worst_gap": worst["gap"],
                "worst_gap_pct": worst["gap_pct"], "total_projected": tot_proj, "total_plan": tot_plan},
        facts=facts,
        method={"summary": "Actual revenue for Q1–Q3 plus the Q4 Monte Carlo median per region, compared to the annual plan.",
                "formula": "Projected FY = Σ actual Q1..Q3 + P50(Q4);  Gap = Projected − Plan",
                "steps": ["Q4 per region from the same 10,000-path simulation as the quarter forecast",
                          "Plan = approved FY revenue plan by region"]},
        inputs=[item(f"{r['region']} plan", r["plan"], "money") for r in rows],
        sources=["revenue_quarterly", "plan_targets", "open_pipeline"],
        chart={"type": "bar", "title": f"FY{CUR_Y} projected vs plan", "fmt": "money",
               "labels": REGIONS,
               "series": [{"name": "Plan", "data": [r["plan"] for r in rows], "muted": True},
                          {"name": "Projected", "data": [r["projected"] for r in rows]}]},
        template=template,
        decisions=[{"key": "intervene", "type": "choice", "question": "Which region needs executive intervention first?",
                    "options": {r["region"]: f"{r['region']} region" for r in rows},
                    "state": _state_text([f"{r['region']}: projected {money(r['projected'])} vs plan {money(r['plan'])}, "
                                          f"gap {money(r['gap'])} ({pct(r['gap_pct'])})" for r in rows])}],
    )


def _coverage_at(date: dt.date) -> dict:
    y = date.year
    qs, qe = dt.date(y, 10, 1), dt.date(y, 12, 31)
    plan = float(q1("SELECT SUM(new_bookings_plan) FROM plan_targets WHERE year=?", y))
    # Q4 share of new bookings in the prior full year
    prev = q("SELECT quarter, SUM(new_business_revenue) n FROM revenue_quarterly WHERE year=? GROUP BY 1", y - 1)
    share = [r["n"] for r in prev if r["quarter"] == 4][0] / sum(r["n"] for r in prev)
    target = plan * share
    won = qtd_won(None, qs, date)
    if date == TODAY:
        pipe = float(q1("SELECT SUM(amount) FROM open_pipeline WHERE close_date BETWEEN ? AND ?", date.isoformat(), qe.isoformat()))
    else:
        pipe = float(q1("SELECT SUM(amount) FROM pipeline_snapshots WHERE snapshot_date=?", date.isoformat()))
    remaining = max(target - won, 1)
    return {"date": date.isoformat(), "pipeline": pipe, "target": target, "won_qtd": won, "remaining": remaining,
            "coverage": pipe / remaining}


def pipeline_coverage() -> dict:
    now = _coverage_at(TODAY)
    ly = _coverage_at(dt.date(TODAY.year - 1, TODAY.month, TODAY.day))
    diff = now["coverage"] - ly["coverage"]
    facts = [f"Pipeline coverage for {QLABEL} is {now['coverage']:.2f}x: {money(now['pipeline'])} of open pipeline against "
             f"{money(now['remaining'])} of new bookings still needed.",
             f"On the same day last year coverage was {ly['coverage']:.2f}x ({money(ly['pipeline'])} against {money(ly['remaining'])}).",
             f"Coverage is {'up' if diff >= 0 else 'down'} {abs(diff):.2f}x year over year."]
    template = (f"Coverage is {now['coverage']:.2f}x today versus {ly['coverage']:.2f}x on this date last year. "
                f"We have {money(now['pipeline'])} of pipeline against {money(now['remaining'])} still to book.")
    return result(
        tool="pipeline_coverage", title=f"{QLABEL} pipeline coverage vs last year",
        headline={"label": "Pipeline coverage", "value": now["coverage"], "fmt": "x",
                  "sub": f"Last year {ly['coverage']:.2f}x · {'+' if diff >= 0 else ''}{diff:.2f}x"},
        values={"coverage": now["coverage"], "coverage_last_year": ly["coverage"], "change": diff,
                "pipeline": now["pipeline"], "remaining_target": now["remaining"], "quarter_target": now["target"],
                "won_qtd": now["won_qtd"], "pipeline_last_year": ly["pipeline"], "remaining_last_year": ly["remaining"]},
        facts=facts,
        method={"summary": "Open pipeline closing this quarter divided by new bookings still needed, measured on the same calendar day both years.",
                "formula": "Coverage = Open pipeline (closing in quarter) ÷ (Quarter new-bookings target − Closed-won QTD)",
                "steps": ["Quarter target = annual new-bookings plan × prior year's Q4 share",
                          "Last year's pipeline from the 8 Oct snapshot"]},
        inputs=[item("Open pipeline", now["pipeline"], "money"), item("Quarter target", now["target"], "money"),
                item("Closed-won QTD", now["won_qtd"], "money"), item("Last year pipeline", ly["pipeline"], "money")],
        sources=["open_pipeline", "pipeline_snapshots (2025-10-08)", "plan_targets", "deals_history"],
        chart={"type": "bar", "title": "Coverage on 8 Oct", "fmt": "x", "labels": [str(TODAY.year - 1), str(TODAY.year)],
               "series": [{"name": "Coverage", "data": [ly["coverage"], now["coverage"]]}], "highlight_last": True,
               "ref": {"value": 3.0, "label": "3.0x rule of thumb"}},
        template=template,
        decisions=[{"key": "enough", "type": "noul", "question": f"Is pipeline coverage sufficient to hit the {QLABEL} bookings target?",
                    "state": _state_text(facts + ["A common rule of thumb is 3x coverage of the remaining target."])}],
    )


def hiring_plan(year: int | None = None) -> dict:
    year = year or CUR_Y + 1
    target = float(q1("SELECT SUM(new_bookings_plan) FROM plan_targets WHERE year=?", year))
    cutoff = (TODAY - dt.timedelta(days=365)).isoformat()
    won_4q = float(q1("SELECT SUM(amount) FROM deals_history WHERE outcome='won' AND closed_date >= ?", cutoff))
    reps = q("SELECT * FROM reps")
    active = [r for r in reps if r["status"] == "active"]
    ramped = [r for r in active if r["hire_date"] <= (TODAY - dt.timedelta(days=365)).isoformat()]
    ramping = [r for r in active if r not in ramped]
    departed = [r for r in reps if r["status"] == "departed" and r["departure_date"] >= cutoff]
    avg_hc = len(active) + len(departed) / 2
    # Productivity per ramped-equivalent rep over the last 4 quarters (ramping reps count 0.5, leavers 0.5)
    equiv = len(ramped) + 0.5 * len(ramping) + 0.5 * len(departed)
    productivity = won_4q / equiv
    attrition = len(departed) / avg_hc
    capacity = len(ramped) * productivity + len(ramping) * productivity * 0.75
    attrition_loss = len(active) * attrition * productivity * 0.5
    net_capacity = capacity - attrition_loss
    gap = max(target - net_capacity, 0)
    ramp_q1 = (0.15 + 0.40 + 0.65 + 0.85) / 4
    per_hire = productivity * ramp_q1
    hires = math.ceil(gap / per_hire) if gap > 0 else 0
    ramp_q2 = (0.15 + 0.40 + 0.65) / 4
    hires_late = math.ceil(gap / (productivity * ramp_q2)) if gap > 0 else 0
    facts = [f"To hit the FY{year} new-bookings target of {money(target)} we need to hire {hires} reps, starting in Q1.",
             f"Before attrition, today's team of {len(active)} reps could deliver about {money(capacity)} of new bookings; expected attrition removes about {money(attrition_loss)}, leaving {money(net_capacity)} and a gap of {money(gap)} to the target.",
             f"A ramped rep closes about {money(productivity)} a year; a new hire delivers about {money(per_hire)} in year one.",
             f"Annual attrition is running at {pct(attrition)}.",
             f"If hiring slips to Q2 we would need {hires_late} reps instead."]
    template = (f"We need {hires} new reps, hired in Q1, to reach the FY{year} target of {money(target)}. "
                f"The current {len(active)} reps cover about {money(net_capacity)} after {pct(attrition)} attrition, "
                f"a gap of {money(gap)}. If hiring slips to Q2 that becomes {hires_late} reps.")
    return result(
        tool="hiring_plan", title=f"Reps needed for FY{year} target",
        headline={"label": "Reps to hire", "value": hires, "fmt": "num",
                  "sub": f"Gap {money(gap)} · {hires_late} if hiring slips to Q2"},
        values={"target": target, "active_reps": len(active), "ramped_reps": len(ramped), "ramping_reps": len(ramping),
                "productivity": productivity, "attrition_rate": attrition, "capacity": capacity,
                "attrition_loss": attrition_loss, "net_capacity": net_capacity, "gap": gap,
                "hires": hires, "hires_if_q2": hires_late, "first_year_per_hire": per_hire, "ramp_factor": ramp_q1},
        facts=facts,
        method={"summary": "Capacity of today's team after attrition, compared to next year's target; the gap is filled with ramping hires.",
                "formula": "Hires = ⌈(Target − (Ramped×P + Ramping×0.75P − Active×Attrition×0.5P)) ÷ (P × ramp)⌉",
                "steps": [f"P (productivity) = {money(won_4q)} won in last 4 quarters ÷ {equiv:.0f} ramped-equivalent reps",
                          "Ramp curve for a Q1 hire: 15% / 40% / 65% / 85% by quarter (avg " + pct(ramp_q1, 0) + ")",
                          f"Attrition = {len(departed)} departures ÷ {avg_hc:.0f} avg headcount"]},
        inputs=[item("FY target (new bookings)", target, "money"), item("Active reps", len(active)),
                item("Productivity / ramped rep", productivity, "money"), item("Attrition", attrition, "pct"),
                item("Year-one ramp", ramp_q1, "pct")],
        sources=["reps (120 records)", "deals_history (last 4 quarters won)", "plan_targets"],
        chart={"type": "bar", "title": f"FY{year} new bookings: capacity vs target", "fmt": "money",
               "labels": ["Current team", "After attrition", "New hires", "Target"],
               "series": [{"name": "Bookings", "data": [capacity, net_capacity, hires * per_hire, target]}],
               "highlight_last": True},
        template=template,
        decisions=[{"key": "feasible", "type": "noul",
                    "question": f"Is hiring {hires} reps in one quarter realistic for a team of {len(active)}?",
                    "state": _state_text(facts)}],
    )


def score_sites(cities: list[str] | None = None, seats: int = 40) -> dict:
    all_rows = [dict(r) for r in q("SELECT * FROM site_candidates")]
    names = [r["city"] for r in all_rows]
    pick = [c for c in (cities or []) if c in names]
    rows = [r for r in all_rows if r["city"] in pick] if len(pick) >= 2 else all_rows
    for r in rows:
        r["five_year_cost"] = seats * (r["rep_cost"] + r["cost_per_seat"]) * 5
    criteria = [("nearby_pipeline", "Nearby pipeline", 25, True, "money"),
                ("rep_cost", "Loaded rep cost", 25, False, "money"),
                ("talent_pool", "Sales talent pool", 20, True, "num"),
                ("cost_per_seat", "Cost per seat", 15, False, "money"),
                ("attrition_rate", "Rep attrition", 15, False, "pct")]
    scores, contrib, norm, wins = score_options(rows, criteria, salt=7, reference=all_rows)
    order = np.argsort(-scores)
    best, second = rows[order[0]], rows[order[1]]
    bi = order[0]
    drivers = [criteria[j][1] for j in np.argsort(-contrib[bi])[:2]]
    weakest = criteria[int(np.argmin(norm[bi]))][1]
    ranked = [{"city": rows[i]["city"], "score": float(scores[i]), "win_share": float(wins[i]),
               "five_year_cost": rows[i]["five_year_cost"]} for i in order]
    close = scores[bi] - scores[order[1]] < 3 or wins[bi] < 0.55
    facts = [(f"It is a close call: {best['city']} edges {second['city']}, {scores[bi]:.0f} to {scores[order[1]]:.0f} out of 100." if close else
              f"The evidence favors {best['city']}, scoring {scores[bi]:.0f} out of 100 versus {scores[order[1]]:.0f} for {second['city']}."),
             f"{best['city']} wins in {pct(wins[bi], 0)} of {2000:,} weight-sensitivity scenarios.",
             f"Its strengths are {drivers[0].lower()} and {drivers[1].lower()}; its weakest factor is {weakest.lower()}.",
             f"A {seats}-seat center in {best['city']} costs about {money(best['five_year_cost'])} over five years, versus {money(second['five_year_cost'])} in {second['city']}."]
    template = (f"{'It is a close call, but the evidence leans to' if close else 'The evidence favors'} {best['city']}: it scores {scores[bi]:.0f} of 100 versus {scores[order[1]]:.0f} "
                f"for {second['city']}, and wins {pct(wins[bi], 0)} of sensitivity scenarios. The tradeoff is "
                f"{weakest.lower()}. Five-year cost for {seats} seats is about {money(best['five_year_cost'])}.")
    state_lines = [f"{r['city']}: loaded rep cost {money(r['rep_cost'])}, talent pool {num(r['talent_pool'])}, "
                   f"cost per seat {money(r['cost_per_seat'])}, nearby open pipeline {money(r['nearby_pipeline'])}, "
                   f"rep attrition {pct(r['attrition_rate'])}" for r in rows]
    return result(
        tool="score_sites", title="Sales center site decision brief",
        headline={"label": "Close call · evidence leans to" if close else "Evidence favors", "value": best["city"], "fmt": "text",
                  "sub": f"Score {scores[bi]:.0f}/100 · wins {pct(wins[bi], 0)} of sensitivity runs"},
        values={"ranked": ranked, "best": best["city"], "runner_up": second["city"], "seats": seats,
                "best_score": float(scores[bi]), "runner_up_score": float(scores[order[1]]),
                "best_win_share": float(wins[bi])},
        facts=facts,
        method={"summary": "Weighted multi-criteria score (min-max normalised across all 5 candidate cities), then 2,000 random re-weightings to test robustness.",
                "formula": "Score = Σ wᵢ × normalised(criterionᵢ) × 100;  robustness = share of perturbed weightings won",
                "steps": [", ".join(f"{c[1]} {c[2]}%" for c in criteria),
                          f"Drivers for {best['city']}: {drivers[0]}, {drivers[1]}",
                          f"Tradeoff: weakest on {weakest}",
                          f"Five-year cost = {seats} seats × (rep cost + seat cost) × 5"]},
        inputs=[item(f"{r['city']} score", float(scores[i]), "score") for i, r in enumerate(rows)],
        sources=["site_candidates", "open_pipeline (nearby pipeline)", "accounts"],
        chart={"type": "hbar", "title": "Site score (0–100)", "fmt": "score",
               "labels": [r["city"] for r in ranked], "series": [{"name": "Score", "data": [r["score"] for r in ranked]}]},
        table={"columns": ["City", "Score", "Wins", "Rep cost", "Talent", "Pipeline", "5-yr cost"],
               "rows": [[rows[i]["city"], f"{scores[i]:.0f}", pct(wins[i], 0), money(rows[i]["rep_cost"]),
                         num(rows[i]["talent_pool"]), money(rows[i]["nearby_pipeline"]),
                         money(rows[i]["five_year_cost"])] for i in order]},
        template=template,
        decisions=[{"key": "site", "type": "choice", "question": "Which site does the evidence favor for a new sales center?",
                    "options": {r["city"]: f"Build the sales center in {r['city']}" for r in rows},
                    "state": _state_text(state_lines + ["Priorities: nearby pipeline and rep cost matter most, then talent pool, seat cost and attrition."])}],
    )


def board_summary() -> dict:
    f = forecast_quarter()
    r = region_vs_plan()
    c = pipeline_coverage()
    d = deal_risk(5)
    v = f["values"]
    facts = [f["facts"][0], f["facts"][2], r["facts"][0], c["facts"][0], d["facts"][0]]
    template = (f"Here is the board summary. {QLABEL} revenue is tracking to {money(v['revenue_p50'])} with operating profit of "
                f"{money(v['profit_p50'])}. {r['values']['worst_region']} is our biggest gap to plan at "
                f"{money(abs(r['values']['worst_gap']))}. Pipeline coverage is {c['values']['coverage']:.2f}x, versus "
                f"{c['values']['coverage_last_year']:.2f}x a year ago. Five deals worth {money(d['values']['total_amount'])} "
                f"need executive attention.")
    return result(
        tool="board_summary", title="Board summary",
        headline={"label": f"{QLABEL} revenue (P50)", "value": v["revenue_p50"], "fmt": "money",
                  "sub": f"Profit {money(v['profit_p50'])} · coverage {c['values']['coverage']:.2f}x · "
                         f"{r['values']['worst_region']} {money(r['values']['worst_gap'])} vs plan"},
        values={"revenue_p50": v["revenue_p50"], "profit_p50": v["profit_p50"], "operating_margin": v["operating_margin"],
                "worst_region": r["values"]["worst_region"], "worst_gap": r["values"]["worst_gap"],
                "worst_gap_pct": r["values"]["worst_gap_pct"], "coverage": c["values"]["coverage"],
                "coverage_last_year": c["values"]["coverage_last_year"], "risk_total": d["values"]["total_amount"],
                "revenue_p10": v["revenue_p10"], "revenue_p90": v["revenue_p90"],
                "total_projected": r["values"]["total_projected"], "total_plan": r["values"]["total_plan"]},
        facts=facts,
        method={"summary": "Four tools run together: quarter forecast, region vs plan, pipeline coverage, deal risk.",
                "formula": "See each underlying tool; no extra math is added in the summary",
                "steps": ["forecast_quarter → revenue, profit", "region_vs_plan → largest gap",
                          "pipeline_coverage → coverage vs last year", "deal_risk → five deals to watch"]},
        inputs=f["inputs"][:4] + [item("Coverage", c["values"]["coverage"], "x")],
        percentiles=f["percentiles"],
        sources=sorted(set(f["sources"] + r["sources"] + c["sources"])),
        chart=r["chart"], template=template,
        decisions=[{"key": "on_track", "type": "noul", "question": "Is the company on track for its annual plan?",
                    "state": _state_text(facts)}] + d.get("decisions", [])[:2],
    )


def _state_text(lines: list[str]) -> str:
    return "Meridian Dynamics (fictional company). " + " ".join(l if l.endswith(".") else l + "." for l in lines)
