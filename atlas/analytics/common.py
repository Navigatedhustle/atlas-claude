"""Shared helpers: database access, formatting, and the tool result contract.

Every tool returns a dict with at least {values, method, sources, chart}. Atlas adds
headline/facts/inputs/template so the UI and narrator never have to compute anything.
"""
from __future__ import annotations

import datetime as dt
import math
import sqlite3
from functools import lru_cache

import numpy as np

from atlas.config import AS_OF, DB_PATH

TODAY = dt.date.fromisoformat(AS_OF)


@lru_cache(maxsize=1)
def _con() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def q(sql: str, *args) -> list[sqlite3.Row]:
    return _con().execute(sql, args).fetchall()


def q1(sql: str, *args):
    row = _con().execute(sql, args).fetchone()
    return None if row is None else row[0]


def rng(salt: int = 0) -> np.random.Generator:
    from atlas.config import SEED
    return np.random.default_rng(SEED + salt)


# ----------------------------------------------------------------------------- formatting

def money(x: float, digits: int | None = None) -> str:
    """$647.2M, $2.41B, $412K. Sign preserved."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e9:
        return f"{s}${a / 1e9:.{2 if digits is None else digits}f}B"
    if a >= 1e6:
        return f"{s}${a / 1e6:.{1 if digits is None else digits}f}M"
    if a >= 1e3:
        return f"{s}${a / 1e3:.{0 if digits is None else digits}f}K"
    return f"{s}${a:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def num(x: float) -> str:
    return f"{x:,.0f}"


FMT = {"money": money, "pct": pct, "num": num, "x": lambda v: f"{v:.2f}x", "score": lambda v: f"{v:.0f}",
       "dec": lambda v: f"{v:.2f}"}


def fmt(v, kind: str) -> str:
    return FMT.get(kind, str)(v)


def item(label: str, value, kind: str = "num", note: str | None = None) -> dict:
    d = {"label": label, "value": value, "fmt": kind, "text": fmt(value, kind) if isinstance(value, (int, float)) else str(value)}
    if note:
        d["note"] = note
    return d


def result(*, tool: str, title: str, headline: dict, values: dict, facts: list[str], method: dict,
           inputs: list[dict], sources: list[str], chart: dict, template: str,
           percentiles: dict | None = None, decisions: list[dict] | None = None, table: dict | None = None,
           followups: list[str] | None = None) -> dict:
    out = {
        "tool": tool, "title": title, "headline": headline, "values": values, "facts": facts,
        "method": method, "inputs": inputs, "sources": sources, "chart": chart, "template": template,
        "as_of": AS_OF,
    }
    if percentiles:
        out["percentiles"] = percentiles
    if decisions:
        out["decisions"] = decisions
    if table:
        out["table"] = table
    if followups:
        out["followups"] = followups
    return out


def p10_50_90(samples: np.ndarray) -> dict:
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])
    return {"P10": float(p10), "P50": float(p50), "P90": float(p90)}


def score_options(rows: list[dict], criteria: list[tuple[str, str, float, bool, str]], salt: int, n_sens: int = 2000,
                  reference: list[dict] | None = None):
    """Weighted multi-criteria scoring with a weight-perturbation sensitivity check.

    criteria: (key, label, weight, higher_is_better, fmt)
    Returns per-option scores (0-100), per-criterion contributions, and how often each option
    wins when the weights are randomly perturbed (Dirichlet around the stated weights).
    """
    keys = [c[0] for c in criteria]
    w = np.array([c[2] for c in criteria], dtype=float)
    w = w / w.sum()
    m = np.array([[float(r[k]) for k in keys] for r in rows])
    # Normalise against the full candidate set when given, so comparing two options never stretches small gaps.
    ref = np.array([[float(r[k]) for k in keys] for r in (reference or rows)])
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    norm = (m - lo) / span
    for j, c in enumerate(criteria):
        if not c[3]:
            norm[:, j] = 1 - norm[:, j]
    contrib = norm * w * 100
    scores = contrib.sum(axis=1)
    g = rng(salt)
    ws = g.dirichlet(w * 40, size=n_sens)
    sims = ws @ norm.T
    wins = np.bincount(sims.argmax(axis=1), minlength=len(rows)) / n_sens
    return scores, contrib, norm, wins


def capabilities(persona: str) -> dict:
    """Out-of-scope / help answer: what Atlas can do. Contains no business numbers."""
    from atlas.config import ASSISTANT_NAME, PERSONAS
    org = PERSONAS[persona]["org"]
    areas = ({"enterprise": ["the Q4 revenue and profit forecast", "which deals are at risk and why",
                             "where to build a sales center", "what-if scenarios on win rates",
                             "which region is behind plan", "how many reps to hire",
                             "pipeline coverage versus last year", "a board summary"],
              "university": ["research funding for FY27", "sponsor dependence and risk",
                             "where to open the next research center", "fall enrollment yield"],
              "developer": ["writing and running SQL against the company database", "Python scripts and CSV exports",
                            "charts from the data", "small REST endpoints with tests", "unit tests for my own functions",
                            "explaining my own source code", "the local model endpoint and its live speed"]})[persona]
    facts = [f"I am {ASSISTANT_NAME}, an executive assistant for {org}, running entirely on this HP ZGX Nano with no cloud.",
             "I can answer questions about " + ", ".join(areas[:-1]) + f", and {areas[-1]}.",
             "Every number I give comes from deterministic analytics on synthetic company data, never from the language model.",
             "I do not answer general knowledge, arithmetic or anything outside this company's data; I suggest a business question instead.",
             "Hold the blue button and ask one of those questions."]
    return result(tool="capabilities", title="What I can answer",
                  headline={"label": "What I can answer", "value": "Ask me about…", "fmt": "text",
                            "sub": "; ".join(a[0].upper() + a[1:] for a in areas[:4])},
                  values={"areas": areas}, facts=facts,
                  method={"summary": "No calculation needed: this is a description of the available analytics tools.",
                          "formula": "n/a", "steps": [a[0].upper() + a[1:] for a in areas]},
                  inputs=[], sources=["tool registry"], chart={"type": "none", "title": "Try asking", "labels": [], "series": []},
                  template=" ".join(facts[:2]) + " " + facts[4])
