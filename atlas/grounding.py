"""Grounding check: every number in an answer must match a number produced by a tool."""
from __future__ import annotations

import re

NUM_RE = re.compile(
    r"(?P<neg>[-−])?(?P<cur>\$)?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<suf>billion|million|thousand|bn|[BMK](?![a-z])|%|x(?![a-z])|percent(?:age points?)?|points?|pts)?",
    re.IGNORECASE)

# Tokens that are not claims: years, quarter/fiscal labels, list numbering.
IGNORE_RE = re.compile(r"\b(?:Q[1-4]|FY\s?\d{2,4}|(?:19|20)\d{2})\b", re.IGNORECASE)


def _collect(obj, out: list[float]):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out)
    elif isinstance(obj, str):
        # Pre-formatted strings in facts/template/table also count as tool output.
        for m in NUM_RE.finditer(IGNORE_RE.sub(" ", obj)):
            v = _value(m)
            if v is not None:
                out.extend(v)


def _value(m) -> list[float] | None:
    raw = m.group("num").replace(",", "")
    try:
        x = float(raw)
    except ValueError:
        return None
    suf = (m.group("suf") or "").lower()
    mult = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}.get(suf, 1)
    x *= mult
    if m.group("neg"):
        x = -x
    if suf in ("%", "percent") or suf.startswith("percent") or suf in ("points", "point", "pts"):
        return [x / 100, x]
    return [x]


GROUND_KEYS = ("facts", "template", "headline", "inputs", "percentiles", "table")


def tool_numbers(result: dict) -> list[float]:
    """Numbers the narrator is allowed to say: what the tool put in front of it, nothing nested deeper."""
    out: list[float] = []
    for k in GROUND_KEYS:
        if k in result:
            _collect(result[k], out)
    return out


def check(text: str, result: dict | list[dict]) -> dict:
    """Return {ok, numbers:[{text, value, matched}], unmatched:[...]}."""
    results = result if isinstance(result, list) else [result]
    pool: list[float] = []
    for r in results:
        pool.extend(tool_numbers(r))
    pool_abs = [abs(p) for p in pool]
    cleaned = IGNORE_RE.sub(" ", text)
    found = []
    for m in NUM_RE.finditer(cleaned):
        vals = _value(m)
        if vals is None:
            continue
        token = m.group(0).strip()
        # small bare integers ("five deals", "top 3") are counts the tools also return; still check them
        matched = any(_close(abs(v), p) for v in vals for p in pool_abs)
        found.append({"text": token, "value": vals[0], "matched": matched})
    unmatched = [f["text"] for f in found if not f["matched"]]
    return {"ok": not unmatched, "numbers": found, "unmatched": unmatched}


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    if b == 0:
        return abs(a) < 1e-9
    rel = abs(a - b) / abs(b)
    # allow display rounding: $644.7M vs 644,707,194 (0.01%), 13.3% vs 0.1334, $0.64B vs 644.7M (0.7%)
    return rel <= 0.0075
