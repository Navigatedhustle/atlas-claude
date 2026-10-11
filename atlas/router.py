"""Deterministic intent router. Always on: used when the LLM is unavailable or picks badly."""
from __future__ import annotations

import re

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "fifteen": 15, "twenty": 20, "twenty five": 25, "thirty": 30}
CITIES = ["Phoenix", "Denver", "Albuquerque", "Austin", "Salt Lake City"]

RULES = {
    "enterprise": [
        ("board_summary", [r"\bboard\b", r"\bsummar", r"\bbrief me\b", r"thirty second", r"30 second"]),
        ("what_if", [r"what happens", r"what if", r"\bdrops?\b", r"\bfalls?\b", r"win rate", r"scenario"]),
        ("deal_risk", [r"\brisk", r"at risk", r"\bdeals?\b", r"slip"]),
        ("score_sites", [r"sales cent", r"\bbuild\b", r"\bsite", r"\bcity\b", r"phoenix", r"denver", r"albuquerque", r"austin", r"salt lake", r"\boffice\b", r"\blocation"]),
        ("region_vs_plan", [r"\bregion", r"behind plan", r"\bplan\b", r"\bbehind\b"]),
        ("hiring_plan", [r"\bhire", r"\breps?\b", r"headcount", r"hiring", r"sales ?people"]),
        ("pipeline_coverage", [r"coverage", r"pipeline", r"last year", r"this time"]),
        ("forecast_quarter", [r"revenue", r"profit", r"forecast", r"project", r"q ?4", r"q for", r"quarter", r"\bnumber"]),
    ],
    "university": [
        ("board_summary_university", [r"\bboard\b", r"\bsummar"]),
        ("sponsor_concentration", [r"sponsor", r"depend", r"concentrat", r"\bnih\b", r"\bnsf\b"]),
        ("score_research_sites", [r"research cent", r"\bopen\b.*\bcent", r"\bsite", r"\bwhere\b", r"\bcampus\b", r"\blocation"]),
        ("enrollment_forecast", [r"enrol", r"yield", r"admit", r"students?", r"\bfall\b", r"freshm"]),
        ("grant_forecast", [r"grant", r"research funding", r"funding", r"fy ?27", r"fy ?2027", r"award", r"pipeline"]),
    ],
}

# Strong phrases win outright (ordered).
DOCS_STRONG = {
    "enterprise": r"\bqbr\b|board'?s concerns|concerns about|\bmemo\b|pricing policy|discount approval|account plan|what did .* say|say about|said about",
    "university": r"track record|sponsor guidance|guidance|guideline|cost shar|strategy risks|enrollment strategy|abstract|\bmemo\b|what does .* say|say about|solicitation",
}

STRONG = {
    "enterprise": [("what_if", r"(what happens|what if).*(win rate|drop|fall)|win rate.*(drop|fall)"),
                   ("pipeline_coverage", r"coverage"),
                   ("board_summary", r"board"),
                   ("region_vs_plan", r"region.*(behind|plan)|behind plan|furthest behind"),
                   ("hiring_plan", r"(how many|need).*(reps|hire|sales ?people)|\bhire\b|hiring"),
                   ("score_sites", r"sales cent|phoenix|albuquerque|denver|salt lake|austin"),
                   ("deal_risk", r"at risk|riskiest|most risk")],
    "university": [("sponsor_concentration", r"sponsor|depend"),
                   ("score_research_sites", r"research cent"),
                   ("enrollment_forecast", r"enrol|yield"),
                   ("grant_forecast", r"grant|research funding")],
}


def route(text: str, persona: str) -> tuple[str, dict, float]:
    """Return (tool, args, confidence)."""
    t = " " + text.lower().replace("-", " ") + " "
    from atlas.config import FEATURES
    if FEATURES["docs"] and persona in DOCS_STRONG and re.search(DOCS_STRONG[persona], t):
        return "search_knowledge", {"question": text}, 0.95
    for tool, pat in STRONG[persona]:
        if re.search(pat, t):
            return tool, extract_args(tool, text), 0.9
    best, score = None, 0
    for tool, pats in RULES[persona]:
        s = sum(1 for p in pats if re.search(p, t))
        if s > score:
            best, score = tool, s
    if best is None:
        return "capabilities", {}, 0.2
    return best, extract_args(best, text), min(0.5 + 0.15 * score, 0.85)


def extract_args(tool: str, text: str) -> dict:
    t = text.lower().replace("-", " ")
    if tool == "score_sites":
        cities = [c for c in CITIES if c.lower() in t]
        return {"cities": cities} if len(cities) >= 2 else {}
    if tool == "what_if":
        stage = next((s for s in ["Prospect", "Qualify", "Propose", "Commit"] if s.lower() in t), "Commit")
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percentage\s*)?(?:points?|pts|%|percent)", t)
        val = float(m.group(1)) if m else None
        if val is None:
            for w, v in sorted(WORDNUM.items(), key=lambda kv: -len(kv[0])):
                if re.search(rf"\b{w}\b\s*(?:percentage\s*)?(points?|pts|percent)", t):
                    val = float(v)
                    break
        val = val if val is not None else 10.0
        sign = 1 if re.search(r"\b(rises?|increases?|improves?|goes up|up)\b", t) else -1
        return {"stage": stage, "delta_pts": sign * val}
    if tool == "deal_risk":
        m = re.search(r"\b(\d+)\b", t)
        n = int(m.group(1)) if m else next((v for w, v in WORDNUM.items() if re.search(rf"\b{w}\b", t) and v <= 10), 5)
        return {"n": n}
    return {}
