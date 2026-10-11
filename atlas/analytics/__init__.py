"""Tool registry: every analytics tool the brain can call, with small JSON schemas for the LLM."""
from __future__ import annotations

from atlas.analytics import enterprise as E
from atlas.analytics.common import capabilities
from atlas.analytics import university as U

TOOLS = {
    "enterprise": {
        "forecast_quarter": (E.forecast_quarter, "Current quarter (Q4) forecast: revenue, profit, margins, downside/upside range (P10/P50/P90), growth versus last year, last quarter actuals, open deal count, largest open deal, weighted pipeline.", {}),
        "deal_risk": (E.deal_risk, "Which specific deals are most at risk of being lost or slipping this quarter, and why.",
                      {"n": {"type": "integer", "description": "how many deals (default 5)"}}),
        "score_sites": (E.score_sites, "Compare candidate cities for a new sales center (decision brief).",
                        {"cities": {"type": "array", "items": {"type": "string"},
                                    "description": "cities named by the user, e.g. Phoenix, Denver, Albuquerque, Austin, Salt Lake City"}}),
        "what_if": (E.what_if, "Only for hypothetical scenarios that change a pipeline stage win rate (e.g. Commit drops ten points).",
                    {"stage": {"type": "string", "enum": ["Prospect", "Qualify", "Propose", "Commit"]},
                     "delta_pts": {"type": "number", "description": "change in win rate in percentage points, e.g. -10"}}),
        "region_vs_plan": (E.region_vs_plan, "Revenue by region versus annual plan (West, Central, East, International): which is behind or ahead, and by how much.", {}),
        "hiring_plan": (E.hiring_plan, "How many sales reps to hire to hit next year's target.", {}),
        "pipeline_coverage": (E.pipeline_coverage, "Pipeline coverage (open pipeline vs bookings still needed) now versus the same day last year.", {}),
        "board_summary": (E.board_summary, "Only when asked for a board summary, overview or briefing: forecast, plan gap, coverage, risk.", {}),
        "capabilities": (lambda: capabilities("enterprise"), "Anything else: greetings, who are you, off-topic or unanswerable questions.", {}),
    },
    "university": {
        "grant_forecast": (U.grant_forecast, "Total projected research funding for FY27 from awards and the grant pipeline, proposal success rates, versus last year, and by college.", {}),
        "sponsor_concentration": (U.sponsor_concentration, "Funding by sponsor (NIH, NSF, DOE, DoD, industry, foundations): amounts, shares, dependence and cut risk.", {}),
        "score_research_sites": (U.score_research_sites, "Where to open the next research center (decision brief).", {}),
        "enrollment_forecast": (U.enrollment_forecast, "Fall 2027 enrollment: yield, new students, admits, by segment, versus target.", {}),
        "board_summary_university": (U.board_summary_university, "Only when asked for a board summary, overview or briefing for the university.", {}),
        "capabilities": (lambda: capabilities("university"), "Anything else: greetings, who are you, off-topic or unanswerable questions.", {}),
    },
}

# Second brain: document search with citations (synthetic corpus, local embeddings). Switch off with ATLAS_DOCS=0.
from atlas.config import FEATURES as _F

if _F["docs"]:
    def _sk(persona):
        def run(question: str = ""):
            from atlas.knowledge import docs
            return docs.search_knowledge(persona, question)
        return run
    for _p, _desc in (("enterprise", "What a document says: board memos, QBR notes, account plans, pricing and discount policy, hiring plan memo, site memo (cites passages)."),
                      ("university", "What a document says: NIH portfolio review and PI track records, grant abstracts, sponsor guidelines such as cost sharing, enrollment strategy, board memos (cites passages).")):
        _cap = TOOLS[_p].pop("capabilities")
        TOOLS[_p]["search_knowledge"] = (_sk(_p), _desc, {"question": {"type": "string"}})
        TOOLS[_p]["capabilities"] = _cap

# Developer persona: the brain picks one of these; atlas/dev.py runs them (code generation + sandbox), not run_tool.
TOOLS["developer"] = {
    "sql_query": (None, "Write a SQL query against the company database and run it (top accounts, overdue deals, any data question asking for a query).", {}),
    "python_script": (None, "Write a Python script (e.g. export data or the forecast to a CSV file) and run it.", {}),
    "plot": (None, "Plot or chart data from the database with a Python script and run it.", {}),
    "rest_endpoint": (None, "Build a small REST API endpoint and test it.", {}),
    "unit_tests": (None, "Write unit tests for a named Atlas function and run them.", {}),
    "explain_code": (None, "Explain how Atlas calculates something and show its real source code.", {}),
    "model_status": (None, "How fast the local model is running, context size, current load, and how to connect your own tools to it.", {}),
    "capabilities": (None, "Anything else: greetings, who are you, off-topic.", {}),
}

def tool_schemas(persona: str) -> list[dict]:
    out = []
    for name, (_, desc, props) in TOOLS[persona].items():
        out.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": []}}})
    return out


def run_tool(persona: str, name: str, args: dict | None = None) -> dict:
    fn, _, props = TOOLS[persona][name]
    clean = {k: v for k, v in (args or {}).items() if k in props and v not in (None, "", [])}
    return fn(**clean)


def find_persona(tool: str) -> str | None:
    for p, tools in TOOLS.items():
        if tool in tools:
            return p
    return None
