"""Developer mode: Atlas writes code against the synthetic data, runs it in the sandbox, and narrates only what
the run actually produced. Code that has not run is labelled "not executed". Errors are shown verbatim and
Atlas gets exactly one attempt to fix and rerun."""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import sqlite3
import time
from typing import AsyncIterator

import httpx
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, SqlLexer

from atlas import sandbox
from atlas.analytics.common import item, money, result
from atlas.config import AS_OF, DB_PATH, LLM_URL, ROOT

from atlas.analytics import TOOLS as _TOOLS

DEV_TOOLS = {k: v[1] for k, v in _TOOLS["developer"].items()}

RULES = [
    ("model_status", r"how fast|tokens? per second|tok/s|connect (my|your|our) (own )?tools|point (my|your) (editor|agent)|endpoint.*(model|editor|agent)|model.*running"),
    ("unit_tests", r"unit tests?|\btests? for\b|write tests"),
    ("rest_endpoint", r"\brest\b|endpoint|\bapi\b"),
    ("explain_code", r"\bexplain\b|how is .* (calculated|computed)|how does .* work|show me the code"),
    ("plot", r"\bplot|\bchart|\bgraph|visuali[sz]"),
    ("python_script", r"python|script|\bcsv\b|export"),
    ("sql_query", r"\bsql\b|\bquery\b|\bselect\b|top (ten|\d+)|deals? whose|\bfind\b|\blist\b|how many"),
]


def route(text: str) -> str:
    t = text.lower()
    for tool, pat in RULES:
        if re.search(pat, t):
            return tool
    return "capabilities"


# ----------------------------------------------------------------------------- schema context for the model

def schema_text() -> str:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    keep = ["accounts", "open_pipeline", "deals_history", "reps", "revenue_quarterly", "plan_targets",
            "pipeline_snapshots", "costs_quarterly", "site_candidates"]
    lines = []
    for t in keep:
        cols = [r[1] + " " + r[2] for r in con.execute(f"PRAGMA table_info({t})")]
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        lines.append(f"{t}({', '.join(cols)})  -- {n} rows")
    con.close()
    return "\n".join(lines)


NOTES = (f"Notes: today is {AS_OF}. The current quarter is Q4 2026 (2026-10-01 to 2026-12-31). "
         "open_pipeline holds every open deal; stage is one of Prospect, Qualify, Propose, Commit; close_date is the "
         "expected close date (ISO text). deals_history holds closed deals; outcome is 'won' or 'lost'; max_stage is the "
         "furthest stage reached; closed_date is when it closed. Open pipeline of an account = SUM(open_pipeline.amount). "
         "Stage order is Prospect < Qualify < Propose < Commit, and every won deal reached Commit, so the win rate from a stage "
         "= won deals ÷ closed deals whose max_stage is that stage or later. Amounts are in US dollars. plan_targets has the "
         "annual new_bookings_plan per region and year. Pipeline coverage for a region = SUM(open_pipeline.amount) for deals "
         "closing in Q4 2026 (close_date between '2026-10-01' and '2026-12-31') ÷ (that region's 2026 new_bookings_plan ÷ 4), "
         "reported as a multiple such as 3.1x. Never compute win rate with GROUP BY max_stage: lost deals stop at the stage "
         "where they died and won deals always reach Commit, so grouping gives 0% for early stages. Compute each stage with "
         "its own cumulative filter instead. Example, win rate from Propose: SELECT AVG(outcome = 'won') FROM deals_history "
         "WHERE max_stage IN ('Propose', 'Commit').")

PY_ENV = ("The script runs in a sandbox with no network. The working directory contains atlas.db (SQLite, read-only): "
          "open it with sqlite3.connect('file:atlas.db?mode=ro', uri=True). Available: sqlite3, csv, json, pandas, "
          "matplotlib (save charts to chart.png; use color '#004BDF'; plot with plt.bar(labels, values) or plt.plot(labels, values) "
          "using string labels and never call set_xticks or set_xticklabels; plt.tight_layout()). Atlas's own analytics are "
          "importable: `from atlas.analytics.enterprise import forecast_by_region` returns a list of dicts "
          "{region, p10, p50, p90}: the Q4 2026 revenue Monte Carlo by region in US dollars. Write output files to "
          "the current directory. Finish by printing the key result data (for example the rows you wrote or plotted) as one "
          "JSON list of records on the last line: print(json.dumps(records)).")

TESTABLE = {
    "site scor": ("atlas.analytics.common", "score_options"),
    "scor": ("atlas.analytics.common", "score_options"),
    "money": ("atlas.analytics.common", "money"),
    "format": ("atlas.analytics.common", "money"),
    "grounding": ("atlas.grounding", "check"),
    "router": ("atlas.router", "route"),
    "forecast by region": ("atlas.analytics.enterprise", "forecast_by_region"),
}

EXPLAIN = [
    (r"forecast|revenue|q4|profit", [("atlas.analytics.enterprise", "_simulate"), ("atlas.analytics.enterprise", "profit_from_revenue")],
     r"conv\[|slip|rng|random|wins = |recurring = |s = recurring|total \+= |gross_margin|opex"),
    (r"risk", [("atlas.analytics.enterprise", "risk_model")], r"fit|logistic|coef|auc|features|predict"),
    (r"coverage", [("atlas.analytics.enterprise", "_coverage_at")], r"plan|share|target|won|pipe|remaining|coverage"),
    (r"site|scor", [("atlas.analytics.common", "score_options")], r"norm|contrib|scores|dirichlet|wins"),
    (r"ground|verif|check", [("atlas.grounding", "check")], r"pool|matched|_close|unmatched|NUM_RE"),
]

FMT_LIGHT = HtmlFormatter(nowrap=True)


def hl(code: str, lang: str, hl_lines: list[int] | None = None, start: int = 1) -> str:
    lexer = SqlLexer() if lang == "sql" else PythonLexer()
    fmt = HtmlFormatter(nowrap=False, linenos="inline", linenostart=start, hl_lines=hl_lines or [], cssclass="hl")
    return highlight(code, lexer, fmt)


def extract_code(text: str, lang: str) -> str:
    m = re.search(r"```(?:" + lang + r"|sqlite|py|python3)?\s*\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n?(.*?)(```|$)", text, re.S)
    return (m.group(1) if m else text).strip()


# ----------------------------------------------------------------------------- generation

async def generate(llm, system: str, user: str, lang: str, max_tokens: int = 700, on_delta=None) -> tuple[str, dict]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = await llm.template(msgs)
    t = time.perf_counter()
    out = ""
    n = 0
    async for tok in llm.complete_stream(prompt + f"```{lang}\n", max_tokens, temperature=0.0):  # deterministic code
        out += tok
        n += 1
        if on_delta and n % 6 == 0:
            await on_delta(out)
        if "```" in out:
            break
    ms = (time.perf_counter() - t) * 1000
    code = out.split("```")[0].strip()
    return code, {"ms": ms, "tokens": n, "tok_s": n / max(ms / 1000, 1e-3)}


def sys_prompt(kind: str) -> str:
    base = "You are a senior engineer writing code for Meridian Dynamics (all data synthetic). "
    if kind == "sql":
        return base + ("Write one correct SQLite query that answers the request. Use only the tables and columns in the "
                       "schema. Give computed columns short snake_case aliases. Order results meaningfully and LIMIT to at "
                       "most 25 rows unless asked. Reply with the SQL only.")
    if kind == "tests":
        return base + ("Write exactly 5 short, focused pytest unit test functions for the given function. Import it from its "
                       "module. Use small hand-made inputs and assert only properties you can verify by reading the source: "
                       "return shapes and lengths, value ranges, which option ranks first when it is best on every criterion, "
                       "shares that sum to 1 (use pytest.approx). Never hard-code exact floating-point results. No network, no files, no fixtures from conftest. Reply with the Python test file only.")
    if kind == "endpoint":
        return base + ("Write one Python file: a FastAPI app exposing the requested GET endpoint computed from atlas.db, and "
                       "under if __name__ == '__main__': test it with fastapi.testclient.TestClient, assert the status is "
                       "200, and print the JSON response with json.dumps(data, indent=1). Never start a server (no uvicorn.run): "
                       "the sandbox has no network, so TestClient is the only way to call the endpoint. " + PY_ENV + " Reply with the code only.")
    return base + "Write one short, correct Python 3 script for the request. " + PY_ENV + " Reply with the code only."


def user_prompt(question: str, extra: str = "") -> str:
    return f"SCHEMA:\n{schema_text()}\n{NOTES}\n{extra}\nREQUEST: {question}"


# ----------------------------------------------------------------------------- results → grounded facts

def table_facts(table: dict | None, limit: int = 10) -> list[str]:
    if not table or not table.get("rows"):
        return ["The query returned no rows."] if table is not None else []
    cols = table["columns"]
    facts = [f"The run returned {table.get('row_count', len(table['rows']))} rows with columns {', '.join(cols)}."]
    for i, r in enumerate(table["rows"][:limit], 1):
        cells = []
        for c, v in zip(cols, r):
            cells.append(f"{c} {_fmt_cell(c, v)}")
        facts.append(f"Row {i}: " + ", ".join(cells) + ".")
    return facts


def _fmt_cell(col: str, v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    c = col.lower()
    if any(k in c for k in ("amount", "pipeline", "revenue", "p10", "p50", "p90", "value", "total", "sum", "usd", "dollar")) and abs(x) >= 1000:
        return f"{money(x)} ({x:,.0f})"
    if any(k in c for k in ("rate", "pct", "share", "ratio")) and abs(x) <= 1.5:
        return f"{x * 100:.1f}% ({x:.4f})"
    if x == int(x):
        return f"{int(x):,}"
    return f"{x:,.4g}"


def _dev_result(tool: str, title: str, headline: dict, facts: list[str], method: dict, sources: list[str],
                template: str, table: dict | None = None, inputs: list | None = None) -> dict:
    r = result(tool=tool, title=title, headline=headline, values={}, facts=facts, method=method, inputs=inputs or [],
               sources=sources, chart={"type": "none", "title": title, "labels": [], "series": []}, template=template,
               table=table)
    return r


# ----------------------------------------------------------------------------- live model status

async def model_status(host_hint: str = "127.0.0.1:8100") -> dict:
    async with httpx.AsyncClient(base_url=LLM_URL, timeout=10) as c:
        props = (await c.get("/props")).json()
        slots = (await c.get("/slots")).json()
        busy = sum(1 for s in slots if s.get("is_processing"))
        t = time.perf_counter()
        r = (await c.post("/completion", json={"prompt": "List the four seasons of the year, one per line:\n",
                                                 "n_predict": 64, "temperature": 0, "cache_prompt": False})).json()
        wall = time.perf_counter() - t
    tim = r.get("timings", {})
    tps = float(tim.get("predicted_per_second") or 0)
    n_ctx = int(props.get("default_generation_settings", {}).get("n_ctx") or slots[0].get("n_ctx", 0))
    model = (props.get("model_alias") or "nemotron-3.5-lightning")
    return {"tok_s": tps, "prompt_tok_s": float(tim.get("prompt_per_second") or 0), "n_predicted": int(tim.get("predicted_n") or 0),
            "n_ctx": n_ctx, "slots": len(slots), "busy": busy, "model": model, "wall_ms": wall * 1000,
            "base_url": f"http://{host_hint}/v1"}


# ----------------------------------------------------------------------------- the answer pipeline

async def answer(question: str, llm, send, host_hint: str = "127.0.0.1:8100") -> AsyncIterator[dict]:
    """Yields events like brain.answer (route, sentence, result) plus code/run events sent directly via `send`."""
    from atlas.analytics import TOOLS
    timings: dict = {}
    t0 = time.perf_counter()
    tool, by = route(question), "router"
    if llm.ok:
        try:
            l_tool, _, meta = await llm.pick_tool(question, "developer")
            timings["llm_route_ms"] = meta["ms"]
            if l_tool in DEV_TOOLS:
                tool, by = l_tool, "llm"
                # The deterministic router wins on explicit verbs the small model can blur ("show me the query").
                r_tool = route(question)
                if r_tool != "capabilities" and r_tool != tool:
                    tool, by = r_tool, "router"
        except Exception as e:  # noqa: BLE001
            timings["llm_route_error"] = str(e)[:80]
    timings["route_ms"] = (time.perf_counter() - t0) * 1000
    yield {"type": "route", "tool": tool, "args": {}, "by": by, "router_tool": route(question), "agree": True,
           "ms": timings.get("llm_route_ms", timings["route_ms"])}

    ack = {"sql_query": "Writing the SQL now, then I'll run it in the sandbox.",
           "python_script": "Writing the script now, then I'll run it in the sandbox.",
           "plot": "Writing the plotting script now, then I'll run it in the sandbox.",
           "rest_endpoint": "Building the endpoint now, then I'll test it in the sandbox.",
           "unit_tests": "Reading the function and writing tests, then I'll run them in the sandbox.",
           "explain_code": "Opening the real source code from my own repository.",
           "model_status": "Measuring the model on this box right now."}.get(tool)
    spoken = []
    if ack and tool != "model_status" and not llm.ok:
        ack = None  # don't announce "writing the code" when the model is known to be down
    if ack:
        spoken.append(ack)
        yield {"type": "sentence", "text": ack, "source": "status"}

    t1 = time.perf_counter()
    try:
        if tool in ("sql_query", "python_script", "plot", "rest_endpoint", "unit_tests"):
            res = await _code_task(tool, question, llm, send, timings)
        elif tool == "explain_code":
            res = await _explain(question, send)
        elif tool == "model_status":
            res = await _status(send, host_hint)
        else:
            from atlas.analytics.common import capabilities
            res = capabilities("developer")
    except Exception as e:  # noqa: BLE001
        msg = (str(e)[:200] or type(e).__name__).rstrip(". ")
        if "connect" in msg.lower() or "refused" in msg.lower():
            msg = "The local language model is not reachable right now"
        res = _dev_result(tool, "Could not complete", {"label": "Not executed", "value": "Not executed", "fmt": "text"},
                          [f"I could not finish this: {msg}", "Nothing was executed, so there is no output to report."],
                          {"summary": msg, "formula": "n/a", "steps": []}, [], f"I could not finish this: {msg}. Nothing was executed.")
    timings["tool_ms"] = (time.perf_counter() - t1) * 1000
    yield {"type": "result", "result": res, "ms": timings["tool_ms"]}
    yield {"type": "spoken_prefix", "sentences": spoken}


async def _code_task(tool: str, question: str, llm, send, timings: dict) -> dict:
    lang = "sql" if tool == "sql_query" else "python"
    kind = {"sql_query": "sql", "unit_tests": "tests", "rest_endpoint": "endpoint"}.get(tool, "python")
    extra, files, entry, target = "", None, None, None
    if tool == "unit_tests":
        mod, fn = _pick_testable(question)
        import importlib
        src = inspect.getsource(getattr(importlib.import_module(mod), fn))
        extra = f"FUNCTION UNDER TEST (import with: from {mod} import {fn}):\n{src}"
        entry = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "main.py"]
        target = f"{mod}.{fn}"
    user = user_prompt(question, extra)
    title = {"sql_query": "SQL", "python_script": "PYTHON", "plot": "PYTHON · PLOT", "rest_endpoint": "PYTHON · REST ENDPOINT",
             "unit_tests": f"PYTEST · {target}"}[tool]

    async def delta(code_so_far):
        await send({"type": "code", "lang": lang, "code": code_so_far, "title": title, "status": "writing", "attempt": 1})

    if not llm.ok:
        raise RuntimeError("The language model is offline, so I can't write code right now")
    budget = 1400 if tool in ("unit_tests", "rest_endpoint") else 800
    code, gen = await generate(llm, sys_prompt(kind), user, lang, max_tokens=budget, on_delta=delta)
    timings["gen_ms"] = gen["ms"]
    await send({"type": "code", "lang": lang, "code": code, "html": hl(code, lang), "title": title,
                "status": "not executed", "attempt": 1, "gen": gen})
    run = await asyncio.to_thread(sandbox.run, lang, code, files, entry)
    await send({"type": "run", **_run_public(run), "attempt": 1})
    attempts = [run]
    if not run["ok"]:
        err = (run["stderr"] or run["stdout"])[-1800:]
        if run["timed_out"]:
            err += ("\nIt was killed at the 10-second sandbox limit, so something blocked: the sandbox has no network, so "
                    "never start a server or wait on a socket; keep the work small.")
        fix_user = user + f"\n\nYOUR PREVIOUS CODE:\n```{lang}\n{code}\n```\nIT FAILED IN THE SANDBOX WITH:\n{err}\nFix it."
        code2, gen2 = await generate(llm, sys_prompt(kind), fix_user, lang, max_tokens=budget)
        code = code2
        await send({"type": "code", "lang": lang, "code": code, "html": hl(code, lang), "title": title,
                    "status": "not executed", "attempt": 2, "gen": gen2})
        run = await asyncio.to_thread(sandbox.run, lang, code, files, entry)
        await send({"type": "run", **_run_public(run), "attempt": 2})
        attempts.append(run)
    return _code_result(tool, question, lang, code, run, attempts, target)


def _pick_testable(q: str) -> tuple[str, str]:
    t = q.lower()
    for k, v in TESTABLE.items():
        if k in t:
            return v
    return TESTABLE["site scor"]


def _run_public(run: dict) -> dict:
    return {k: run[k] for k in ("executed", "ok", "exit_code", "timed_out", "ms", "stdout", "stderr", "table", "images",
                                "files", "run_id", "method", "lang")}


def _code_result(tool, question, lang, code, run, attempts, target) -> dict:
    n_att = len(attempts)
    status = ("ran successfully" if run["ok"] else ("timed out" if run["timed_out"] else f"failed with exit code {run['exit_code']}"))
    facts = [f"The code ran in the sandbox and {status} in {run['ms']:.0f} milliseconds" +
             (" after one automatic fix." if n_att == 2 and run["ok"] else ".")]
    if n_att == 2 and not attempts[0]["ok"]:
        first_err = (attempts[0]["stderr"].strip().splitlines() or ["error"])[-1][:160]
        facts.append(f"The first attempt failed with: {first_err}")
    table = run.get("table")
    if tool == "unit_tests":
        m = re.search(r"(\d+) passed", run["stdout"]); f = re.search(r"(\d+) failed", run["stdout"])
        passed, failed = int(m.group(1)) if m else 0, int(f.group(1)) if f else 0
        facts.append(f"pytest reported {passed} passed and {failed} failed for {target}.")
        headline = {"label": f"Unit tests · {target}", "value": f"{passed} passed · {failed} failed", "fmt": "text"}
        table = {"columns": ["result", "count"], "rows": [["passed", passed], ["failed", failed]]}
    elif not run["ok"]:
        last = (run["stderr"].strip().splitlines() or ["no output"])[-1][:200]
        facts.append(f"The error was: {last}")
        headline = {"label": "Sandbox run", "value": "Failed", "fmt": "text"}
    else:
        facts += table_facts(table)
        if run["images"]:
            facts.append(f"It produced the chart file {run['images'][0]['name']}.")
        csvs = [f["name"] for f in run["files"] if f["name"].endswith(".csv")]
        if csvs:
            facts.append(f"It wrote {csvs[0]} with {table.get('row_count', len(table['rows'])) if table else 0} data rows.")
        out = run["stdout"].strip()
        if out and lang == "python":
            facts.append("The program printed: " + " ".join(out.split())[:400])
        n = (table or {}).get("row_count", len((table or {}).get("rows", [])))
        headline = {"label": "Executed in sandbox", "value": f"{n} rows" if table else "Ran OK", "fmt": "text"}
    template = " ".join(facts[:3])
    return _dev_result(tool, f"{tool.replace('_', ' ')}", headline, facts,
                       {"summary": "Generated by the local LLM, executed in the sandbox; the answer only uses the real output.",
                        "formula": sandbox.METHOD, "steps": [f"attempt {i + 1}: exit {a['exit_code']}, {a['ms']:.0f} ms" for i, a in enumerate(attempts)]},
                       ["atlas.db (read-only copy)", f"sandbox run {run['run_id']}"], template, table=table)


async def _explain(question: str, send) -> dict:
    import importlib
    t = question.lower()
    spec = next((e for e in EXPLAIN if re.search(e[0], t)), EXPLAIN[0])
    parts = []
    for mod, fn in spec[1]:
        obj = getattr(importlib.import_module(mod), fn)
        lines, start = inspect.getsourcelines(obj)
        code = "".join(lines)
        key = [i + 1 for i, ln in enumerate(lines) if re.search(spec[2], ln) and not ln.strip().startswith(("#", '"""'))][:14]
        path = inspect.getsourcefile(obj).replace(str(ROOT) + "/", "")
        parts.append({"module": mod, "fn": fn, "path": path, "start": start, "end": start + len(lines) - 1, "code": code,
                      "key_lines": [start + k - 1 for k in key], "html": hl(code, "python", key, start)})
    for i, p in enumerate(parts):
        await send({"type": "code", "lang": "python", "code": p["code"], "html": p["html"], "status": "source",
                    "title": f"{p['path']} · {p['fn']}() · lines {p['start']}–{p['end']}", "attempt": 1, "part": i,
                    "parts": len(parts), "read_only": True})
    facts = [f"The code is {p['fn']} in {p['path']}, lines {p['start']} to {p['end']}." for p in parts]
    if spec is EXPLAIN[0]:
        from atlas.analytics import enterprise as E
        from atlas.config import MC_SIMS
        fq = E.forecast_quarter()
        facts += [fq["method"]["summary"], "Formula: " + fq["method"]["formula"],
                  f"It runs {MC_SIMS:,} Monte Carlo simulations with a fixed random seed, so the answer is reproducible."]
        facts += [f for f in fq["facts"][:3]]
        facts += fq["method"]["steps"][:2]
        res = _dev_result("explain_code", "How the Q4 forecast is calculated", {"label": "Source", "value": f"{parts[0]['fn']}()", "fmt": "text",
                          "sub": parts[0]["path"]}, facts, fq["method"], [p["path"] for p in parts], " ".join(facts[:3]))
        res["percentiles"] = fq.get("percentiles")
        return res
    doc = inspect.getdoc(getattr(importlib.import_module(spec[1][0][0]), spec[1][0][1])) or ""
    facts.append(doc.split("\n\n")[0].replace("\n", " "))
    return _dev_result("explain_code", "Code explanation", {"label": "Source", "value": f"{parts[0]['fn']}()", "fmt": "text",
                       "sub": parts[0]["path"]}, facts, {"summary": doc[:300], "formula": "see source", "steps": []},
                       [p["path"] for p in parts], " ".join(facts[:2]))


async def _status(send, host_hint: str) -> dict:
    st = await model_status(host_hint)
    await send({"type": "endpoint", **st})
    facts = [f"The local model is generating {st['tok_s']:.1f} tokens per second right now, measured on a {st['n_predicted']}-token test just now.",
             f"Prompt processing runs at {st['prompt_tok_s']:.0f} tokens per second.",
             f"Each request gets a context window of {st['n_ctx']:,} tokens, with {st['slots']} parallel slots; {st['busy']} were busy at the moment of measurement.",
             f"Point your editor or coding agent at {st['base_url']} with the model name {st['model']}; it speaks the standard chat completions API.",
             "The endpoint is served from this box and nothing leaves it."]
    curl = (f"curl {st['base_url']}/chat/completions -H 'Content-Type: application/json' "
            f"-d '{{\"model\": \"{st['model']}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}]}}'")
    await send({"type": "code", "lang": "bash", "code": curl, "html": highlight(curl, __import__("pygments.lexers", fromlist=["BashLexer"]).BashLexer(),
                HtmlFormatter(cssclass="hl")), "title": "CONNECT YOUR OWN TOOLS · curl example", "status": "example", "attempt": 1})
    return _dev_result("model_status", "Local model status", {"label": "Generation speed, measured now", "value": f"{st['tok_s']:.1f} tok/s", "fmt": "text",
                       "sub": f"context {st['n_ctx']:,} · {st['busy']}/{st['slots']} slots busy"}, facts,
                       {"summary": "A 64-token generation timed by the model server itself, plus its live slot table.", "formula": "tokens ÷ generation seconds", "steps": []},
                       ["model server /props, /slots, /completion timings"], " ".join(facts[:2]),
                       inputs=[item("Generation speed", round(st["tok_s"], 1), "dec"), item("Context per slot", st["n_ctx"], "num")])
