"""Second brain: a knowledge graph built live from the synthetic SQLite data, plus provenance (which nodes a
tool's answer actually drew on). Every count is computed from the data; nothing here is hand-entered.

Node ids are "<type>:<key>". The layout is computed once (numpy force layout, type-anchored clusters) and
cached in data/brain/graph_<persona>.json; documents added later are placed next to their neighbours."""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

from atlas.config import AS_OF, DATA_DIR, DB_PATH

BRAIN_DIR = DATA_DIR / "brain"
TODAY = dt.date.fromisoformat(AS_OF)
Q_END = dt.date(TODAY.year, 12, 31)

# Entity types per persona: (type, label, anchor angle in degrees, hub?)
TYPES = {
    "enterprise": [("region", "Regions", 0, True), ("quarter", "Quarters", 0, True), ("center", "Sales centers", 0, True),
                   ("industry", "Industries", 0, True), ("segment", "Segments", 0, True),
                   ("rep", "Reps", 300, False), ("account", "Accounts", 60, False), ("deal", "Open deals", 180, False),
                   ("document", "Documents", 240, False), ("passage", "Passages", 240, False)],
    "university": [("college", "Colleges", 0, True), ("sponsor", "Sponsors", 0, True), ("fy", "Fiscal years", 0, True),
                   ("site", "Research sites", 0, True), ("cohort", "Enrollment cohorts", 0, True),
                   ("pi", "PIs", 300, False), ("award", "Grants (awarded)", 60, False), ("proposal", "Proposals (pipeline)", 180, False),
                   ("document", "Documents", 240, False), ("passage", "Passages", 240, False)],
}


def _con():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _q(con, sql, *a):
    return con.execute(sql, a).fetchall()


# ----------------------------------------------------------------------------- build

def _enterprise(con):
    nodes, edges = {}, []

    def node(i, t, label, value, fmt="money"):
        nodes[i] = {"id": i, "type": t, "label": label, "value": float(value or 0), "fmt": fmt}

    for r in _q(con, "SELECT region, SUM(revenue) v FROM revenue_quarterly WHERE year=? GROUP BY 1", TODAY.year):
        node(f"region:{r['region']}", "region", r["region"], r["v"])
    for r in _q(con, "SELECT year, quarter, SUM(revenue) v FROM revenue_quarterly GROUP BY 1,2"):
        node(f"quarter:{r['year']}Q{r['quarter']}", "quarter", f"Q{r['quarter']} {r['year']}", r["v"])
    q4 = f"quarter:{TODAY.year}Q4"
    if q4 not in nodes:
        v = _q(con, "SELECT SUM(amount) v FROM open_pipeline WHERE close_date BETWEEN ? AND ?", TODAY.isoformat(), Q_END.isoformat())[0]["v"]
        node(q4, "quarter", f"Q4 {TODAY.year} (open)", v)
    for r in _q(con, "SELECT city, nearby_pipeline FROM site_candidates"):
        node(f"center:{r['city']}", "center", r["city"], r["nearby_pipeline"])
    for r in _q(con, "SELECT a.industry k, SUM(o.amount) v FROM accounts a LEFT JOIN open_pipeline o USING(account_id) GROUP BY 1"):
        node(f"industry:{r['k']}", "industry", r["k"], r["v"])
    for r in _q(con, "SELECT a.segment k, SUM(o.amount) v FROM accounts a LEFT JOIN open_pipeline o USING(account_id) GROUP BY 1"):
        node(f"segment:{r['k']}", "segment", r["k"], r["v"])
    for r in _q(con, "SELECT rep_id, name, region, annual_quota FROM reps"):
        node(f"rep:{r['rep_id']}", "rep", r["name"], r["annual_quota"])
        edges.append((f"rep:{r['rep_id']}", f"region:{r['region']}"))
    for r in _q(con, "SELECT a.account_id, a.name, a.region, a.industry, a.segment, COALESCE(SUM(o.amount),0) v "
                     "FROM accounts a LEFT JOIN open_pipeline o USING(account_id) GROUP BY a.account_id"):
        i = f"account:{r['account_id']}"
        node(i, "account", r["name"], r["v"])
        edges += [(i, f"region:{r['region']}"), (i, f"industry:{r['industry']}"), (i, f"segment:{r['segment']}")]
    for r in _q(con, "SELECT deal_id, account_id, account_name, rep_id, stage, amount, close_date FROM open_pipeline"):
        i = f"deal:{r['deal_id']}"
        cd = dt.date.fromisoformat(r["close_date"])
        node(i, "deal", f"{r['account_name']} · {r['stage']}", r["amount"])
        qn = f"quarter:{cd.year}Q{(cd.month - 1) // 3 + 1}"
        edges += [(i, f"account:{r['account_id']}"), (i, f"rep:{r['rep_id']}")]
        if qn in nodes:
            edges.append((i, qn))
    return nodes, edges


def _university(con):
    nodes, edges = {}, []

    def node(i, t, label, value, fmt="money"):
        nodes[i] = {"id": i, "type": t, "label": label, "value": float(value or 0), "fmt": fmt}

    for r in _q(con, "SELECT college, SUM(total_amount) v FROM grant_awards GROUP BY 1"):
        node(f"college:{r['college']}", "college", r["college"], r["v"])
    for r in _q(con, "SELECT sponsor, SUM(total_amount) v FROM grant_awards GROUP BY 1"):
        node(f"sponsor:{r['sponsor']}", "sponsor", r["sponsor"], r["v"])
    for r in _q(con, "SELECT start_fy, SUM(annual_amount) v FROM grant_awards GROUP BY 1"):
        node(f"fy:{r['start_fy']}", "fy", f"FY{str(r['start_fy'])[2:]}", r["v"])
    for r in _q(con, "SELECT site, aligned_pipeline, focus_colleges FROM research_sites"):
        node(f"site:{r['site']}", "site", r["site"], r["aligned_pipeline"])
        for c in (r["focus_colleges"] or "").split(","):
            c = c.strip()
            if f"college:{c}" in nodes:
                edges.append((f"site:{r['site']}", f"college:{c}"))
    for r in _q(con, "SELECT fall_year, segment, enrolled FROM enrollment_funnel"):
        node(f"cohort:{r['fall_year']}:{r['segment']}", "cohort", f"Fall {r['fall_year']} · {r['segment']}", r["enrolled"], "num")
    pis = {}
    for r in _q(con, "SELECT p.pi_id, p.name, p.college, COALESCE(SUM(a.total_amount),0) v FROM faculty_pis p "
                     "LEFT JOIN grant_awards a ON a.pi_name = p.name GROUP BY p.pi_id"):
        i = f"pi:{r['pi_id']}"
        pis[r["name"]] = i
        node(i, "pi", r["name"], r["v"])
        edges.append((i, f"college:{r['college']}"))
    for r in _q(con, "SELECT award_id, college, sponsor, pi_name, start_fy, total_amount FROM grant_awards"):
        i = f"award:{r['award_id']}"
        node(i, "award", f"{r['sponsor']} award · {r['pi_name']}", r["total_amount"])
        edges += [(i, f"sponsor:{r['sponsor']}"), (i, f"college:{r['college']}"), (i, f"fy:{r['start_fy']}")]
        if r["pi_name"] in pis:
            edges.append((i, pis[r["pi_name"]]))
    for r in _q(con, "SELECT proposal_id, college, sponsor, pi_name, stage, annual_amount FROM grant_pipeline"):
        i = f"proposal:{r['proposal_id']}"
        node(i, "proposal", f"{r['sponsor']} proposal · {r['stage']}", r["annual_amount"])
        edges += [(i, f"sponsor:{r['sponsor']}"), (i, f"college:{r['college']}")]
        if r["pi_name"] in pis:
            edges.append((i, pis[r["pi_name"]]))
    return nodes, edges


def layout(nodes: list[dict], edges: list[list[int]], persona: str, seed: int = 42, iters: int = 260) -> np.ndarray:
    """Force layout: edge springs + sampled repulsion + pull toward each type's anchor (clusters by type)."""
    rng = np.random.default_rng(seed)
    types = TYPES[persona]
    tinfo = {t[0]: t for t in types}
    n = len(nodes)
    anchors = np.zeros((n, 2))
    grav = np.zeros(n)
    hubs = [t for t in types if t[3]]
    per_type = Counter(nd["type"] for nd in nodes)
    seen = Counter()
    for i, nd in enumerate(nodes):
        t = tinfo[nd["type"]]
        if t[3]:
            k = hubs.index(t)
            a = 2 * math.pi * k / len(hubs) + 0.3
            j = seen[nd["type"]]
            seen[nd["type"]] += 1
            b = 2 * math.pi * j / max(per_type[nd["type"]], 1)
            rr = 0.035 + 0.006 * per_type[nd["type"]]
            anchors[i] = [0.30 * math.cos(a) + rr * math.cos(b), 0.30 * math.sin(a) + rr * math.sin(b)]
            grav[i] = 0.9
        else:
            a = math.radians(t[2])
            anchors[i] = [0.70 * math.cos(a), 0.70 * math.sin(a)]
            grav[i] = 0.11
    pos = anchors + rng.normal(0, 0.12, (n, 2))
    E = np.array(edges, dtype=np.int64).reshape(-1, 2)
    step = 0.03
    for _ in range(iters):
        disp = (anchors - pos) * grav[:, None]
        if len(E):
            d = pos[E[:, 1]] - pos[E[:, 0]]
            f = d * 0.025
            np.add.at(disp, E[:, 0], f)
            np.add.at(disp, E[:, 1], -f)
        idx = rng.integers(0, n, (n, 24))
        diff = pos[:, None, :] - pos[idx]
        d2 = (diff ** 2).sum(-1) + 1e-4
        disp += (diff / d2[..., None]).sum(1) * (2.2e-5 * n / 24)
        ln = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        pos += disp / ln * np.minimum(ln, step)
        step *= 0.992
    lo, hi = pos.min(0), pos.max(0)
    return (pos - lo) / np.where(hi - lo == 0, 1, hi - lo)


def build(persona: str, extra_nodes: list[dict] | None = None, extra_edges: list[tuple] | None = None) -> dict:
    t0 = time.perf_counter()
    con = _con()
    nodes, edges = (_enterprise if persona == "enterprise" else _university)(con)
    con.close()
    for nd in extra_nodes or []:
        nodes[nd["id"]] = nd
    edges += list(extra_edges or [])
    order = list(nodes)
    index = {k: i for i, k in enumerate(order)}
    E = sorted({(index[a], index[b]) for a, b in edges if a in index and b in index})
    nl = [nodes[k] for k in order]
    pos = layout(nl, E, persona)
    _sizes(nl)
    for nd, (x, y) in zip(nl, pos):
        nd["x"], nd["y"] = round(float(x), 4), round(float(y), 4)
    g = {"persona": persona, "nodes": nl, "edges": [list(e) for e in E], "built_at": time.time(),
         "build_ms": round((time.perf_counter() - t0) * 1000), "types": [{"type": t[0], "label": t[1]} for t in TYPES[persona]]}
    save(persona, g)
    return g


def _sizes(nl: list[dict]):
    """Node radius reflects the real value (dollars or volume), scaled within each type."""
    by = {}
    for nd in nl:
        by.setdefault(nd["type"], []).append(nd["value"])
    for nd in nl:
        mx = max(by[nd["type"]]) or 1
        hub = nd["type"] in ("region", "quarter", "center", "industry", "segment", "college", "sponsor", "fy", "site", "cohort", "document")
        nd["r"] = round((4.0 if hub else 1.2) + (9.0 if hub else 4.5) * math.sqrt(max(nd["value"], 0) / mx), 2)


def path(persona: str) -> Path:
    return BRAIN_DIR / f"graph_{persona}.json"


def save(persona: str, g: dict):
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path(persona).with_suffix(".tmp")
    tmp.write_text(json.dumps(g, separators=(",", ":")))
    tmp.replace(path(persona))


_CACHE: dict = {}


def load(persona: str) -> dict:
    p = path(persona)
    mt = p.stat().st_mtime if p.exists() else None
    c = _CACHE.get(persona)
    if c and c[0] == mt:
        return c[1]
    g = json.loads(p.read_text()) if p.exists() else build(persona)
    _CACHE[persona] = (path(persona).stat().st_mtime, g)
    return g


def add_nodes(persona: str, new_nodes: list[dict], new_edges: list[tuple]) -> dict:
    """Add nodes (e.g. an ingested document and its passages) next to their neighbours, without re-laying out."""
    g = load(persona)
    idx = {nd["id"]: i for i, nd in enumerate(g["nodes"])}
    rng = np.random.default_rng(len(g["nodes"]))
    for nd in new_nodes:
        if nd["id"] in idx:
            continue
        nbrs = [g["nodes"][idx[b]] for a, b in new_edges if a == nd["id"] and b in idx]
        if nbrs:
            x = float(np.mean([n["x"] for n in nbrs])) + rng.normal(0, 0.03)
            y = float(np.mean([n["y"] for n in nbrs])) + rng.normal(0, 0.03)
        else:
            x, y = 0.85 + rng.normal(0, 0.04), 0.12 + rng.normal(0, 0.04)
        nd.setdefault("r", 6.0 if nd["type"] == "document" else 2.2)
        nd["x"], nd["y"] = round(min(max(x, 0.0), 1.0), 4), round(min(max(y, 0.0), 1.0), 4)
        idx[nd["id"]] = len(g["nodes"])
        g["nodes"].append(nd)
    es = {tuple(e) for e in g["edges"]}
    for a, b in new_edges:
        if a in idx and b in idx:
            es.add((idx[a], idx[b]))
    g["edges"] = [list(e) for e in sorted(es)]
    g["built_at"] = time.time()
    save(persona, g)
    return g


def stats(g: dict) -> dict:
    """Counts computed from the graph itself."""
    nt = Counter(nd["type"] for nd in g["nodes"])
    et = Counter()
    for a, b in g["edges"]:
        ta, tb = g["nodes"][a]["type"], g["nodes"][b]["type"]
        et[" – ".join(sorted((ta, tb)))] += 1
    return {"nodes": len(g["nodes"]), "edges": len(g["edges"]), "nodes_by_type": dict(nt), "edges_by_type": dict(et.most_common()),
            "built_at": g["built_at"]}


# ----------------------------------------------------------------------------- node records (click a node)

RECORD_SQL = {
    "account": ("SELECT a.*, COALESCE(SUM(o.amount),0) AS open_pipeline, COUNT(o.deal_id) AS open_deals FROM accounts a "
                "LEFT JOIN open_pipeline o USING(account_id) WHERE a.account_id=? GROUP BY a.account_id", "accounts + open_pipeline"),
    "deal": ("SELECT * FROM open_pipeline WHERE deal_id=?", "open_pipeline"),
    "rep": ("SELECT r.*, (SELECT COALESCE(SUM(amount),0) FROM open_pipeline WHERE rep_id=r.rep_id) AS open_pipeline FROM reps r WHERE rep_id=?", "reps"),
    "region": ("SELECT region, year, SUM(revenue) AS revenue FROM revenue_quarterly WHERE region=? GROUP BY year", "revenue_quarterly"),
    "quarter": (None, "revenue_quarterly"),
    "center": ("SELECT * FROM site_candidates WHERE city=?", "site_candidates"),
    "industry": ("SELECT a.industry, COUNT(DISTINCT a.account_id) accounts, COALESCE(SUM(o.amount),0) open_pipeline FROM accounts a LEFT JOIN open_pipeline o USING(account_id) WHERE a.industry=?", "accounts"),
    "segment": ("SELECT a.segment, COUNT(DISTINCT a.account_id) accounts, COALESCE(SUM(o.amount),0) open_pipeline FROM accounts a LEFT JOIN open_pipeline o USING(account_id) WHERE a.segment=?", "accounts"),
    "college": ("SELECT college, COUNT(*) awards, SUM(total_amount) total_awarded FROM grant_awards WHERE college=?", "grant_awards"),
    "sponsor": ("SELECT sponsor, sponsor_type, COUNT(*) awards, SUM(total_amount) total_awarded FROM grant_awards WHERE sponsor=?", "grant_awards"),
    "fy": ("SELECT start_fy, COUNT(*) awards, SUM(annual_amount) annual_amount FROM grant_awards WHERE start_fy=?", "grant_awards"),
    "site": ("SELECT * FROM research_sites WHERE site=?", "research_sites"),
    "pi": ("SELECT p.*, COUNT(a.award_id) awards, COALESCE(SUM(a.total_amount),0) total_awarded FROM faculty_pis p LEFT JOIN grant_awards a ON a.pi_name=p.name WHERE p.pi_id=? GROUP BY p.pi_id", "faculty_pis + grant_awards"),
    "award": ("SELECT * FROM grant_awards WHERE award_id=?", "grant_awards"),
    "proposal": ("SELECT * FROM grant_pipeline WHERE proposal_id=?", "grant_pipeline"),
    "cohort": ("SELECT * FROM enrollment_funnel WHERE fall_year=? AND segment=?", "enrollment_funnel"),
}


def record(node_id: str) -> dict:
    t, _, key = node_id.partition(":")
    if t in ("doc", "document", "passage"):
        from atlas.knowledge import docs
        return docs.record(node_id)
    sql, table = RECORD_SQL.get(t, (None, None))
    con = _con()
    try:
        if t == "quarter":
            y, qn = key.split("Q")
            rows = _q(con, "SELECT region, revenue, recurring_revenue, new_business_revenue FROM revenue_quarterly WHERE year=? AND quarter=?", int(y), int(qn))
            if not rows:
                rows = _q(con, "SELECT COUNT(*) open_deals_closing, SUM(amount) open_pipeline FROM open_pipeline WHERE close_date BETWEEN ? AND ?", TODAY.isoformat(), Q_END.isoformat())
        elif t == "cohort":
            fy, seg = key.split(":", 1)
            rows = _q(con, sql, int(fy), seg)
        else:
            rows = _q(con, sql, key)
        return {"id": node_id, "type": t, "table": table, "rows": [dict(r) for r in rows][:12]}
    finally:
        con.close()


# ----------------------------------------------------------------------------- provenance

def provenance(persona: str, tool: str, result: dict) -> dict:
    """Which graph nodes (rows) a tool's answer drew on. Mirrors each tool's own selection SQL."""
    con = _con()
    ids: list[str] = []
    rows = 0
    try:
        def deals_q4():
            rs = _q(con, "SELECT deal_id, account_id, region FROM open_pipeline WHERE close_date BETWEEN ? AND ?", TODAY.isoformat(), Q_END.isoformat())
            return rs
        if persona == "enterprise":
            if tool in ("forecast_quarter", "what_if", "pipeline_coverage", "board_summary"):
                rs = deals_q4()
                rows += len(rs)
                ids += [f"deal:{r['deal_id']}" for r in rs] + sorted({f"account:{r['account_id']}" for r in rs})
                ids += [f"region:{r['region']}" for r in _q(con, "SELECT DISTINCT region FROM revenue_quarterly")]
                ids += [f"quarter:{r['year']}Q{r['quarter']}" for r in _q(con, "SELECT DISTINCT year, quarter FROM revenue_quarterly")] + [f"quarter:{TODAY.year}Q4"]
                rows += _q(con, "SELECT COUNT(*) n FROM revenue_quarterly")[0]["n"]
            if tool == "deal_risk":
                for d in result.get("values", {}).get("deals", []):
                    r = _q(con, "SELECT deal_id, account_id, rep_id, region FROM open_pipeline WHERE account_name=? AND amount=? AND close_date=?",
                           d["account"], d["amount"], d["close_date"])
                    for x in r:
                        ids += [f"deal:{x['deal_id']}", f"account:{x['account_id']}", f"rep:{x['rep_id']}", f"region:{x['region']}"]
                        rows += 1
            if tool == "score_sites":
                ids += [f"center:{r['city']}" for r in result.get("values", {}).get("ranked", [])]
                rows += len(ids)
            if tool in ("region_vs_plan", "board_summary"):
                ids += [f"region:{r['region']}" for r in result.get("values", {}).get("regions", [])] or \
                       [f"region:{r['region']}" for r in _q(con, "SELECT DISTINCT region FROM plan_targets")]
                ids += [f"quarter:{TODAY.year}Q{k}" for k in (1, 2, 3, 4)]
                rows += _q(con, "SELECT COUNT(*) n FROM plan_targets")[0]["n"]
            if tool == "hiring_plan":
                rs = _q(con, "SELECT rep_id, region FROM reps")
                ids += [f"rep:{r['rep_id']}" for r in rs] + sorted({f"region:{r['region']}" for r in rs})
                rows += len(rs)
        else:
            if tool in ("grant_forecast", "board_summary_university"):
                rs = _q(con, "SELECT proposal_id, college, sponsor FROM grant_pipeline WHERE start_fy=2027")
                ids += [f"proposal:{r['proposal_id']}" for r in rs] + sorted({f"college:{r['college']}" for r in rs}) + sorted({f"sponsor:{r['sponsor']}" for r in rs})
                aw = _q(con, "SELECT award_id FROM grant_awards WHERE start_fy + duration_years > 2027 AND start_fy <= 2027")
                ids += [f"award:{r['award_id']}" for r in aw] + ["fy:2027"]
                rows += len(rs) + len(aw)
            if tool == "sponsor_concentration":
                sp = list(result.get("values", {}).get("amounts", {}).keys())
                ids += [f"sponsor:{s}" for s in sp]
                aw = _q(con, "SELECT award_id FROM grant_awards WHERE start_fy + duration_years > 2027 AND start_fy <= 2027")
                ids += [f"award:{r['award_id']}" for r in aw]
                rows += len(aw)
            if tool == "score_research_sites":
                ids += [f"site:{r['site']}" for r in result.get("values", {}).get("ranked", [])]
                rows += len(ids)
            if tool in ("enrollment_forecast", "board_summary_university"):
                rs = _q(con, "SELECT fall_year, segment FROM enrollment_funnel")
                ids += [f"cohort:{r['fall_year']}:{r['segment']}" for r in rs]
                rows += len(rs)
    finally:
        con.close()
    g = load(persona)
    cited = []
    for c in result.get("citations", []) or []:
        cited += [c["passage_id"], c["doc_id"]]
    if cited:
        # the passages it cited, their documents, and the entities those passages name (edges built at ingest)
        index = {nd["id"]: i for i, nd in enumerate(g["nodes"])}
        pidx = {index[c["passage_id"]] for c in result["citations"] if c["passage_id"] in index}
        named = [g["nodes"][b if a in pidx else a]["id"] for a, b in g["edges"] if (a in pidx) != (b in pidx)]
        ids += cited + [n for n in named if not n.startswith(("doc:", "passage:"))]
    have = {nd["id"] for nd in g["nodes"]}
    seen, out = set(), []
    for i in ids:
        if i in have and i not in seen:
            seen.add(i)
            out.append(i)
    return {"nodes": out, "rows": rows, "tables": result.get("sources", []), "tool": tool}


PERSONA_TABLES = {
    "enterprise": ["accounts", "reps", "open_pipeline", "deals_history", "revenue_quarterly", "costs_quarterly",
                   "plan_targets", "pipeline_snapshots", "site_candidates"],
    "university": ["faculty_pis", "grant_awards", "grant_pipeline", "proposal_history", "enrollment_funnel",
                   "enrollment_plan", "research_sites", "operating_budget"],
}


def data_connected(persona: str) -> dict:
    con = _con()
    try:
        counts = {t: _q(con, f"SELECT COUNT(*) n FROM {t}")[0]["n"] for t in PERSONA_TABLES[persona]}
        if persona == "enterprise":
            lo = _q(con, "SELECT MIN(created_date) d FROM deals_history")[0]["d"]
            hi = _q(con, "SELECT MAX(close_date) d FROM open_pipeline")[0]["d"]
        else:
            lo = "FY" + str(_q(con, "SELECT MIN(fy) d FROM proposal_history")[0]["d"])
            hi = "FY" + str(_q(con, "SELECT MAX(start_fy) d FROM grant_pipeline")[0]["d"])
    finally:
        con.close()
    return {"tables": len(counts), "rows": sum(counts.values()), "by_table": counts, "from": lo, "to": hi,
            "db_bytes": DB_PATH.stat().st_size}
