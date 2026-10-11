"""Second brain, part 2: a local document index over a SYNTHETIC corpus.

The base corpus is generated from the live database and tool outputs (so every number matches the data), split
into passages, embedded with NVIDIA Nemotron-3-Embed-1B on this box, and searched by cosine similarity.
`search_knowledge` answers with citations (document + passage). "Feed the brain" asks the local LLM to write one
new synthetic document from tool facts, drops any sentence whose numbers are not in those facts, then ingests it
live (chunk, embed, add graph nodes) and times every step."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
import numpy as np

from atlas.config import AS_OF, DATA_DIR

DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "brain"
EMBED_URL = "http://127.0.0.1:8140"
EMBED_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
SYN = "SYNTHETIC DOCUMENT · fictional organization · generated for the Atlas demo"


# ----------------------------------------------------------------------------- base corpus (numbers from the data)

def _enterprise_docs() -> list[dict]:
    from atlas.analytics import enterprise as E
    from atlas.analytics.common import money, q
    fq, rv, hp, bs, ss, pc = E.forecast_quarter(), E.region_vs_plan(), E.hiring_plan(), E.board_summary(), E.score_sites(), E.pipeline_coverage()
    v = fq["values"]
    risk = E.deal_risk(60)["values"]["deals"]
    docs = []
    docs.append({"id": "doc:ent-board-q3", "kind": "Board memo", "date": "2026-10-02", "title": "Board memo: Q3 2026 business review and margin",
                 "paras": [
                     f"Prepared for the Meridian Dynamics board. Projected Q4 2026 revenue is {money(v['revenue_p50'])} at the median, with an 80% range of {money(v['revenue_p10'])} to {money(v['revenue_p90'])}. Projected operating profit is {money(v['profit_p50'])}, an operating margin of {v['operating_margin'] * 100:.1f}%.",
                     f"The board's first concern about margin is cost: the Q4 2026 operating expense budget is {money(v['opex_budget'])}, against a gross margin of {v['gross_margin'] * 100:.1f}%. Directors asked whether opex growth is outrunning projected Q4 revenue growth of {v['yoy_growth'] * 100:.1f}% year over year.",
                     "The board's second concern about margin is discounting. Late-stage deals are carrying deeper discounts than the pricing policy intends, and directors asked the CRO to report every deal above the approval threshold at the next meeting.",
                     f"The board's third concern is concentration of forecast risk: {fq['facts'][6]} Directors asked for a mitigation plan for the largest late-stage deals.",
                 ]})
    weakest = min(rv["values"]["regions"], key=lambda x: x["gap_pct"])["region"]  # furthest behind plan
    for reg in rv["values"]["regions"]:
        r = reg["region"]
        rd = [d for d in risk if d["region"] == r][:3]
        risk_lines = "; ".join(f"{d['account']} ({money(d['amount'])}, {d['stage']}, {d['risk'] * 100:.0f}% loss risk" +
                               (f": {', '.join(x for x in d['reasons'][:2] if x)})" if any(d['reasons'][:2]) else ")") for d in rd)
        weak = r == weakest
        status = "behind" if reg["gap"] < 0 else "ahead of"
        docs.append({"id": f"doc:ent-qbr-{r.lower()}", "kind": "QBR notes", "date": "2026-10-06", "title": f"QBR notes: {r} region, Q3 2026",
                     "paras": [
                         f"Quarterly business review for the {r} region. FY2026 revenue is projected at {money(reg['projected'])} against a plan of {money(reg['plan'])}, {abs(reg['gap_pct']) * 100:.1f}% {status} plan.",
                         ((f"Risk discussion for {r}, our weakest region (furthest behind plan): " if weak else f"Risk discussion for {r}: ") +
                          (f"the deals most likely to be lost this quarter are {risk_lines}." if rd else "no late-stage deals were flagged as high risk this quarter.")),
                         (f"Because {r} is the region furthest behind plan, the team agreed weekly deal reviews for every Commit-stage deal and an executive sponsor on each deal above $1M." if r == weakest
                          else f"The {r} team will keep its standard bi-weekly pipeline review cadence."),
                     ]})
    for row in q("SELECT account_id, account_name, SUM(amount) amt, COUNT(*) n FROM open_pipeline GROUP BY account_id ORDER BY amt DESC LIMIT 3"):
        deals = q("SELECT stage, amount, close_date, rep_name FROM open_pipeline WHERE account_id=? ORDER BY amount DESC", row["account_id"])
        dl = "; ".join(f"{d['stage']} deal of {money(d['amount'])} closing {d['close_date']} (owner {d['rep_name']})" for d in deals)
        docs.append({"id": f"doc:ent-plan-{row['account_id']}", "kind": "Account plan", "date": "2026-09-30", "title": f"Account plan: {row['account_name']}",
                     "paras": [f"{row['account_name']} has {row['n']} open deals worth {money(row['amt'])} in total.",
                               f"Open deals: {dl}.",
                               "Plan: secure an executive sponsor, confirm budget timing, and keep discounts inside policy."]})
    flagged = q("SELECT account_name, MAX(discount_pct) d, COUNT(*) n FROM open_pipeline WHERE discount_pct > 20 GROUP BY account_name ORDER BY d DESC LIMIT 10")
    n20 = q("SELECT COUNT(*) n FROM open_pipeline WHERE discount_pct > 20")[0]["n"]
    n25 = q("SELECT COUNT(*) n FROM open_pipeline WHERE discount_pct > 25")[0]["n"]
    docs.append({"id": "doc:ent-pricing", "kind": "Policy", "date": "2026-01-15", "title": "Pricing and discount approval policy (FY2026)",
                 "paras": ["Reps may discount up to 10% at their own discretion. Discounts above 10% and up to 20% need regional VP approval.",
                           "Any discount above 20% is flagged for discount approval by the deal desk and the CFO before a quote is sent. Discounts above 25% also require a written margin justification.",
                           f"Status as of {AS_OF}: {n20} open deals carry a discount above 20% and {n25} carry a discount above 25%.",
                           "Accounts flagged for discount approval (highest discount first): " + "; ".join(f"{r['account_name']} ({r['d']:.1f}%)" for r in flagged) + "."]})
    docs.append({"id": "doc:ent-hiring", "kind": "Plan", "date": "2026-09-20", "title": "FY2027 sales hiring plan",
                 "paras": hp["facts"][:3] + [" ".join(hp["facts"][3:5])]})
    docs.append({"id": "doc:ent-sites", "kind": "Memo", "date": "2026-09-25", "title": "Sales center site selection memo",
                 "paras": [" ".join(ss["facts"][:2]), " ".join(ss["facts"][2:4])]})
    docs.append({"id": "doc:ent-coverage", "kind": "Memo", "date": "2026-10-07", "title": "Pipeline coverage note, Q4 2026",
                 "paras": [" ".join(pc["facts"])]})
    return docs


def _university_docs() -> list[dict]:
    from atlas.analytics import university as U
    from atlas.analytics.common import money, q
    gf, sc, ef, rs = U.grant_forecast(), U.sponsor_concentration(), U.enrollment_forecast(), U.score_research_sites()
    docs = []
    top = q("SELECT pi_name, COUNT(*) n, SUM(total_amount) amt, MIN(start_fy) f0 FROM grant_awards WHERE sponsor='NIH' GROUP BY pi_name ORDER BY amt DESC LIMIT 5")
    docs.append({"id": "doc:uni-nih-review", "kind": "Portfolio review", "date": "2026-09-15", "title": "NIH portfolio review, FY2026",
                 "paras": ["This review ranks Alta Vista principal investigators by total NIH funding awarded since FY2022.",
                           "Strongest NIH track records, ranked by total NIH funding: " + "; ".join(f"{r['pi_name']} is {o} with {money(r['amt'])} from {r['n']} NIH awards" for o, r in zip(["first", "second", "third", "fourth", "fifth"], top)) + ".",
                           f"{top[0]['pi_name']} has the strongest NIH track record, leads the portfolio and is recommended to mentor early-career applicants in the next NIH cycle."]})
    topics = ["machine learning for early sepsis detection", "community health outcomes in rural counties", "neural circuits of decision making"]
    for r, topic in zip(top[:3], topics):
        a = q("SELECT award_id, college, start_fy, duration_years, total_amount, annual_amount FROM grant_awards WHERE sponsor='NIH' AND pi_name=? ORDER BY total_amount DESC LIMIT 1", r["pi_name"])[0]
        docs.append({"id": f"doc:uni-abstract-{a['award_id']}", "kind": "Grant abstract", "date": f"FY{a['start_fy']}", "title": f"Grant abstract: {topic} ({r['pi_name']})",
                     "paras": [f"Principal investigator {r['pi_name']} ({a['college']}) leads a {a['duration_years']}-year NIH project on {topic}, starting in FY{a['start_fy']}.",
                               f"The award totals {money(a['total_amount'])}, or {money(a['annual_amount'])} per year.",
                               "Aims: build a shared dataset, validate methods across two partner sites, and train graduate researchers."]})
    docs.append({"id": "doc:uni-sponsor-guide", "kind": "Sponsor guidelines", "date": "2026-07-01", "title": "Sponsor guidelines digest: cost sharing",
                 "paras": ["NSF: voluntary committed cost sharing is prohibited unless the solicitation requires it; do not include it in budgets.",
                           "NIH: cost sharing is not required for most research grants; institutional commitments are reported but not budgeted.",
                           "DOE: research and development awards typically require 20% cost sharing, and demonstration projects require 50%.",
                           "DoD: cost sharing is not required.",
                           "Industry sponsors: cost sharing is negotiated separately in each industry contract.",
                           "Alta Vista policy: any voluntary cost share must be approved by the VP for Research before submission."]})
    docs.append({"id": "doc:uni-enrollment", "kind": "Strategy", "date": "2026-09-10", "title": "Enrollment strategy, Fall 2027",
                 "paras": [" ".join(ef["facts"][:3]),
                           "Risks: " + ef["facts"][3] + " International yield is exposed to visa processing times.",
                           ef["facts"][4] + " The strategy shifts recruiting spend toward in-state and transfer pipelines."]})
    docs.append({"id": "doc:uni-board-funding", "kind": "Board memo", "date": "2026-09-28", "title": "Board memo: research funding outlook FY27",
                 "paras": [" ".join(gf["facts"][:3]), " ".join(gf["facts"][3:6]), " ".join(sc["facts"][:2])]})
    docs.append({"id": "doc:uni-sites", "kind": "Memo", "date": "2026-09-05", "title": "Research center siting memo",
                 "paras": [" ".join(rs["facts"][:2]), " ".join(rs["facts"][2:4])]})
    return docs


def build_corpus(persona: str) -> list[dict]:
    docs = _enterprise_docs() if persona == "enterprise" else _university_docs()
    d = DOCS_DIR / persona
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        if not f.name.startswith("feed-"):
            f.unlink()
    for doc in docs:
        doc.update({"synthetic": True, "label": SYN, "persona": persona, "source": "generated from the live database"})
        (d / (doc["id"].split(":")[1] + ".json")).write_text(json.dumps(doc, indent=1))
    return docs


# ----------------------------------------------------------------------------- index

def _ipath(persona: str) -> tuple[Path, Path]:
    return INDEX_DIR / f"index_{persona}.json", INDEX_DIR / f"vectors_{persona}.npy"


def embed(texts: list[str], kind: str) -> tuple[np.ndarray, float]:
    r = httpx.post(f"{EMBED_URL}/embed", json={"texts": texts, "kind": kind}, timeout=60)
    r.raise_for_status()
    d = r.json()
    return np.array(d["vectors"], dtype=np.float32), d["ms"]


def passages_of(doc: dict) -> list[dict]:
    return [{"id": f"passage:{doc['id'].split(':')[1]}:{i}", "doc_id": doc["id"], "title": doc["title"], "text": p, "n": i}
            for i, p in enumerate(doc["paras"])]


def build_index(persona: str) -> dict:
    docs = sorted((json.loads(f.read_text()) for f in (DOCS_DIR / persona).glob("*.json")), key=lambda d: d["id"])
    ps = [p for d in docs for p in passages_of(d)]
    vecs, ms = embed([f"{p['title']}. {p['text']}" for p in ps], "passage")
    meta = {"persona": persona, "model": EMBED_MODEL, "dim": int(vecs.shape[1]), "docs": [{k: d[k] for k in ("id", "title", "kind", "date", "synthetic", "source")} for d in docs],
            "passages": ps, "updated_at": time.time(), "embed_ms": ms}
    jp, vp = _ipath(persona)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(vp, vecs.astype(np.float16))
    jp.write_text(json.dumps(meta))
    _CACHE.pop(persona, None)
    link_graph(persona, docs)
    return meta


_CACHE: dict = {}


def build_keyword_index(persona: str) -> dict:
    """Regenerate a synthetic corpus without model access; zero-width vectors are not embeddings."""
    documents = build_corpus(persona)
    passages = [p for doc in documents for p in passages_of(doc)]
    meta = {"persona": persona, "model": "keyword-only", "dim": 0,
            "docs": [{k: doc[k] for k in ("id", "title", "kind", "date", "synthetic", "source")} for doc in documents],
            "passages": passages, "updated_at": time.time(), "embed_ms": None}
    jp, vp = _ipath(persona)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(vp, np.empty((len(passages), 0), dtype=np.float16))
    jp.write_text(json.dumps(meta))
    _CACHE.pop(persona, None)
    return meta


def load(persona: str):
    jp, vp = _ipath(persona)
    if not jp.exists() or not vp.exists():
        build_keyword_index(persona)
    mt = jp.stat().st_mtime
    c = _CACHE.get(persona)
    if c and c[0] == mt:
        return c[1], c[2]
    meta = json.loads(jp.read_text())
    vecs = np.load(vp).astype(np.float32)
    _CACHE[persona] = (mt, meta, vecs)
    return meta, vecs


def stats(persona: str) -> dict:
    meta, vecs = load(persona)
    return {"docs": len(meta["docs"]), "passages": len(meta["passages"]), "vectors": int(vecs.shape[0]), "dim": int(vecs.shape[1]),
            "model": meta["model"], "updated_at": meta["updated_at"], "synthetic": all(d.get("synthetic") for d in meta["docs"])}


def search(persona: str, query: str, k: int = 4) -> tuple[list[dict], dict]:
    meta, vecs = load(persona)
    t = time.perf_counter()
    mode = "vector"
    try:
        if meta.get("dim", 0) == 0:
            raise RuntimeError("Keyword-only index; build the vector index with a local embedding service")
        qv, _ = embed([query], "query")
        scores = vecs @ qv[0]
    except Exception:  # embedding service down: keyword overlap, clearly labelled
        mode = "keyword fallback"
        qt = set(re.findall(r"[a-z0-9]+", query.lower()))
        scores = np.array([len(qt & set(re.findall(r"[a-z0-9]+", (p["title"] + " " + p["text"]).lower()))) / (len(qt) or 1) for p in meta["passages"]])
    order = np.argsort(-scores)[:k]
    hits = [{**meta["passages"][i], "score": float(scores[i])} for i in order]
    return hits, {"ms": (time.perf_counter() - t) * 1000, "mode": mode, "searched": len(meta["passages"])}


# ----------------------------------------------------------------------------- the tool

def search_knowledge(persona: str, question: str) -> dict:
    from atlas.analytics.common import result
    hits, info = search(persona, question)
    top_doc = hits[0]["doc_id"]
    # Keep the best passages; prefer the top document's own passages so the answer reads as one source.
    cited = [h for h in hits if h["doc_id"] == top_doc] + [h for h in hits if h["doc_id"] != top_doc][:1]
    cited = cited[:3]
    facts = [f"From \"{h['title']}\" (synthetic document), passage {h['n'] + 1}: {h['text']}" for h in cited]
    title = cited[0]["title"]
    return result(
        tool="search_knowledge", title=f"From the documents: {title}",
        headline={"label": "Grounded in a cited passage", "value": title, "fmt": "text",
                  "sub": f"{len(cited)} passages cited · best match {cited[0]['score']:.2f} · {info['searched']} passages searched in {info['ms']:.0f} ms"},
        values={"hits": [{"doc_id": h["doc_id"], "passage_id": h["id"], "score": h["score"]} for h in hits]},
        facts=facts,
        method={"summary": f"The question is embedded with {EMBED_MODEL} on this box and compared with every indexed passage (cosine similarity, {info['mode']}). The answer may only use the cited passages.",
                "formula": "score = cos(embed('query: ' + question), embed('passage: ' + passage))",
                "steps": [f"{h['title']} · passage {h['n'] + 1} · score {h['score']:.2f}" for h in hits[:3]]},
        inputs=[], sources=[f"{h['title']} (passage {h['n'] + 1})" for h in cited],
        chart={"type": "none", "title": "", "labels": [], "series": []},
        template=" ".join(h["text"] for h in cited[:1]),
    ) | {"citations": [{"doc_id": h["doc_id"], "passage_id": h["id"], "title": h["title"], "n": h["n"], "text": h["text"], "score": h["score"]} for h in cited]}


# ----------------------------------------------------------------------------- graph links + records

ENTITY_PATTERNS = {
    "enterprise": "SELECT 'region:' || region, region FROM (SELECT DISTINCT region FROM revenue_quarterly) UNION ALL "
                  "SELECT 'account:' || account_id, name FROM accounts UNION ALL SELECT 'rep:' || rep_id, name FROM reps UNION ALL "
                  "SELECT 'center:' || city, city FROM site_candidates",
    "university": "SELECT 'college:' || college, college FROM (SELECT DISTINCT college FROM grant_awards) UNION ALL "
                  "SELECT 'sponsor:' || sponsor, sponsor FROM (SELECT DISTINCT sponsor FROM grant_awards) UNION ALL "
                  "SELECT 'pi:' || pi_id, name FROM faculty_pis UNION ALL SELECT 'site:' || site, site FROM research_sites",
}


def mentions(persona: str, text: str) -> list[str]:
    from atlas.knowledge.graph import _con
    con = _con()
    ents = con.execute(ENTITY_PATTERNS[persona]).fetchall()
    con.close()
    out = []
    for nid, name in ents:
        if name and re.search(r"\b" + re.escape(name) + r"\b", text):
            out.append(nid)
    return out


def doc_nodes(persona: str, doc: dict) -> tuple[list[dict], list[tuple]]:
    nodes = [{"id": doc["id"], "type": "document", "label": doc["title"], "value": float(len(doc["paras"])), "fmt": "num", "r": 6.5}]
    edges = []
    for p in passages_of(doc):
        nodes.append({"id": p["id"], "type": "passage", "label": f"{doc['title']} · passage {p['n'] + 1}", "value": float(len(p["text"])), "fmt": "num", "r": 2.4})
        edges.append((p["id"], doc["id"]))
        for m in mentions(persona, p["text"]):
            edges.append((p["id"], m))
    return nodes, edges


def link_graph(persona: str, docs: list[dict]):
    from atlas.knowledge import graph as G
    nodes, edges = [], []
    for d in docs:
        n, e = doc_nodes(persona, d)
        nodes += n
        edges += e
    G.build(persona, nodes, edges)


def record(node_id: str) -> dict:
    for persona in ("enterprise", "university"):
        try:
            meta, _ = load(persona)
        except FileNotFoundError:
            continue
        if node_id.startswith("passage:"):
            p = next((p for p in meta["passages"] if p["id"] == node_id), None)
            if p:
                return {"id": node_id, "type": "passage", "table": "document index", "source": p["title"], "text": p["text"]}
        d = next((d for d in meta["docs"] if d["id"] == node_id), None)
        if d:
            ps = [p["text"] for p in meta["passages"] if p["doc_id"] == node_id]
            return {"id": node_id, "type": "document", "table": "document index", "source": f"{d['kind']} · {d['date']} · SYNTHETIC",
                    "text": " ".join(ps)[:1400]}
    return {"id": node_id, "rows": [], "table": "not found"}


# ----------------------------------------------------------------------------- feed the brain

FEED_TOPICS = {
    "enterprise": [
        {"key": "q4-board-margin", "title": "Q4 2026 board memo: margin concerns", "kind": "Board memo",
         "tools": ["forecast_quarter", "board_summary"],
         "brief": "A memo from the CFO to the board about margin concerns for Q4 2026: projected revenue and operating profit, operating margin, gross margin, the opex budget, and the deal adding the most uncertainty. End with two concrete actions management will take on discounting and opex, described in words only, with no numbers.",
         "question": "What does the Q4 board memo say about margin concerns?"},
        {"key": "intl-recovery", "title": "International recovery plan, Q4 2026", "kind": "Plan",
         "tools": ["region_vs_plan"], "facts_fn": "intl_risk_facts",
         "brief": "A recovery plan for the International region, which is furthest behind plan: the gap to plan, the at-risk deals, and three actions for the quarter, described in words only, with no numbers.",
         "question": "What is in the International recovery plan?"},
    ],
    "university": [
        {"key": "nsf-ai-science", "title": "New NSF solicitation: AI for science", "kind": "Sponsor solicitation brief",
         "tools": ["sponsor_concentration", "grant_forecast"],
         "brief": "A brief for faculty about a new NSF solicitation called AI for Science: what NSF already funds at Alta Vista, which colleges are best placed, how it changes sponsor concentration, and next steps for proposal teams.",
         "question": "What does the new NSF AI for science solicitation mean for us?"},
        {"key": "nih-scenario", "title": "NIH budget scenario memo", "kind": "Memo",
         "tools": ["sponsor_concentration"],
         "brief": "A memo to the VP for Research on what a cut to NIH funding would mean: dependence on NIH, exposure in dollars, and mitigation steps.",
         "question": "What does the NIH budget scenario memo recommend?"},
    ],
}

FEED_SYS = ("You write short internal documents for {org}, a fictional organization; everything is synthetic. Write 3 or 4 "
            "paragraphs of 40 to 80 words each, plain text, separated by blank lines. No title, no headings, no markdown, no "
            "lists. Use only numbers that appear in FACTS, copied exactly as written (including $ and %). Never calculate, "
            "round or invent numbers, dates or percentages; if you need a number that is not in FACTS, describe it in words. "
            "Do not claim causes, shares or totals that FACTS do not state (for example, do not say a gap is concentrated "
            "in or explained by particular deals).")


def _feed_dir(persona: str) -> Path:
    return DOCS_DIR / persona


def intl_risk_facts() -> list[str]:
    """At-risk deals in the International region only, so a regional document cannot borrow other regions' deals."""
    from atlas.analytics import enterprise as E
    from atlas.analytics.common import money
    deals = [d for d in E.deal_risk(60)["values"]["deals"] if d["region"] == "International"][:4]
    return [f"International at-risk deal: {d['account']}, {money(d['amount'])}, {d['stage']} stage, {d['risk'] * 100:.0f}% loss risk"
            + (f" ({', '.join(x for x in d['reasons'][:2] if x)})." if any(d['reasons'][:2]) else ".") for d in deals]


def fed_docs(persona: str) -> list[str]:
    return sorted(f.stem for f in _feed_dir(persona).glob("feed-*.json"))


def feed(persona: str, llm_url: str) -> dict:
    from atlas import grounding
    from atlas.analytics import run_tool
    from atlas.config import PERSONAS
    from atlas.knowledge import graph as G
    steps = []
    t_all = time.perf_counter()
    done = set(fed_docs(persona))
    topic = next((t for t in FEED_TOPICS[persona] if f"feed-{t['key']}" not in done), None)
    if topic is None:
        return {"ok": False, "error": "Every feed topic is already ingested. Reset the brain to run it again."}
    doc_id = f"doc:feed-{topic['key']}"
    # 1. facts from the real tools
    facts = []
    for t in topic["tools"]:
        facts += run_tool(persona, t, {})["facts"]
    if topic.get("facts_fn"):
        facts += globals()[topic["facts_fn"]]()
    # 2. the local LLM writes the document
    t = time.perf_counter()
    msgs = [{"role": "system", "content": FEED_SYS.format(org=PERSONAS[persona]["org"])},
            {"role": "user", "content": f"DOCUMENT: {topic['title']}\nWHAT IT SHOULD COVER: {topic['brief']}\nFACTS:\n" + "\n".join(f"- {f}" for f in facts)}]
    r = httpx.post(f"{llm_url}/v1/chat/completions", json={"messages": msgs, "max_tokens": 520, "temperature": 0.3}, timeout=60)
    r.raise_for_status()
    js = r.json()
    raw = js["choices"][0]["message"]["content"].strip()
    steps.append({"step": "Written by the local LLM", "ms": (time.perf_counter() - t) * 1000, "detail": f"{js.get('usage', {}).get('completion_tokens', 0)} tokens"})
    # 3. every number must come from the facts; sentences that fail are dropped
    t = time.perf_counter()
    paras, kept_n, dropped = [], 0, []
    for para in [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]:
        para = re.sub(r"^[#*\-\s]+", "", para)
        sents = re.split(r"(?<=[.!?])\s+", para)
        ok = []
        for s in sents:
            g = grounding.check(s, {"facts": facts})
            if g["ok"]:
                ok.append(s)
                kept_n += len(g["numbers"])
            else:
                dropped.append({"sentence": s, "unmatched": g["unmatched"]})
        if ok:
            paras.append(" ".join(ok))
    steps.append({"step": "Numbers checked against the data", "ms": (time.perf_counter() - t) * 1000,
                  "detail": f"{kept_n} numbers verified, {len(dropped)} sentence(s) dropped"})
    if not paras:
        return {"ok": False, "error": "The generated document had no sentence that passed the number check.", "dropped": dropped}
    doc = {"id": doc_id, "kind": topic["kind"], "date": AS_OF, "title": topic["title"], "paras": paras, "synthetic": True, "label": SYN,
           "persona": persona, "source": "written on this box by the local LLM from tool facts (Feed the brain)"}
    # 4. before: what does the index return for the suggested question?
    before, _ = search(persona, topic["question"], k=1)
    # 5. ingest: chunk, embed, append to the index, add graph nodes
    t_ing = time.perf_counter()
    t = time.perf_counter()
    ps = passages_of(doc)
    steps.append({"step": "Chunked", "ms": (time.perf_counter() - t) * 1000, "detail": f"{len(ps)} passages"})
    vecs, emb_ms = embed([f"{p['title']}. {p['text']}" for p in ps], "passage")
    steps.append({"step": "Embedded on this box", "ms": emb_ms, "detail": f"{len(ps)} vectors × {vecs.shape[1]} dims"})
    t = time.perf_counter()
    meta, old = load(persona)
    meta["docs"].append({k: doc[k] for k in ("id", "title", "kind", "date", "synthetic", "source")})
    meta["passages"] += ps
    meta["updated_at"] = time.time()
    jp, vp = _ipath(persona)
    np.save(vp, np.vstack([old, vecs]).astype(np.float16))
    jp.write_text(json.dumps(meta))
    (_feed_dir(persona) / f"feed-{topic['key']}.json").write_text(json.dumps(doc, indent=1))
    _CACHE.pop(persona, None)
    steps.append({"step": "Added to the vector index", "ms": (time.perf_counter() - t) * 1000, "detail": f"{len(meta['passages'])} passages indexed"})
    t = time.perf_counter()
    nodes, edges = doc_nodes(persona, doc)
    g = G.add_nodes(persona, nodes, edges)
    steps.append({"step": "Added to the knowledge graph", "ms": (time.perf_counter() - t) * 1000, "detail": f"+{len(nodes)} nodes, +{len(edges)} edges"})
    ingest_ms = (time.perf_counter() - t_ing) * 1000
    # 6. after: the same question now retrieves the new document
    after, _ = search(persona, topic["question"], k=1)
    return {"ok": True, "doc": doc, "steps": steps, "ingest_ms": ingest_ms, "total_ms": (time.perf_counter() - t_all) * 1000,
            "new_nodes": [n["id"] for n in nodes], "new_edges": len(edges), "graph_nodes": len(g["nodes"]), "graph_edges": len(g["edges"]),
            "question": topic["question"], "dropped": dropped,
            "before": {"doc_id": before[0]["doc_id"], "title": before[0]["title"], "score": before[0]["score"]},
            "after": {"doc_id": after[0]["doc_id"], "title": after[0]["title"], "score": after[0]["score"]},
            "proven": after[0]["doc_id"] == doc_id and before[0]["doc_id"] != doc_id}


def reset_feed(persona: str) -> dict:
    for f in _feed_dir(persona).glob("feed-*.json"):
        f.unlink()
    meta = build_index(persona)
    return {"ok": True, "docs": len(meta["docs"]), "passages": len(meta["passages"])}


def ensure(persona: str):
    """Cold start: build the corpus, index and graph if any are missing (needs the embedding service)."""
    jp, vp = _ipath(persona)
    if not (DOCS_DIR / persona).exists() or not any((DOCS_DIR / persona).glob("*.json")):
        build_corpus(persona)
    if not (jp.exists() and vp.exists()):
        build_index(persona)
