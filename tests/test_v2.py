"""v2 tests: the sandbox really isolates, graph counts match the database, and document answers stay grounded."""
import sqlite3

import httpx
import pytest

from atlas import grounding, sandbox
from atlas.config import DB_PATH, ROOT


# ----------------------------------------------------------------------------- sandbox

def test_sandbox_runs_sql_against_readonly_copy():
    r = sandbox.run("sql", "SELECT COUNT(*) AS n FROM accounts")
    assert r["ok"] and r["executed"] and r["table"]["rows"][0][0] == 600


def test_sandbox_timeout_kills_runaway_code():
    r = sandbox.run("python", "while True:\n    pass\n", timeout=2)
    assert r["timed_out"] and not r["ok"] and r["ms"] < 6000
    assert "killed after" in r["stderr"]


def test_sandbox_has_no_network():
    code = ("import socket\n"
            "for fam, addr in ((socket.AF_INET, ('1.1.1.1', 80)), (socket.AF_INET, ('127.0.0.1', 8100)), (socket.AF_INET6, ('::1', 8100))):\n"
            "    try:\n"
            "        s = socket.socket(fam, socket.SOCK_STREAM); s.settimeout(2); s.connect(addr); print('OPEN', addr)\n"
            "    except OSError as e:\n"
            "        print('blocked', addr, e.errno)\n")
    r = sandbox.run("python", code)
    assert r["ok"], r["stderr"]
    assert "OPEN" not in r["stdout"] and r["stdout"].count("blocked") == 3


def test_sandbox_database_is_read_only_and_original_untouched():
    before = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    code = ("import sqlite3, os\n"
            "c = sqlite3.connect('atlas.db')\n"
            "for stmt in ('DELETE FROM accounts', 'DROP TABLE accounts'):\n"
            "    try:\n        c.execute(stmt); c.commit(); print('WROTE')\n    except Exception as e:\n        print('blocked', type(e).__name__)\n"
            "for f in (lambda: os.chmod('atlas.db', 0o666), lambda: open('../db/atlas.db', 'ab'), lambda: open(%r, 'w')):\n"
            "    try:\n        f(); print('ESCAPED')\n    except OSError as e:\n        print('blocked', e.errno)\n" % str(ROOT / "SANDBOX_ESCAPE"))
    r = sandbox.run("python", code)
    assert r["ok"], r["stderr"]
    assert "WROTE" not in r["stdout"] and "ESCAPED" not in r["stdout"]
    assert not (ROOT / "SANDBOX_ESCAPE").exists()
    after = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    assert before == after == 600


def test_sandbox_reports_errors_verbatim():
    r = sandbox.run("python", "raise ValueError('boom 42')")
    assert not r["ok"] and r["exit_code"] == 1 and "ValueError: boom 42" in r["stderr"]


# ----------------------------------------------------------------------------- knowledge graph counts

@pytest.mark.parametrize("persona", ["enterprise", "university"])
def test_graph_counts_match_database(persona):
    from atlas.knowledge import graph as G
    g = G.load(persona)
    s = G.stats(g)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    n = lambda sql: con.execute(sql).fetchone()[0]
    t = s["nodes_by_type"]
    if persona == "enterprise":
        assert t["account"] == n("SELECT COUNT(*) FROM accounts")
        assert t["deal"] == n("SELECT COUNT(*) FROM open_pipeline")
        assert t["rep"] == n("SELECT COUNT(*) FROM reps")
        assert t["region"] == n("SELECT COUNT(DISTINCT region) FROM revenue_quarterly")
        assert t["center"] == n("SELECT COUNT(*) FROM site_candidates")
    else:
        assert t["pi"] == n("SELECT COUNT(*) FROM faculty_pis")
        assert t["award"] == n("SELECT COUNT(*) FROM grant_awards")
        assert t["proposal"] == n("SELECT COUNT(*) FROM grant_pipeline")
        assert t["sponsor"] == n("SELECT COUNT(DISTINCT sponsor) FROM grant_awards")
    assert s["nodes"] == len(g["nodes"]) == sum(t.values())
    assert s["edges"] == len(g["edges"]) == sum(s["edges_by_type"].values())
    assert all(0 <= a < s["nodes"] and 0 <= b < s["nodes"] for a, b in g["edges"])


def test_provenance_names_the_rows_the_tool_used():
    from atlas.analytics import run_tool
    from atlas.knowledge import graph as G
    r = run_tool("enterprise", "deal_risk", {"n": 5})
    pr = G.provenance("enterprise", "deal_risk", r)
    deals = [i for i in pr["nodes"] if i.startswith("deal:")]
    assert len(deals) == 5 and pr["rows"] == 5
    rec = G.record(deals[0])
    assert rec["rows"] and rec["rows"][0]["stage"] in ("Propose", "Commit")


# ----------------------------------------------------------------------------- citation grounding

def _embed_up():
    try:
        return httpx.get("http://127.0.0.1:8140/health", timeout=1).json().get("ok")
    except Exception:
        return False


def test_number_in_cited_passage_is_grounded_and_invented_number_is_not():
    passage = {"facts": ['From "Pricing and discount approval policy (FY2026)" (synthetic document), passage 2: Any discount above 20% is flagged for discount approval.']}
    assert grounding.check("Discounts above 20% are flagged for approval.", passage)["ok"]
    assert not grounding.check("Discounts above 35% are flagged for approval.", passage)["ok"]


@pytest.mark.skipif(not _embed_up(), reason="embedding service not running")
@pytest.mark.parametrize("persona,q,doc", [
    ("enterprise", "What did the last QBR say about risk in our weakest region?", "doc:ent-qbr-international"),
    ("enterprise", "Summarize the board's concerns about margin.", "doc:ent-board-q3"),
    ("enterprise", "Which accounts does our pricing policy flag for discount approval?", "doc:ent-pricing"),
    ("university", "Which PIs have the strongest NIH track record?", "doc:uni-nih-review"),
    ("university", "What does the sponsor guidance say about cost sharing?", "doc:uni-sponsor-guide"),
    ("university", "Summarize our enrollment strategy risks.", "doc:uni-enrollment"),
])
def test_search_knowledge_cites_the_right_document(persona, q, doc):
    from atlas.knowledge import docs
    r = docs.search_knowledge(persona, q)
    assert r["citations"][0]["doc_id"] == doc
    # every number in the cited passages is available to the grounding check, and nothing else is added
    for c in r["citations"]:
        assert grounding.check(c["text"], r)["ok"]


def test_corpus_numbers_match_the_database():
    """Documents are generated from live tool output: their numbers must be checkable against those tools."""
    from atlas.analytics import run_tool
    from atlas.knowledge import docs
    d = {x["id"]: x for x in docs._enterprise_docs()}
    rv = run_tool("enterprise", "region_vs_plan", {})
    assert grounding.check(d["doc:ent-qbr-international"]["paras"][0], rv)["ok"]
    hp = run_tool("enterprise", "hiring_plan", {})
    for p in d["doc:ent-hiring"]["paras"]:
        assert grounding.check(p, hp)["ok"]
    for doc in d.values():
        assert doc["title"]
    assert all(x["synthetic"] for x in docs.load("enterprise")[0]["docs"])


@pytest.mark.skipif(not _embed_up(), reason="embedding service not running")
def test_document_answer_lights_cited_passages_and_named_entities():
    from atlas.knowledge import docs, graph as G
    r = docs.search_knowledge("university", "Which PIs have the strongest NIH track record?")
    pr = G.provenance("university", "search_knowledge", r)
    assert r["citations"][0]["passage_id"] in pr["nodes"] and "doc:uni-nih-review" in pr["nodes"]
    assert any(n.startswith("pi:") for n in pr["nodes"])
    rec = G.record("doc:uni-nih-review")
    assert rec["type"] == "document" and "NIH" in rec["text"]
    assert G.record(r["citations"][0]["passage_id"])["text"] == r["citations"][0]["text"]
