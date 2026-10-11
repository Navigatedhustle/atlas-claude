"""Labelled decisions from the synthetic deal history for an optional Kev-4B fine-tune.
Task: "Will this Commit-stage deal close won?" (only Commit-stage deals have both outcomes in the data, so the stage
itself cannot leak the label). 1,000 deals sampled with seed 42; 20% held out. Same state format as production."""
import json
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "finetune"
QUESTION = "Will this Commit-stage deal close won?"


def money(x):
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"


def state(r):
    lines = [f"Account: {r['name']}", f"Industry: {r['industry']}, segment {r['segment']}, region {r['region']}", "Stage: Commit",
             f"Amount: {money(r['amount'])}", f"Executive sponsor engaged: {'yes' if r['exec_sponsor'] else 'no'}",
             f"Competitor involved: {'yes' if r['competitor'] else 'no'}", f"Discount offered: {r['discount_pct']:.1f}%",
             f"Close date pushed {r['close_date_pushes']} times", f"Days since last customer activity: {r['days_since_activity']}"]
    return "Meridian Dynamics (fictional company). " + " ".join(l + "." for l in lines)


def main():
    con = sqlite3.connect(f"file:{ROOT / 'data' / 'atlas.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT d.*, a.name, a.industry, a.segment FROM deals_history d JOIN accounts a USING(account_id) "
                       "WHERE d.max_stage = 'Commit' ORDER BY d.deal_id").fetchall()
    rng = random.Random(42)
    rows = rng.sample(list(rows), 1000)
    recs = [{"state": state(r), "questions": {"won": {"type": "noul", "instructions": QUESTION, "label": r["outcome"] == "won"}}} for r in rows]
    cut = int(len(recs) * 0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", recs[:cut]), ("heldout", recs[cut:])):
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in part) + "\n")
    hp = sum(x["questions"]["won"]["label"] for x in recs[cut:]) / len(recs[cut:])
    print(f"train {cut}, heldout {len(recs) - cut}, held-out share won {hp:.3f} (majority-class accuracy {max(hp, 1 - hp):.3f})")


if __name__ == "__main__":
    main()
