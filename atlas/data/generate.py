"""Synthetic data generator for Atlas.

All companies, people, and numbers are fictional and generated from a fixed seed.

    python -m atlas.data.generate --seed 42
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np

from atlas.config import DATA_DIR, DB_PATH

AS_OF = dt.date(2026, 10, 8)

FIRST = ["Avery", "Jordan", "Priya", "Mateo", "Lena", "Marcus", "Sofia", "Kenji", "Amara", "Diego",
         "Hannah", "Omar", "Grace", "Ravi", "Elena", "Tobias", "Nadia", "Caleb", "Yuki", "Isabel",
         "Wesley", "Leah", "Andre", "Mina", "Rowan", "Tessa", "Felix", "Zara", "Hugo", "Celia",
         "Dmitri", "Noor", "Simone", "Arjun", "Bianca", "Theo", "Ingrid", "Malik", "Paloma", "Quinn"]
LAST = ["Okafor", "Lindqvist", "Ramires", "Castellano", "Whitfield", "Nakamura", "Adeyemi", "Brennan",
        "Valdez", "Kowalski", "Hargrove", "Iyer", "Montrose", "Delacroix", "Sato", "Ferreira",
        "Abernathy", "Mbeki", "Larkspur", "Quintero", "Halloway", "Petrov", "Ashdown", "Moreau",
        "Tanaka", "Oyelaran", "Fairbanks", "Rasmussen", "Calloway", "Esposito"]
ACCT_A = ["Ardent", "Bluepeak", "Cobalt", "Driftwood", "Ember", "Fernhill", "Granite", "Harbor",
          "Ironleaf", "Juniper", "Kestrel", "Larchmont", "Mesa", "Northvale", "Oakline", "Pinecrest",
          "Quarry", "Redwing", "Silverbirch", "Tidewater", "Umber", "Vantage", "Westbrook", "Yarrow",
          "Zephyr", "Alderwood", "Brightwater", "Cedarline", "Duskfield", "Evergale"]
ACCT_B = ["Logistics", "Health Systems", "Manufacturing", "Energy", "Retail Group", "Financial",
          "Telecom", "Foods", "Aerospace", "Materials", "Pharma", "Mobility", "Insurance", "Labs",
          "Utilities", "Freight", "Biotech", "Media", "Hospitality", "Robotics"]
INDUSTRY = {"Logistics": "Transportation", "Health Systems": "Healthcare", "Manufacturing": "Industrial",
            "Energy": "Energy", "Retail Group": "Retail", "Financial": "Financial Services",
            "Telecom": "Telecom", "Foods": "Consumer", "Aerospace": "Industrial", "Materials": "Industrial",
            "Pharma": "Healthcare", "Mobility": "Transportation", "Insurance": "Financial Services",
            "Labs": "Healthcare", "Utilities": "Energy", "Freight": "Transportation", "Biotech": "Healthcare",
            "Media": "Telecom", "Hospitality": "Consumer", "Robotics": "Industrial"}

REGIONS = ["West", "Central", "East", "International"]
REGION_WEIGHT = {"West": 0.30, "Central": 0.20, "East": 0.32, "International": 0.18}
STAGES = ["Prospect", "Qualify", "Propose", "Commit"]
# Conditional probability of advancing from each stage to the next (last = Commit -> Won).
ADVANCE = {"West": [0.56, 0.61, 0.62, 0.80], "Central": [0.55, 0.60, 0.60, 0.78],
           "East": [0.57, 0.62, 0.61, 0.79], "International": [0.52, 0.57, 0.56, 0.72]}


def qstart(year: int, q: int) -> dt.date:
    return dt.date(year, 3 * (q - 1) + 1, 1)


def qend(year: int, q: int) -> dt.date:
    if q == 4:
        return dt.date(year, 12, 31)
    return qstart(year, q + 1) - dt.timedelta(days=1)


def quarter_of(d: dt.date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def person(rng, used: set) -> str:
    while True:
        n = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if n not in used:
            used.add(n)
            return n


def write_table(con, name: str, cols: list[str], rows: list[tuple], csv_dir: Path):
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"CREATE TABLE {name} ({', '.join(cols)})")
    con.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(cols))})", rows)
    with open(csv_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c.split()[0] for c in cols])
        w.writerows(rows)


# --------------------------------------------------------------------------- enterprise

def gen_enterprise(rng, con, csv_dir):
    used_names: set = set()
    # Accounts
    acct_names = set()
    accounts = []
    while len(accounts) < 600:
        a, b = rng.choice(ACCT_A), rng.choice(ACCT_B)
        nm = f"{a} {b}"
        if nm in acct_names:
            continue
        acct_names.add(nm)
        region = rng.choice(REGIONS, p=[REGION_WEIGHT[r] for r in REGIONS])
        seg = rng.choice(["Enterprise", "Mid-Market", "Strategic"], p=[0.5, 0.38, 0.12])
        accounts.append((len(accounts) + 1, nm, region, INDUSTRY[b], seg))
    write_table(con, "accounts", ["account_id INTEGER", "name TEXT", "region TEXT", "industry TEXT", "segment TEXT"],
                accounts, csv_dir)
    acct_by_region = {r: [a for a in accounts if a[2] == r] for r in REGIONS}

    # Reps
    reps = []
    for i in range(120):
        region = REGIONS[i % 4] if i < 100 else rng.choice(REGIONS, p=[REGION_WEIGHT[r] for r in REGIONS])
        hire = dt.date(2019, 1, 1) + dt.timedelta(days=int(rng.integers(0, 365 * 7 + 150)))
        hire = min(hire, dt.date(2026, 7, 1))
        quota = int(round(rng.normal(2_400_000, 250_000), -4))
        reps.append([i + 1, person(rng, used_names), region, hire.isoformat(), quota, "active", None])
    # ~13% annual attrition over the last 12 months
    for r in rng.choice(len(reps), 16, replace=False):
        left = dt.date(2025, 10, 1) + dt.timedelta(days=int(rng.integers(0, 360)))
        if left.isoformat() > reps[r][3]:
            reps[r][5], reps[r][6] = "departed", left.isoformat()
    reps = [tuple(r) for r in reps]
    write_table(con, "reps", ["rep_id INTEGER", "name TEXT", "region TEXT", "hire_date TEXT", "annual_quota REAL",
                              "status TEXT", "departure_date TEXT"], reps, csv_dir)
    reps_by_region = {r: [x for x in reps if x[2] == r] for r in REGIONS}

    def deal_features():
        exec_sponsor = int(rng.random() < 0.62)
        competitor = int(rng.random() < 0.38)
        discount = float(np.clip(rng.normal(12, 6), 0, 35).round(1))
        pushes = int(rng.poisson(0.7))
        idle = int(np.clip(rng.gamma(2.0, 6.5), 0, 90))
        return exec_sponsor, competitor, discount, pushes, idle

    def loss_lift(es, comp, disc, pushes, idle):
        # Log-odds shift on each stage advance; drives the risk model that analytics later re-learns.
        return -0.55 * (1 - es) - 0.45 * comp - 0.025 * max(disc - 12, 0) - 0.35 * pushes - 0.03 * max(idle - 10, 0) + 0.45

    def logit(p):
        return np.log(p / (1 - p))

    def sig(x):
        return 1 / (1 + np.exp(-x))

    # Historical deals: created 2023-07 .. 2026-09, closed 2023-10 .. 2026-10-07
    history = []
    did = 0
    quarters = [(y, q) for y in (2023, 2024, 2025, 2026) for q in (1, 2, 3, 4)
                if (2023, 3) <= (y, q) <= (2026, 4)]
    for (y, q) in quarters:
        for region in REGIONS:
            n_new = int(rng.normal(400 * REGION_WEIGHT[region] * 4, 18))
            growth = 1.0 + 0.025 * (quarters.index((y, q)))
            adv = list(ADVANCE[region])
            if region == "International" and y == 2026:
                adv = [a - 0.05 for a in adv]  # 2026 softness in International
            for _ in range(n_new):
                created = qstart(y, q) + dt.timedelta(days=int(rng.integers(0, 89)))
                if created >= AS_OF:
                    continue
                es, comp, disc, pushes, idle = deal_features()
                shift = loss_lift(es, comp, disc, pushes, idle)
                stage_reached = 0
                won = False
                for s in range(4):
                    p = sig(logit(adv[s]) + shift * (0.4 if s < 2 else 1.0))
                    if rng.random() < p:
                        if s == 3:
                            won = True
                        else:
                            stage_reached = s + 1
                    else:
                        break
                else:
                    stage_reached = 3
                cycle = int(np.clip(rng.normal(95, 30), 30, 210))
                orig_close = created + dt.timedelta(days=cycle)
                slip = int(rng.random() < 0.18) * int(rng.integers(20, 75))
                closed = orig_close + dt.timedelta(days=slip + pushes * 7)
                if closed >= AS_OF:
                    continue  # still open; open pipeline is generated separately
                amt = float(round(rng.lognormal(np.log(330_000 * growth), 0.85), -3))
                acct = acct_by_region[region][int(rng.integers(len(acct_by_region[region])))]
                rep = reps_by_region[region][int(rng.integers(len(reps_by_region[region])))]
                did += 1
                history.append((did, acct[0], rep[0], region, created.isoformat(), orig_close.isoformat(),
                                closed.isoformat(), amt, STAGES[stage_reached], "won" if won else "lost",
                                es, comp, disc, pushes, idle))
    write_table(con, "deals_history",
                ["deal_id INTEGER", "account_id INTEGER", "rep_id INTEGER", "region TEXT", "created_date TEXT",
                 "original_close_date TEXT", "closed_date TEXT", "amount REAL", "max_stage TEXT", "outcome TEXT",
                 "exec_sponsor INTEGER", "competitor INTEGER", "discount_pct REAL", "close_date_pushes INTEGER",
                 "days_since_activity INTEGER"], history, csv_dir)

    # Open pipeline as of AS_OF: Q4 2026 and Q1 2027 close dates
    open_rows = []
    active_reps = {r: [x for x in reps_by_region[r] if x[5] == "active"] for r in REGIONS}
    mix = {"Q4": {"Prospect": 175, "Qualify": 185, "Propose": 150, "Commit": 105},
           "Q1": {"Prospect": 260, "Qualify": 170, "Propose": 85, "Commit": 22}}
    for qlabel, per_stage in mix.items():
        for stage, n in per_stage.items():
            for _ in range(n):
                region = rng.choice(REGIONS, p=[REGION_WEIGHT[r] for r in REGIONS])
                if qlabel == "Q4":
                    close = AS_OF + dt.timedelta(days=int(rng.integers(3, 84)))
                else:
                    close = dt.date(2027, 1, 1) + dt.timedelta(days=int(rng.integers(0, 89)))
                es, comp, disc, pushes, idle = deal_features()
                stage_age = int(np.clip(rng.gamma(2.2, 11), 2, 120))
                amt = float(round(rng.lognormal(np.log(330_000 * 1.33), 0.85), -3))
                acct = acct_by_region[region][int(rng.integers(len(acct_by_region[region])))]
                rep = active_reps[region][int(rng.integers(len(active_reps[region])))]
                did += 1
                created = close - dt.timedelta(days=int(rng.integers(40, 160)))
                open_rows.append((did, acct[0], acct[1], rep[0], rep[1], region, stage, amt, close.isoformat(),
                                  min(created, AS_OF).isoformat(), es, comp, disc, pushes, idle, stage_age))
    # A handful of large, visibly stalled Commit/Propose deals so the risk story is crisp.
    stalled = [("Granite Aerospace", "West", "Commit", 4_850_000, 41, 3, 0, 1, 24.0),
               ("Tidewater Health Systems", "East", "Commit", 3_920_000, 33, 2, 0, 1, 21.5),
               ("Northvale Energy", "International", "Propose", 3_400_000, 46, 2, 1, 1, 18.0),
               ("Kestrel Financial", "Central", "Commit", 2_760_000, 28, 2, 0, 0, 26.0),
               ("Cobalt Robotics", "East", "Propose", 2_580_000, 38, 1, 0, 1, 19.5)]
    for nm, region, stage, amt, idle, pushes, es, comp, disc in stalled:
        acct = next((a for a in accounts if a[1] == nm), None)
        if acct is None:
            acct = (len(accounts) + 1, nm, region, INDUSTRY[nm.split(" ", 1)[1]], "Strategic")
            accounts.append(acct)
            con.execute("INSERT INTO accounts VALUES (?,?,?,?,?)", acct)
        rep = active_reps[region][0]
        did += 1
        close = AS_OF + dt.timedelta(days=int(rng.integers(20, 70)))
        open_rows.append((did, acct[0], acct[1], rep[0], rep[1], region, stage, float(amt), close.isoformat(),
                          (AS_OF - dt.timedelta(days=150)).isoformat(), es, comp, disc, pushes, idle,
                          int(rng.integers(45, 80))))
    write_table(con, "open_pipeline",
                ["deal_id INTEGER", "account_id INTEGER", "account_name TEXT", "rep_id INTEGER", "rep_name TEXT",
                 "region TEXT", "stage TEXT", "amount REAL", "close_date TEXT", "created_date TEXT",
                 "exec_sponsor INTEGER", "competitor INTEGER", "discount_pct REAL", "close_date_pushes INTEGER",
                 "days_since_activity INTEGER", "days_in_stage INTEGER"], open_rows, csv_dir)
    # rewrite accounts csv (stalled accounts may have been added)
    write_table(con, "accounts", ["account_id INTEGER", "name TEXT", "region TEXT", "industry TEXT", "segment TEXT"],
                accounts, csv_dir)

    # Quarterly P&L by region (recurring + new business). New business = won bookings in quarter.
    won_by_q = {}
    for h in history:
        if h[9] == "won":
            yq = quarter_of(dt.date.fromisoformat(h[6]))
            won_by_q[(yq, h[3])] = won_by_q.get((yq, h[3]), 0.0) + h[7]
    pnl = []
    base = {"West": 128e6, "Central": 86e6, "East": 137e6, "International": 77e6}
    for (y, q) in [(y, q) for y in (2023, 2024, 2025, 2026) for q in (1, 2, 3, 4)]:
        if (y, q) < (2023, 4) or (y, q) > (2026, 3):
            continue
        for region in REGIONS:
            g = 0.016 + rng.normal(0, 0.004)
            if region == "International" and y == 2026:
                g = 0.004 + rng.normal(0, 0.003)
            base[region] *= (1 + g)
            rec = round(base[region], -3)
            new = round(won_by_q.get(((y, q), region), 0.0), -3)
            pnl.append((y, q, region, rec, new, rec + new))
    write_table(con, "revenue_quarterly", ["year INTEGER", "quarter INTEGER", "region TEXT",
                                           "recurring_revenue REAL", "new_business_revenue REAL", "revenue REAL"],
                pnl, csv_dir)

    # Company-level cost lines per quarter + Q4 2026 approved budget
    costs = []
    for (y, q) in sorted({(p[0], p[1]) for p in pnl}):
        rev = sum(p[5] for p in pnl if (p[0], p[1]) == (y, q))
        cogs = rev * float(np.clip(rng.normal(0.312, 0.006), 0.29, 0.33))
        costs.append((y, q, "actual", round(cogs, -3), round(rev * 0.235 + rng.normal(0, 2e6), -3),
                      round(rev * 0.078 + rng.normal(0, 1e6), -3), round(rev * 0.168 + rng.normal(0, 1.5e6), -3),
                      round(rev * 0.081 + rng.normal(0, 1e6), -3)))
    last_rev = sum(p[5] for p in pnl if (p[0], p[1]) == (2026, 3))
    costs.append((2026, 4, "budget", None, round(last_rev * 0.236, -3), round(last_rev * 0.082, -3),
                  round(last_rev * 0.171, -3), round(last_rev * 0.080, -3)))
    write_table(con, "costs_quarterly", ["year INTEGER", "quarter INTEGER", "kind TEXT", "cogs REAL",
                                         "opex_sales REAL", "opex_marketing REAL", "opex_rnd REAL", "opex_ga REAL"],
                costs, csv_dir)

    # Annual plan by region: FY2026 = FY2025 actual x 1.09 (stretch); FY2027 new-bookings target
    plans = []
    for region in REGIONS:
        fy24 = sum(p[5] for p in pnl if p[0] == 2024 and p[2] == region)
        fy24_new = sum(p[4] for p in pnl if p[0] == 2024 and p[2] == region)
        plans.append((2025, region, round(fy24 * 1.08, -5), round(fy24_new * 1.12, -5)))
        fy25 = sum(p[5] for p in pnl if p[0] == 2025 and p[2] == region)
        fy25_new = sum(p[4] for p in pnl if p[0] == 2025 and p[2] == region)
        plans.append((2026, region, round(fy25 * 1.09, -5), round(fy25_new * 1.12, -5)))
        plans.append((2027, region, None, round(fy25_new * 1.12, -5)))
    write_table(con, "plan_targets", ["year INTEGER", "region TEXT", "revenue_plan REAL", "new_bookings_plan REAL"],
                plans, csv_dir)

    # Pipeline snapshots taken on the same calendar day in prior years (for coverage comparisons)
    snaps = []
    for yr, scale in ((2024, 0.86), (2025, 1.0)):
        for region in REGIONS:
            for stage, n in mix["Q4"].items():
                amt = n * REGION_WEIGHT[region] * 330_000 * (1 + 0.1 * (yr - 2024)) * float(rng.normal(1.0, 0.06)) * 1.25 * scale
                snaps.append((f"{yr}-10-08", yr, 4, region, stage, round(amt, -3)))
    write_table(con, "pipeline_snapshots", ["snapshot_date TEXT", "year INTEGER", "quarter INTEGER", "region TEXT",
                                            "stage TEXT", "amount REAL"], snaps, csv_dir)

    # Candidate sales-center cities
    sites = [  # city, loaded rep cost, sales talent pool, cost per seat/yr, accounts within 500mi, open pipeline nearby, attrition
        ("Phoenix", 148_000, 21_400, 9_800, 0, 0.0, 0.142),
        ("Denver", 162_000, 18_900, 12_600, 0, 0.0, 0.118),
        ("Albuquerque", 131_000, 6_300, 7_400, 0, 0.0, 0.127),
        ("Austin", 171_000, 24_800, 13_900, 0, 0.0, 0.151),
        ("Salt Lake City", 145_000, 11_200, 9_100, 0, 0.0, 0.109),
    ]
    near = {"Phoenix": ("West", 0.34), "Denver": ("Central", 0.42), "Albuquerque": ("Central", 0.21),
            "Austin": ("Central", 0.37), "Salt Lake City": ("West", 0.26)}
    out = []
    for c in sites:
        reg, share = near[c[0]]
        n_acc = int(len([a for a in accounts if a[2] == reg]) * share)
        pipe = sum(o[7] for o in open_rows if o[5] == reg) * share
        out.append((c[0], c[1], c[2], c[3], n_acc, round(pipe, -3), c[6]))
    write_table(con, "site_candidates", ["city TEXT", "rep_cost REAL", "talent_pool INTEGER", "cost_per_seat REAL",
                                         "nearby_accounts INTEGER", "nearby_pipeline REAL", "attrition_rate REAL"],
                out, csv_dir)


# --------------------------------------------------------------------------- university

COLLEGES = ["Engineering", "Medicine & Health Sciences", "Arts & Sciences", "Business", "Education",
            "Agriculture & Environment"]
SPONSORS = {  # name: (type, share of awards, success rate, volatility)
    "NIH": ("Federal", 0.36, 0.21, 0.10),
    "NSF": ("Federal", 0.19, 0.26, 0.08),
    "DOE": ("Federal", 0.12, 0.30, 0.12),
    "DoD": ("Federal", 0.11, 0.28, 0.14),
    "Industry": ("Industry", 0.14, 0.42, 0.20),
    "Foundations": ("Private", 0.08, 0.35, 0.09),
}
COLLEGE_SPONSOR_AFFINITY = {
    "Engineering": {"NSF": 3, "DOE": 3, "DoD": 3, "Industry": 3, "NIH": 0.5, "Foundations": 1},
    "Medicine & Health Sciences": {"NIH": 6, "Industry": 1.5, "Foundations": 1.5, "NSF": 0.4, "DOE": 0.2, "DoD": 0.6},
    "Arts & Sciences": {"NSF": 3, "NIH": 1.5, "DOE": 1, "Foundations": 1.5, "DoD": 0.5, "Industry": 0.5},
    "Business": {"Industry": 2, "Foundations": 1.5, "NSF": 0.5, "NIH": 0.1, "DOE": 0.1, "DoD": 0.1},
    "Education": {"Foundations": 2, "NSF": 1.2, "NIH": 0.4, "DOE": 0.1, "DoD": 0.1, "Industry": 0.2},
    "Agriculture & Environment": {"DOE": 2, "NSF": 1.5, "Industry": 1.2, "Foundations": 1, "NIH": 0.3, "DoD": 0.2},
}
UNI_STAGES = ["Drafting", "Submitted", "Under Review", "Awarded pending"]


def gen_university(rng, con, csv_dir):
    used: set = set()
    pis = [(i + 1, "Dr. " + person(rng, used), COLLEGES[i % 6]) for i in range(180)]
    write_table(con, "faculty_pis", ["pi_id INTEGER", "name TEXT", "college TEXT"], pis, csv_dir)

    # Award history FY24..FY27 (FY = July..June; FY27 started 2026-07-01)
    awards = []
    gid = 0
    for fy in (2022, 2023, 2024, 2025, 2026, 2027):
        n = {2022: 300, 2023: 320, 2024: 340, 2025: 355, 2026: 370, 2027: 235}[fy]
        for _ in range(n):
            pi = pis[int(rng.integers(len(pis)))]
            aff = COLLEGE_SPONSOR_AFFINITY[pi[2]]
            names = list(SPONSORS)
            w = np.array([SPONSORS[s][1] * aff[s] for s in names])
            sponsor = names[int(rng.choice(len(names), p=w / w.sum()))]
            dur = int(rng.choice([1, 2, 3, 4, 5], p=[0.12, 0.18, 0.32, 0.18, 0.20]))
            annual = float(round(rng.lognormal(np.log(220_000), 0.75), -3))
            if sponsor == "NIH":
                annual *= 1.35
            gid += 1
            awards.append((gid, pi[2], sponsor, SPONSORS[sponsor][0], pi[1], fy, dur, round(annual, -3),
                           round(annual * dur, -3)))
    write_table(con, "grant_awards", ["award_id INTEGER", "college TEXT", "sponsor TEXT", "sponsor_type TEXT",
                                      "pi_name TEXT", "start_fy INTEGER", "duration_years INTEGER",
                                      "annual_amount REAL", "total_amount REAL"], awards, csv_dir)

    # Historical proposal outcomes by stage (for stage success rates)
    props = []
    pid = 0
    for fy in (2024, 2025, 2026):
        for _ in range(900):
            pi = pis[int(rng.integers(len(pis)))]
            aff = COLLEGE_SPONSOR_AFFINITY[pi[2]]
            names = list(SPONSORS)
            w = np.array([SPONSORS[s][1] * aff[s] for s in names])
            sponsor = names[int(rng.choice(len(names), p=w / w.sum()))]
            base = SPONSORS[sponsor][2]
            # Stage advance: Drafting->Submitted .82, Submitted->Under Review .78, Review->Pending x, Pending->Award .96
            p_rev = min(0.95, base / (0.82 * 0.78 * 0.96) * 0.82 * 0.78)
            reached = 0
            awarded = False
            for i, p in enumerate([0.82, 0.78, p_rev / (0.82 * 0.78), 0.96]):
                if rng.random() < p:
                    if i == 3:
                        awarded = True
                    else:
                        reached = i + 1
                else:
                    break
            pid += 1
            props.append((pid, fy, pi[2], sponsor, UNI_STAGES[reached], "awarded" if awarded else "not awarded"))
    write_table(con, "proposal_history", ["proposal_id INTEGER", "fy INTEGER", "college TEXT", "sponsor TEXT",
                                          "max_stage TEXT", "outcome TEXT"], props, csv_dir)

    # Open proposal pipeline (expected start in FY27 or FY28)
    pipe = []
    for stage, n in (("Drafting", 150), ("Submitted", 260), ("Under Review", 240), ("Awarded pending", 70)):
        for _ in range(n):
            pi = pis[int(rng.integers(len(pis)))]
            aff = COLLEGE_SPONSOR_AFFINITY[pi[2]]
            names = list(SPONSORS)
            w = np.array([SPONSORS[s][1] * aff[s] for s in names])
            sponsor = names[int(rng.choice(len(names), p=w / w.sum()))]
            annual = float(round(rng.lognormal(np.log(240_000), 0.75), -3))
            start = AS_OF + dt.timedelta(days=int(rng.integers(20 if stage != "Awarded pending" else 5, 200 if stage != "Drafting" else 330)))
            fy = start.year + (1 if start.month >= 7 else 0)
            pid += 1
            pipe.append((pid, pi[2], sponsor, pi[1], stage, annual, start.isoformat(), fy,
                         int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.35, 0.15, 0.2]))))
    write_table(con, "grant_pipeline", ["proposal_id INTEGER", "college TEXT", "sponsor TEXT", "pi_name TEXT",
                                        "stage TEXT", "annual_amount REAL", "expected_start TEXT", "start_fy INTEGER",
                                        "duration_years INTEGER"], pipe, csv_dir)

    # Enrollment funnel by segment for Fall 2024..2026, and Fall 2027 admit plan
    funnel = []
    seg = {"In-state first-year": (21000, 0.71, 0.38), "Out-of-state first-year": (16500, 0.62, 0.19),
           "International": (6200, 0.44, 0.22), "Transfer": (5100, 0.66, 0.52)}
    for yr in (2024, 2025, 2026):
        for s, (apps, admit, yld) in seg.items():
            a = int(apps * (1 + 0.04 * (yr - 2024)) * rng.normal(1, 0.02))
            ad = int(a * admit * rng.normal(1, 0.015))
            trend = {"In-state first-year": 0.006, "Out-of-state first-year": -0.008,
                     "International": -0.012, "Transfer": 0.004}[s]
            y = yld + trend * (yr - 2024) + rng.normal(0, 0.004)
            dep = int(ad * (y + 0.035))
            en = int(ad * y)
            funnel.append((yr, s, a, ad, dep, en))
    write_table(con, "enrollment_funnel", ["fall_year INTEGER", "segment TEXT", "applications INTEGER",
                                           "admits INTEGER", "deposits INTEGER", "enrolled INTEGER"], funnel, csv_dir)
    plan = []
    for s in seg:
        last = [f for f in funnel if f[0] == 2026 and f[1] == s][0]
        plan.append((2027, s, int(last[3] * 1.03), int(last[5] * 1.05)))
    write_table(con, "enrollment_plan", ["fall_year INTEGER", "segment TEXT", "planned_admits INTEGER",
                                         "enrollment_target INTEGER"], plan, csv_dir)

    # Candidate research-center sites
    sites = [  # site, build+fit-out cost, faculty within 15 min, aligned open proposals $, industry partners nearby, months to open
        ("North Campus Innovation Park", 56_000_000, 142, 0.0, 11, 18),
        ("Downtown Health District", 88_000_000, 118, 0.0, 14, 26),
        ("Mesa Ridge Technology Corridor", 52_000_000, 61, 0.0, 17, 16),
        ("Riverside Federal Lab Annex", 47_000_000, 74, 0.0, 6, 14),
    ]
    focus = {"North Campus Innovation Park": ["Engineering", "Arts & Sciences"],
             "Downtown Health District": ["Medicine & Health Sciences"],
             "Mesa Ridge Technology Corridor": ["Engineering", "Business"],
             "Riverside Federal Lab Annex": ["Agriculture & Environment", "Engineering"]}
    out = []
    for s in sites:
        aligned = sum(p[5] for p in pipe if p[1] in focus[s[0]])
        out.append((s[0], s[1], s[2], round(aligned, -3), s[4], s[5], ", ".join(focus[s[0]])))
    write_table(con, "research_sites", ["site TEXT", "capital_cost REAL", "faculty_nearby INTEGER",
                                        "aligned_pipeline REAL", "industry_partners INTEGER", "months_to_open INTEGER",
                                        "focus_colleges TEXT"], out, csv_dir)

    # Operating budget (for context; ~$900M)
    budget = [("Instruction", 296e6), ("Research", 309e6), ("Academic support", 92e6), ("Student services", 61e6),
              ("Institutional support", 84e6), ("Facilities", 58e6)]
    write_table(con, "operating_budget", ["category TEXT", "fy27_budget REAL"], budget, csv_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_dir = DATA_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    rng = np.random.default_rng(args.seed)
    gen_enterprise(rng, con, csv_dir)
    gen_university(rng, con, csv_dir)
    con.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    con.executemany("INSERT INTO meta VALUES (?,?)", [("seed", str(args.seed)), ("as_of", AS_OF.isoformat()),
                                                      ("note", "All data synthetic. Fictional companies.")])
    con.commit()
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t:22s} {n:7d}")
    con.close()


if __name__ == "__main__":
    main()
