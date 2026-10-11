"""Run generated code in a locked-down subprocess and report exactly what happened.

Each run gets a fresh directory under run/sandbox/<id>/:
  work/        the only writable place (Landlock); the code runs here
  db/atlas.db  a read-only copy of the synthetic database (mode 0444, directory not writable under Landlock),
               reachable from the code as work/atlas.db (symlink)
Limits: 10 s wall clock (process group killed), 10 s CPU, 2 GB address space, 64 MB per file, no network
(seccomp refuses non-AF_UNIX sockets; Landlock denies TCP bind/connect). See atlas/sandbox_boot.py.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from atlas.config import DB_PATH, ROOT

BASE = ROOT / "run" / "sandbox"
BOOT = ROOT / "atlas" / "sandbox_boot.py"
PY = sys.executable
TIMEOUT_S = 10
MEM_MB = 2048
METHOD = ("subprocess in a temp dir · read-only DB copy · 10 s timeout · 2 GB memory cap · "
          "no network (seccomp: no IP sockets; Landlock: writes only in the temp dir, no TCP)")

SQL_RUNNER = r'''
import json, sqlite3, sys
sql = open("query.sql").read()
con = sqlite3.connect("file:atlas.db?mode=ro", uri=True)
cur = con.execute(sql)
cols = [d[0] for d in cur.description] if cur.description else []
rows = cur.fetchmany(500)
print(json.dumps({"columns": cols, "rows": rows}, default=str))
'''


def _prepare() -> Path:
    run = BASE / f"{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    (run / "work").mkdir(parents=True)
    (run / "db").mkdir()
    shutil.copy2(DB_PATH, run / "db" / "atlas.db")
    os.chmod(run / "db" / "atlas.db", 0o444)
    os.chmod(run / "db", 0o555)
    os.symlink(run / "db" / "atlas.db", run / "work" / "atlas.db")
    return run


def cleanup(keep: int = 40):
    if not BASE.exists():
        return
    runs = sorted(BASE.iterdir())
    for d in runs[:-keep]:
        try:
            os.chmod(d / "db", 0o755)
        except OSError:
            pass
        shutil.rmtree(d, ignore_errors=True)


def run(lang: str, code: str, files: dict[str, str] | None = None, entry: list[str] | None = None,
        timeout: float = TIMEOUT_S) -> dict:
    """Execute `code` (lang 'sql' or 'python'). `files` are extra files written next to it (e.g. a module under
    test). `entry` overrides the command (e.g. ['-m', 'pytest', '-q']). Returns what really happened."""
    cleanup()
    run_dir = _prepare()
    work = run_dir / "work"
    for name, text in (files or {}).items():
        (work / name).write_text(text)
    if lang == "sql":
        (work / "query.sql").write_text(code)
        (work / "_run_sql.py").write_text(SQL_RUNNER)
        cmd = [PY, "-s", "_run_sql.py"]
    else:
        (work / "main.py").write_text(code)
        cmd = [PY, "-s"] + (entry or ["main.py"])
    env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "MPLCONFIGDIR": str(work / ".mpl"), "MPLBACKEND": "Agg",
           "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "ATLAS_DB": str(work / "atlas.db"), "PYTHONPATH": str(ROOT)}  # Atlas analytics importable (read-only)
    boot = [PY, "-E", "-s", str(BOOT), str(work), str(MEM_MB), str(int(timeout)), "--"] + cmd
    before = set()
    t = time.perf_counter()
    p = subprocess.Popen(boot, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
    ms = (time.perf_counter() - t) * 1000
    stdout = out.decode("utf-8", "replace")
    stderr = err.decode("utf-8", "replace")
    res = {"executed": True, "lang": lang, "exit_code": p.returncode, "ok": p.returncode == 0 and not timed_out,
           "timed_out": timed_out, "ms": round(ms, 1), "stdout": stdout[-6000:], "stderr": stderr[-4000:],
           "table": None, "images": [], "files": [], "run_id": run_dir.name, "method": METHOD}
    if timed_out:
        res["stderr"] = (stderr[-3500:] + f"\nTimeoutError: killed after {timeout:.0f} s (sandbox limit)").strip()
    if lang == "sql" and res["ok"]:
        try:
            res["table"] = json.loads(stdout.strip().splitlines()[-1])
            res["stdout"] = ""
        except Exception:  # noqa: BLE001
            pass
    # Artifacts the code produced: CSV tables and PNG charts.
    for f in sorted(work.iterdir()):
        if f.name in before or f.name.startswith((".", "_")) or f.name in ("main.py", "query.sql", "atlas.db") or f.is_dir():
            continue
        res["files"].append({"name": f.name, "bytes": f.stat().st_size})
        if f.suffix == ".csv" and res["table"] is None:
            rows = list(csv.reader(io.StringIO(f.read_text(errors="replace"))))
            if rows:
                res["table"] = {"columns": rows[0], "rows": rows[1:501], "file": f.name, "row_count": len(rows) - 1}
        elif f.suffix == ".png" and len(res["images"]) < 2 and f.stat().st_size < 3_000_000:
            res["images"].append({"name": f.name, "b64": base64.b64encode(f.read_bytes()).decode()})
    if res["table"] is None and res["ok"]:
        res["table"] = _table_from_stdout(stdout)
    return res


def _table_from_stdout(text: str) -> dict | None:
    """If the program printed a JSON list of records or CSV-ish lines, show it as a table too."""
    s = text.strip()
    last = s.splitlines()[-1].strip() if s else ""
    if last.startswith("[") and last.endswith("]"):
        s = last
    if s.startswith("[") and s.endswith("]"):
        try:
            data = json.loads(s)
            if data and isinstance(data[0], dict):
                cols = list(data[0].keys())
                return {"columns": cols, "rows": [[r.get(c) for c in cols] for r in data[:500]]}
        except Exception:  # noqa: BLE001
            return None
    return None
