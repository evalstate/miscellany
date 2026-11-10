"""Agent time in the trace corpus: how long the agents were at work, summed.

For each unique trial in the 8 Nov census, take the agent's working time:
  1. result.json `agent_execution` started/finished, when the trial has one;
  2. otherwise the first and last timestamp in the trajectory file (read from
     the first and last 256 KB only).
Trials with neither are counted as unknown, not as zero.

Reads private trace locations; writes only totals to data/corpus-hours.json.

Run: python3 analysis/corpus-hours/hours.py
"""
import collections
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

CENSUS = Path.home() / ".cache/atif-scan/reviews/talk-hunt-2026-11-08/census"
DECK = Path(__file__).resolve().parent.parent.parent
STAMP = re.compile(rb"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d")
CAP = 24 * 3600  # a single trial longer than a day is a clock or resume artefact


def parse(s: str) -> datetime:
    return datetime.fromisoformat(s[:19])


def seconds(path: str):
    d = os.path.dirname(path)
    for res in (os.path.join(d, "result.json"), os.path.join(os.path.dirname(d), "result.json")):
        if os.path.exists(res):
            try:
                ae = (json.load(open(res)).get("agent_execution") or {})
                if ae.get("started_at") and ae.get("finished_at"):
                    return "result", (parse(ae["finished_at"]) - parse(ae["started_at"])).total_seconds()
            except Exception:
                pass
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(262144)
            if size > 524288:
                f.seek(size - 262144)
            tail = f.read(262144) if size > 262144 else b""
        first = STAMP.search(head)
        last = STAMP.findall(tail or head)
        if first and last:
            return "trajectory", (parse(last[-1].decode()) - parse(first.group(0).decode())).total_seconds()
    except Exception:
        pass
    return "unknown", None


if __name__ == "__main__":
    ns = {}
    # the census's own model-name normaliser (its helper functions only, not its data load)
    helpers = "\n".join(l for l in (CENSUS / "agg.py").read_text().split("seen=set()")[0].split("\n") if not l.startswith("C="))
    exec(helpers, ns)
    rows, seen = [], set()
    for r in json.loads((CENSUS / "census.json").read_text()):
        key = (r["trial"], ns["norm"](r["model"]), r.get("steps"), r.get("tool_calls"))
        if key not in seen:
            seen.add(key)
            rows.append(r)
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(seconds, [r["rp"] for r in rows], chunksize=200))
    how = collections.Counter(k for k, _ in res)
    good = [min(s, CAP) for _, s in res if s is not None and s >= 0]
    capped = sum(1 for _, s in res if s is not None and s > CAP)
    total = sum(good)
    out = dict(
        trials=len(rows), timed=len(good), sources=dict(how), capped=capped,
        hours=round(total / 3600), days=round(total / 86400), years=round(total / (365.25 * 86400), 2),
        medianMinutes=round(sorted(good)[len(good) // 2] / 60, 1),
    )
    (DECK / "data/corpus-hours.json").write_text(json.dumps(out, indent=1))
    print(out)
