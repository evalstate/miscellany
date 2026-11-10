"""Add public-leaderboard runs to data/run-costs.json as `publicRuns`.

Source: the local mirror of public Terminal-Bench 2.1 runs on Harbor Hub
(hub-listing.json per job: per-trial reward, cost_usd, tokens, agent, model).
Cost is what each run recorded per trial, summed; it is not repriced, so it is
not on the same basis as the fast-agent runs' same-table cost.

Only counts are written: no job or trial identifiers.

Run: python3 analysis/run-costs/public_runs.py
"""
import collections
import json
import os
from pathlib import Path

MIRROR = Path.home() / ".cache/atif-scan/harbor"
DECK = Path(__file__).resolve().parent.parent.parent
# (agent, model) pairs to keep: the OpenAI model before the Sol line, and the
# public Codex run of the first Luna model.
KEEP = {("terminus-2", "gpt-5.5"): "Sol", ("codex", "gpt-5.5"): "Sol", ("codex", "gpt-5.6-luna"): "Luna"}

runs = []
for job in sorted(os.listdir(MIRROR)):
    listing = MIRROR / job / "hub-listing.json"
    if not listing.exists():
        continue
    rows = json.loads(listing.read_text()).get("rows") or []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r.get("agent_name"), r.get("model_name"))].append(r)
    for key, rs in groups.items():
        if key not in KEEP or len(rs) != 445 or len({r.get("task_name") for r in rs}) != 89:
            continue
        priced = [r["cost_usd"] for r in rs if r.get("cost_usd")]
        won = sum(1 for r in rs if (r.get("reward") or 0) >= 1)
        dates = sorted(r["started_at"][:10] for r in rs if r.get("started_at"))
        runs.append(dict(
            designation=KEEP[key], model=key[1], harness=key[0], month=dates[0][:7], date=dates[0], trials=len(rs),
            score=round(won / len(rs), 4), rewarded=won, costUsd=round(sum(priced), 2), costTrials=len(priced),
            outputTokens=sum(r.get("output_tokens") or 0 for r in rs),
            costBasis="as recorded per trial by the run, summed; not repriced",
            source="public Terminal-Bench 2.1 run on Harbor Hub (local mirror)",
        ))

runs.sort(key=lambda r: (r["date"], r["harness"]))
path = DECK / "data/run-costs.json"
data = json.loads(path.read_text())
data["publicRuns"] = runs
path.write_text(json.dumps(data, indent=1))
for r in runs:
    print(r["date"], r["harness"], r["model"], f"{r['rewarded']}/{r['trials']}", r["score"], "$%.2f" % r["costUsd"], f"({r['costTrials']} priced)")
