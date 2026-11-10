"""Build data/leaderboard-adjust.json: two Terminal-Bench 2.1 leaderboard rows as
published, and as atif-scan sees them.

Source: the public Space evalstate/tb21-annotated (local checkout
~/source/tb21-annotated/site/data/leaderboard.js), which is the leaderboard's
own trial files scanned with atif-scan. Counts only.

Run: python3 analysis/leaderboard-adjust/extract.py
"""
import json
import re
from pathlib import Path

SITE = Path.home() / "source/tb21-annotated/site/data/leaderboard.js"
DECK = Path(__file__).resolve().parent.parent.parent
ROWS = ["06-claude-code--fable-5--xhigh", "11-codex--gpt-5-6-terra--max"]
# List price per million output tokens (pricing catalogue 2026-11-05.1). Fable 5 lists at
# $10 / $1 / $50 (input / cache read / output); Opus 4.8 at $5 / $0.50 / $25.
LIST_OUTPUT = {"Fable 5": 50.0}

board = json.loads(re.search(r"=\s*(\{.*\})\s*;?\s*$", SITE.read_text(), re.S).group(1))
out = dict(generated=board["generated"], scanner=board["scanner"]["tag"], source="https://evalstate-tb21-annotated.static.hf.space/index.html", rows=[])
for row in board["rows"]:
    if row["id"] not in ROWS:
        continue
    lb, scan = row["lb"], row["scan"]
    holes = scan["holes"]
    served = holes["models"]["served"]
    other = [m for m in served if not m["configured"]]
    out["rows"].append(dict(
        id=row["id"], model=row["model"], effort=row["effort"], agent=row["agent"], date=row["date"],
        trials=scan["trials"],
        leaderboard=dict(accuracy=lb["accuracy"], cost=lb["cost"]),
        rewarded=scan["rewarded"],
        adjusted=dict(rewarded=scan["adjusted"]["rewarded"], accuracy=scan["adjusted"]["accuracy"], fallback=scan["adjusted"]["fallback"], leads=scan["adjusted"]["leads"]),
        categories=scan["categories"],
        hiddenOutputShare=holes["tokens"]["mix"]["hidden"],
        otherModel=dict(name=other[0]["model"], trials=other[0]["trials"], stepShare=other[0]["share"]) if other else None,
        webTrials=holes["webTrials"], webUnresolved=holes["webUnresolved"], noTrajectory=holes["noTrajectory"],
        cost=dict(leaderboard=holes["lbCost"], recorded=holes["cost"]["recorded"], estimate=holes["cost"]["estimate"], uncosted=holes["uncosted"],
                  # per-million-token rates that best fit the cost each trial file records
                  fitted=holes["cost"]["rates"], listOutput=LIST_OUTPUT.get(row["model"])),
    ))
out["rows"].sort(key=lambda r: ROWS.index(r["id"]))
(DECK / "data/leaderboard-adjust.json").write_text(json.dumps(out, indent=1))
for r in out["rows"]:
    print(r["id"], r["leaderboard"], r["rewarded"], r["adjusted"], r["otherModel"], r["hiddenOutputShare"], r["webTrials"], r["cost"])
