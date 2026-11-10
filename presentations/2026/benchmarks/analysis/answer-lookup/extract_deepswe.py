"""Build data/deepswe-attempts.json: answer-lookup attempts in the public DeepSWE v1.1 runs.

Source: the public Space evalstate/deepswe-annotated (local checkout: site/data/leaderboard.js
for the rows, build/rows-v4/<row>/report.json, atif-scan v0.8.23, for per-trial assessments).
DeepSWE sandboxes have no network, so these are attempts: the scan records the request or the
git command, not what came back. Counts only.

An attempt is an unexcused medium-or-higher match of
- deepswe.upstream_lookup ("Upstream task repository requested offline"), leaving out the
  actionlint task, whose generated documentation URLs were false positives in an earlier review;
- lookup.git_unreachable_objects ("Git objects outside the branch history searched").

Run: python3 analysis/answer-lookup/extract_deepswe.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path.home() / "source/deepswe-annotated"
DECK = Path(__file__).resolve().parent.parent.parent
SEV = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
EFFORTS = ["low", "medium", "high", "xhigh", "max"]

board = json.loads(re.search(r"=\s*(\{.*\})\s*;?\s*$", (REPO / "site/data/leaderboard.js").read_text(), re.S).group(1))
assert board["scanner"]["tag"] == "v0.8.23"
rows, total = [], Counter()
for row in board["rows"]:
    report = json.loads((REPO / "build/rows-v4" / row["id"] / "report.json").read_text())
    assert report["scanner_version"] == "0.8.23"
    c = Counter()
    for i in report["inputs"]:
        won = (i.get("reward") or 0) > 0
        ids = {a["id"] for a in i["assessments"]
               if a["status"] == "match" and not a.get("expected_by") and SEV.get(a["severity"], -1) >= SEV["medium"]}
        upstream = "deepswe.upstream_lookup" in ids and not i["task"].startswith("actionlint")
        git = "lookup.git_unreachable_objects" in ids
        c["trials"] += 1
        c["rewarded"] += won
        c["upstream"] += upstream
        c["git"] += git
        c["attempt"] += upstream or git
        c["attemptRewarded"] += (upstream or git) and won
    assert c["trials"] == row["scan"]["trials"]
    rows.append(dict(model=row["model"], effort=row["effort"], **c))
    total.update(c)

by_model = defaultdict(dict)
for r in rows:
    by_model[r["model"]][r["effort"]] = r
grid = [dict(model=m, cells=[by_model[m].get(e) and dict(effort=e, **{k: by_model[m][e][k] for k in ("trials", "attempt", "rewarded", "attemptRewarded")})
                             for e in EFFORTS])
        for m in by_model
        # every model with four or more effort settings, plus any single run where a fifth of trials try
        if sum(e in by_model[m] for e in EFFORTS) >= 4 or any(r["attempt"] * 5 >= r["trials"] for r in by_model[m].values())]
grid.sort(key=lambda g: -max(c["attempt"] / c["trials"] for c in g["cells"] if c))
out = dict(generated="2026-11-10", scanner="v0.8.23", source="https://evalstate-deepswe-annotated.static.hf.space",
           total=dict(runs=len(rows), **total), efforts=EFFORTS, grid=grid,
           rows=sorted(rows, key=lambda r: -r["attempt"]))
(DECK / "data/deepswe-attempts.json").write_text(json.dumps(out, indent=1))
print(out["total"])
for g in grid:
    print(f"{g['model']:18}", "  ".join(f"{c['attempt']:3}/{c['trials']} {100*c['attempt']//c['trials']:2}%" if c else "   -      " for c in g["cells"]))
for r in out["rows"][:8]:
    print(r)
t = out["total"]
print("pass with attempt", t["attemptRewarded"] / t["attempt"], "without", (t["rewarded"] - t["attemptRewarded"]) / (t["trials"] - t["attempt"]))
