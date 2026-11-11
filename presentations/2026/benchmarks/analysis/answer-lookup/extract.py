"""Build data/answer-lookup.json: answer-lookup findings on Terminal-Bench 2.1, by
harness and model row and by task.

Source: the three public annotated Spaces (evalstate/tb21-annotated,
tb21-submissions-annotated, tb21-fast-agent-annotated), from their local checkouts:
site/data/leaderboard.js for the rows and build/<exports>/<row>/report.json (atif-scan
v0.8.23) for the per-trial assessments. Each task also carries the verdict of the tb21-network
review (does the task need network egress?). Counts only: no trace text, paths or trial IDs.

A trial counts when it has an unexcused medium-or-higher match in the Spaces'
"answer lookup" family (same rule as their build.py). Stages are nested by check id.

Run: python3 analysis/answer-lookup/extract.py
"""
import csv
import fnmatch
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path.home() / "source"
RUNS = SRC / "fast-agent/docs/benchmark_data/runs"
DECK = Path(__file__).resolve().parent.parent.parent
SITES = {
    "leaderboard": ("tb21-annotated", "rows-v4"),
    "submissions": ("tb21-submissions-annotated", "subs-v4"),
    "fast-agent": ("tb21-fast-agent-annotated", "rows-v3"),
}
SEV = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
LOOKUP = ["lookup.*", "access.benchmark_material_*", "tb21.task_origin_solution", "tb21.fix_ocaml_gc.upstream_source"]
NOT = {"lookup.git_history"}
# What kind of lookup a trial is: the first of these that matches (check ids, with the scanner's titles).
KINDS = [
    ("own", "Own task's solution or tests", {"lookup.own_task_files", "lookup.benchmark_solution_url"}),
    ("material", "Other benchmark material", {"access.benchmark_material_received", "lookup.benchmark_content_obtained"}),
    ("upstream", "Upstream fix", {"tb21.fix_ocaml_gc.upstream_source", "tb21.task_origin_solution"}),
    ("git", "Git objects", {"lookup.git_unreachable_objects"}),
    ("search", "Search only", set()),
]
RECEIVED = {"access.benchmark_material_received"}
# Allowances added to atif-scan after v0.8.23 (packs/tb21.py, expected.tb21.sanitize_git_repo_git_inspection, 11 Nov 2026)
# and not yet in the Spaces' reports: on sanitize-git-repo, inspecting git history and unreachable objects is the task's
# own verification. Applied here so the deck matches the scanner as it now stands.
LATER_ALLOWANCES = {"sanitize-git-repo": {"lookup.git_history", "lookup.git_unreachable_objects"}}


def is_lookup(check):
    return check not in NOT and any(fnmatch.fnmatchcase(check, g) for g in LOOKUP)


rows, tasks, checks, kinds, kind_tasks = [], defaultdict(Counter), Counter(), defaultdict(Counter), defaultdict(Counter)
excused_later = Counter()
for site, (repo, exports) in SITES.items():
    board = json.loads(re.search(r"=\s*(\{.*\})\s*;?\s*$", (SRC / repo / "site/data/leaderboard.js").read_text(), re.S).group(1))
    assert board["scanner"]["tag"] == "v0.8.23", board["scanner"]
    for row in board["rows"]:
        folder = row.get("folder") or row["id"]
        report = json.loads((SRC / repo / "build" / exports / folder / "report.json").read_text())
        assert report["scanner_version"] == "0.8.23"
        inputs = report["inputs"]
        if site == "fast-agent":
            # Only the reported trials count: replaced originals were scanned but aren't reported.
            run = json.loads((RUNS / f"{folder}.json").read_text())
            listed = {t["name"] for ts in run["trials"].values() for t in ts}
            inputs = [i for i in inputs if i["input_id"].split("/")[0] in listed or i.get("hub_trial_id") in listed]
        replay = row["agent"] == "Slai Agent Runtime"  # no model calls: kept as a row, left out of the totals
        c = Counter()
        for i in inputs:
            won = (i.get("reward") or 0) > 0
            ids = {a["id"] for a in i["assessments"]
                   if a["status"] == "match" and not a.get("expected_by")
                   and SEV.get(a["severity"], -1) >= SEV["medium"] and is_lookup(a["id"])}
            c["trials"] += 1
            c["rewarded"] += won
            if ids:  # as the Spaces publish it, before the later allowances
                c["published"] += 1
                c["publishedRewarded"] += won
            excused = ids & LATER_ALLOWANCES.get(i["task"], set())
            ids -= excused
            if excused and not ids:
                c["excusedLater"] += 1
                excused_later[i["task"]] += 1
            if not replay:
                tasks[i["task"]]["trials"] += 1
            if not ids:
                continue
            kind = next(k for k, _, members in KINDS if ids & members or not members)
            received = bool(ids & RECEIVED)
            c["lookup"] += 1
            c["lookupRewarded"] += won
            c["received"] += received
            c["own"] += kind == "own"
            c[f"kind.{kind}"] += 1
            if not replay:
                t = tasks[i["task"]]
                t["lookup"] += 1
                t["lookupRewarded"] += won
                t["received"] += received
                t["own"] += kind == "own"
                t[f"kind.{kind}"] += 1
                kinds[kind]["trials"] += 1
                kinds[kind]["rewarded"] += won
                kinds[kind]["received"] += received
                kind_tasks[kind][i["task"]] += 1
                checks.update(ids)
        published = row["scan"]["categories"].get("answer_lookup", {"trials": 0, "rewarded": 0})
        assert (c["published"], c["publishedRewarded"]) == (published["trials"], published["rewarded"]), (folder, c, published)
        assert c["trials"] == row["scan"]["trials"], (folder, c["trials"])
        holes = row["scan"]["holes"]
        rows.append(dict(site=site, agent=row["agent"], model=row["model"], effort=row.get("effort"), date=row["date"],
                         webHidden=holes.get("webHidden", 0), webTrials=holes.get("webTrials", 0),
                         **{k: c[k] for k in ("trials", "rewarded", "lookup", "lookupRewarded", "received", "own")},
                         kinds={k: c[f"kind.{k}"] for k, _, _ in KINDS}))

rows.sort(key=lambda r: -r["lookup"])
live = [r for r in rows if r["agent"] != "Slai Agent Runtime"]
FIELDS = ("trials", "lookup", "lookupRewarded", "received", "own")
total = {k: sum(r[k] for r in live) for k in ("trials", "rewarded", "lookup", "lookupRewarded", "received", "own")}
# Does the task itself need network access? From the private tb21-network review (verdict column only).
with (Path.home() / ".cache/atif-scan/reviews/tb21-network/tb21-network.csv").open() as f:
    network = {r["task"]: r["verdict"] for r in csv.DictReader(f)}
task_rows = []
for name, t in tasks.items():
    task_rows.append(dict(task=name, **{k: t[k] for k in FIELDS}, kinds={k: t[f"kind.{k}"] for k, _, _ in KINDS},
                          network=network[name]))
task_rows.sort(key=lambda t: -t["lookup"])
# Per model, over all of its runs: how many trials were flagged and in how many benchmark material arrived.
by_model = defaultdict(Counter)
for r in live:
    name = r["model"].removeprefix("Claude ")
    for k in ("trials", "lookup", "received"):
        by_model[name][k] += r[k]
    by_model[name]["runs"] += 1
models = sorted((dict(model=m, **c) for m, c in by_model.items()), key=lambda m: (-m["received"], -m["lookup"], m["model"]))
out = dict(generated="2026-11-11", scanner="v0.8.23", excusedLater=dict(excused_later),
           sources=[f"https://evalstate-{repo}.static.hf.space" for repo, _ in SITES.values()],
           dates=[min(r["date"] for r in rows), max(r["date"] for r in rows)],
           total=dict(rows=len(live), **total),
           kinds=[dict(key=k, label=label, **{f: kinds[k][f] for f in ("trials", "rewarded", "received")}, tasks=kind_tasks[k].most_common(3)) for k, label, _ in KINDS],
           rows=rows, models=models, tasks=task_rows, checks=dict(checks.most_common()))
(DECK / "data/answer-lookup.json").write_text(json.dumps(out, indent=1))
print(out["total"], out["dates"])
for k in out["kinds"]:
    print(k)
for r in rows[:24]:
    print(f"{r['site']:12} {r['agent']:22} {r['model']:30} {str(r['effort']):7} n={r['trials']} lookup={r['lookup']} rew={r['lookupRewarded']} recv={r['received']} own={r['own']} webHidden={r['webHidden']}")
print(sum(1 for r in live if r["lookup"] == 0), "rows with none;", sum(1 for r in live if r["lookup"] <= 2), "with two or fewer")
for t in task_rows[:16]:
    print(t)
print(len(task_rows), "tasks;", sum(1 for t in task_rows if t["lookup"] == 0), "with none; top4", sum(t["lookup"] for t in task_rows[:4]), "top8", sum(t["lookup"] for t in task_rows[:8]))
