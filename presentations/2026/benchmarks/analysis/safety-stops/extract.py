"""Build data/safety-stops.json: trials that ended on a provider safety stop, Terminal-Bench 2.1.

Source: the three public annotated Spaces' report files (atif-scan v0.8.23), as in
analysis/answer-lookup/extract.py. A safety stop is a trial whose error_type is
AgentSafetyStopError or AgentSafetyRefusalError. Counts only: model, harness, task, how many
agent steps and tool calls had happened, and whether the trial was still rewarded.

Run: python3 analysis/safety-stops/extract.py   (reads about 2 GB of reports; takes a few minutes)
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path.home() / "source"
DECK = Path(__file__).resolve().parent.parent.parent
RUNS = SRC / "fast-agent/docs/benchmark_data/runs"
SITES = {"leaderboard": ("tb21-annotated", "rows-v4"), "submissions": ("tb21-submissions-annotated", "subs-v4"),
         "fast-agent": ("tb21-fast-agent-annotated", "rows-v3")}
LAB = [("Claude", r"Opus|Sonnet|Haiku|Fable"), ("OpenAI", r"GPT")]

rows, stops, handoffs = [], [], []
task_trials = defaultdict(Counter)  # lab -> task -> trials, over the runs of that lab that had at least one stop
for site, (repo, exports) in SITES.items():
    board = json.loads(re.search(r"=\s*(\{.*\})\s*;?\s*$", (SRC / repo / "site/data/leaderboard.js").read_text(), re.S).group(1))
    for row in board["rows"]:
        if row["agent"] == "Slai Agent Runtime":
            continue  # no model calls
        folder = row.get("folder") or row["id"]
        inputs = json.loads((SRC / repo / "build" / exports / folder / "report.json").read_text())["inputs"]
        if site == "fast-agent":
            run = json.loads((RUNS / f"{folder}.json").read_text())
            listed = {t["name"] for ts in run["trials"].values() for t in ts}
            inputs = [i for i in inputs if i["input_id"].split("/")[0] in listed or i.get("hub_trial_id") in listed]
        lab = next((name for name, pat in LAB if re.search(pat, row["model"])), "other")
        mine = [i for i in inputs if "Safety" in str(i.get("error_type"))]
        # Hand-offs: trials where a model other than the row's main model answered some steps (a harness fallback).
        steps = Counter()
        for i in inputs:
            steps.update(i.get("step_models") or {})
        main = steps.most_common(1)[0][0] if steps else None
        handed = [i for i in inputs if any(m != main for m in (i.get("step_models") or {}))]
        other = Counter(m for i in handed for m in i["step_models"] if m != main).most_common(1)
        rows.append(dict(site=site, agent=re.sub(r" \d+\.[\d.]+$", "", row["agent"]), model=row["model"], effort=row.get("effort"), lab=lab,
                         rewarded=sum((i.get("reward") or 0) > 0 for i in inputs), stopsRewarded=sum((i.get("reward") or 0) > 0 for i in inputs if "Safety" in str(i.get("error_type"))),
                         trials=len(inputs), stops=len(mine), handed=len(handed), handedRewarded=sum((i.get("reward") or 0) > 0 for i in handed),
                         handedTo=other[0][0] if other else None, handedTasks=len({i["task"] for i in handed}),
                         handedErrors=sum("Safety" in str(i.get("error_type")) for i in handed)))
        for i in handed:
            handoffs.append(dict(agent=rows[-1]["agent"], model=row["model"], task=i["task"], rewarded=(i.get("reward") or 0) > 0))
        if mine:
            for i in inputs:
                task_trials[lab][i["task"]] += 1
        for i in mine:
            stops.append(dict(lab=lab, model=row["model"], agent=rows[-1]["agent"], task=i["task"],
                              steps=i.get("agent_steps") or 0, tools=i.get("tool_calls") or 0,
                              rewarded=(i.get("reward") or 0) > 0))

labs = {}
for lab in ("Claude", "OpenAI"):
    mine = [s for s in stops if s["lab"] == lab]
    tasks = Counter(s["task"] for s in mine)
    labs[lab] = dict(stops=len(mine), beforeAnyTool=sum(s["tools"] == 0 for s in mine), rewarded=sum(s["rewarded"] for s in mine),
                     runs=sum(1 for r in rows if r["lab"] == lab and r["stops"]),
                     trialsInThoseRuns=sum(r["trials"] for r in rows if r["lab"] == lab and r["stops"]),
                     tasks=[dict(task=t, stops=n, trials=task_trials[lab][t],
                                 rewarded=sum(s["rewarded"] for s in mine if s["task"] == t),
                                 beforeAnyTool=sum(s["tools"] == 0 for s in mine if s["task"] == t)) for t, n in tasks.most_common()],
                     models=Counter(s["model"] for s in mine).most_common(),
                     harnesses=Counter(s["agent"] for s in mine).most_common(),
                     steps=sorted(Counter(min(s["steps"], 6) for s in mine).items()))
out = dict(generated="2026-11-10", scanner="v0.8.23", trials=sum(r["trials"] for r in rows), runs=len(rows), stops=len(stops),
           rewarded=sum(s["rewarded"] for s in stops), labs=labs,
           rows=sorted((r for r in rows if r["stops"] or r["handed"]), key=lambda r: -r["stops"]),
           handoffTasks=[dict(agent=a, model=m, task=t, handed=n, rewarded=sum(h["rewarded"] for h in handoffs if (h["agent"], h["model"], h["task"]) == (a, m, t)))
                         for (a, m, t), n in sorted(Counter((h["agent"], h["model"], h["task"]) for h in handoffs).items())],
           noStops=dict(Claude=[r for r in rows if r["lab"] == "Claude" and not r["stops"]],
                        OpenAI=[r for r in rows if r["lab"] == "OpenAI" and not r["stops"]]))
(DECK / "data/safety-stops.json").write_text(json.dumps(out, indent=1))
print(out["trials"], out["runs"], out["stops"], out["rewarded"])
for lab, v in labs.items():
    print(lab, {k: v[k] for k in ("stops", "beforeAnyTool", "rewarded", "runs", "trialsInThoseRuns")}, v["models"], v["harnesses"], v["steps"])
    for t in v["tasks"]:
        print("  ", t)
for r in out["rows"]:
    print(r)
print(len(out["handoffTasks"]))
print("no stops:", [(r["agent"], r["model"], r["effort"]) for lab in out["noStops"].values() for r in lab])
