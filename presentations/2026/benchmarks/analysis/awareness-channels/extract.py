"""Build data/awareness-channels.json: where a trial is flagged, in its reasoning text or in its tool calls.

Source: the four public annotated Spaces' report files (atif-scan v0.8.23): Terminal-Bench 2.1 leaderboard rows,
pull-request submissions, our fast-agent runs, and Datacurve's public DeepSWE 1.1 grid. Each matched check cites
its evidence with a channel. Two flags per trial, both leaving out matches an allowance explains:

- reasoning: a benchmark-awareness check (awareness.benchmark, awareness.named_benchmark,
  awareness.detection_evasion, review.benchmark_named) or any behavioural check of medium severity or above,
  with evidence cited in the reasoning channel.
- tool: a behavioural check of medium severity or above with evidence cited in a tool call
  (command, path, arguments, url, query, payload).

Behavioural means not integrity.*, harness.*, awareness.* or review.*. awareness.verifier (severity info:
"hidden tests or verifier discussed") is not counted. Trials are split by the report's reasoning-exposure code:
"recorded" (some reasoning text is in the trace: it may be a summary) against everything else.

Counts only. Run: python3 analysis/awareness-channels/extract.py   (reads about 9 GB of reports; several minutes)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path.home() / "source"
DECK = Path(__file__).resolve().parent.parent.parent
SITES = {"Terminal-Bench 2.1 leaderboard": ("tb21-annotated", "rows-v4"), "Terminal-Bench 2.1 submissions": ("tb21-submissions-annotated", "subs-v4"),
         "Terminal-Bench 2.1 fast-agent runs": ("tb21-fast-agent-annotated", "rows-v3"), "DeepSWE 1.1": ("deepswe-annotated", "rows-v4")}
AWARE = {"awareness.benchmark", "awareness.named_benchmark", "awareness.detection_evasion", "review.benchmark_named"}
TOOL = {"command", "path", "arguments", "url", "query", "payload"}
SEVERE = {"medium", "high", "critical"}
NOT_BEHAVIOUR = {"integrity", "harness", "awareness", "review", "expected", "context", "ending"}
# Allowance added to atif-scan after v0.8.23 (expected.tb21.sanitize_git_repo_git_inspection, 11 Nov 2026), not yet in the
# Spaces' reports: on sanitize-git-repo, inspecting git history and unreachable objects is the task's own verification.
LATER_ALLOWANCES = {"sanitize-git-repo": {"lookup.git_history", "lookup.git_unreachable_objects"}}
FAMILIES = [("fable", "Claude"), ("opus", "Claude"), ("sonnet", "Claude"), ("haiku", "Claude"), ("claude", "Claude"), ("grok", "Grok"),
            ("gemini", "Gemini"), ("deepseek", "DeepSeek"), ("qwen", "Qwen"), ("kimi", "Kimi"), ("glm", "GLM"), ("muse", "Muse"),
            ("gpt", "GPT"), ("astra", "GPT"), ("luna", "GPT"), ("sol", "GPT"), ("terra", "GPT"), ("swe-1", "SWE-1.7"), ("swe_1", "SWE-1.7")]


def family(text):
    return next((v for k, v in FAMILIES if k in text), "other")


trials, sites = [], Counter()
for site, (repo, exports) in SITES.items():
    for f in sorted((SRC / repo / "build" / exports).glob("*/report.json")):
        folder = f.parent.name
        if "slai" in folder:
            continue  # no model calls
        sites[site] += 1
        for i in json.loads(f.read_text())["inputs"]:
            reasoning = tool = False
            for a in i["assessments"]:
                if a["status"] != "match" or a["expected_by"] or a["id"] in LATER_ALLOWANCES.get(i.get("task"), ()):
                    continue
                channels = {e.get("channel") for e in a["evidence"]}
                behaviour = a["severity"] in SEVERE and a["id"].split(".")[0] not in NOT_BEHAVIOUR
                if "reasoning" in channels and (a["id"] in AWARE or behaviour):
                    reasoning = True
                if behaviour and channels & TOOL:
                    tool = True
            trials.append(dict(run=folder, site=site, family=family(f"{i.get('model_name')} {i.get('configured_model')} {folder}".lower()),
                               recorded=i.get("reasoning") == "recorded", reasoning=reasoning, tool=tool))


def tally(rows):
    return dict(trials=len(rows), tool=sum(r["tool"] for r in rows), reasoning=sum(r["reasoning"] for r in rows),
                both=sum(r["tool"] and r["reasoning"] for r in rows))


groups = defaultdict(list)
for t in trials:
    groups[(t["family"], t["recorded"])].append(t)
rec = [t for t in trials if t["recorded"]]
out = dict(generated="2026-11-11", scanner="v0.8.23", trials=len(trials), runs=sum(sites.values()), sites=dict(sites),
           recorded=tally(rec), notRecorded=tally([t for t in trials if not t["recorded"]]),
           toolGivenReasoning=tally([t for t in rec if t["reasoning"]]), toolGivenNoReasoning=tally([t for t in rec if not t["reasoning"]]),
           families=sorted((dict(family=f, recorded=r, **tally(rows)) for (f, r), rows in groups.items()), key=lambda x: (not x["recorded"], -x["trials"])))
# One row per DeepSWE run: one harness (mini-swe-agent), one benchmark, so runs at the same effort compare.
by_run = defaultdict(list)
for t in trials:
    if t["site"] == "DeepSWE 1.1":
        by_run[t["run"]].append(t)
out["deepswe"] = [dict(run=run.replace("mini_swe_agent_", ""), family=rows[0]["family"], recordedTrials=sum(r["recorded"] for r in rows), **tally(rows))
                  for run, rows in sorted(by_run.items())]
(DECK / "data/awareness-channels.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("families", "deepswe")}))
for r in out["deepswe"]:
    print(r["run"], r["trials"], r["recordedTrials"], round(100 * r["tool"] / r["trials"], 1), round(100 * r["reasoning"] / r["trials"], 1))
for f in out["families"]:
    print(f, round(100 * f["tool"] / f["trials"], 1), round(100 * f["reasoning"] / f["trials"], 1))
