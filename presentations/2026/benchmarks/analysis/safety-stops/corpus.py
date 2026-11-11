"""Build data/safety-corpus.json: provider safety stops across the whole private corpus. Counts only.

The public-run slides (data/safety-stops.json, extract.py) cover 65 Terminal-Bench 2.1 runs. This
covers every trial in the 8 Nov 2026 census (talk-hunt-2026-11-08/census/census.json), deduplicated
the way census/agg.py does it, and reads each trial's Harbor result.json for the error type.
A safety stop is exception_type AgentSafetyStopError or AgentSafetyRefusalError.

Writes model names, task names and counts. No paths, run IDs or trace text.
Run: python3 analysis/safety-stops/corpus.py   (reads about 170,000 small files; a minute or two)
"""
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path.home()
DECK = Path(__file__).resolve().parent.parent.parent
CENSUS = HOME / ".cache/atif-scan/reviews/talk-hunt-2026-11-08/census/census.json"
FABLE_ROW = HOME / "source/tb21-annotated/build/rows-v4/06-claude-code--fable-5--xhigh/report.json"
INCIDENT = "tb21-astra-low-hf-c10-r1-01018"  # one job, 6 Sep 2026: stops on 37 unrelated tasks

PREFIX = (r"^(hf\.|openai/|xai/|x-ai/|deepseek/|deepseek\.|anthropic/|google/|gemini/|vertex_ai/|zai-org/|zai/|z-ai/|moonshotai/|moonshot/|qwen/|"
          r"meta-models/|meta/|harbor-endpoint/|copilot/|copilot\.|codexresponses\.|responses\.|openai-codex/|hlb/|cursor/|openrouter/|together/|"
          r"fireworks/|devin/|hai_console/|aws-bedrock/anthropic\.)")


def norm(m):
    m = (m or "").lower().split("?")[0]
    m = re.sub(PREFIX, "", m)
    m = re.sub(r":.*$", "", m).replace("deepseek-ai/", "")
    m = re.sub(r"(claude-(?:opus|sonnet|haiku|fable)-\d)-(\d)$", r"\1.\2", m)
    return m or "unknown"


FAMILIES = [("claude", "Claude"), ("fable", "Claude"), ("opus", "Claude"), ("sonnet", "Claude"), ("haiku", "Claude"), ("grok", "Grok"),
            ("gemini", "Gemini"), ("deepseek", "DeepSeek"), ("qwen", "Qwen"), ("kimi", "Kimi"), ("glm", "GLM"), ("muse", "Muse"), ("gpt", "GPT")]


def family(m):
    return next((v for k, v in FAMILIES if k in m), "other")


def display(m):
    m = re.sub(r"^claude-", "", m)
    m = re.sub(r"^haiku-4-5-\d+$", "haiku-4.5", m)
    if m.startswith("gpt-"):
        return "GPT-" + " ".join(w.capitalize() for w in m[4:].split("-"))
    return " ".join(w.capitalize() for w in m.split("-"))


def result(path):
    d = os.path.dirname(path)
    for cand in (os.path.join(os.path.dirname(d), "result.json"), os.path.join(d, "result.json")):
        if os.path.exists(cand):
            try:
                j = json.load(open(cand))
            except Exception:
                return None, None
            exc = (j.get("exception_info") or {}).get("exception_type")
            rewards = (j.get("verifier_result") or {}).get("rewards")
            return exc, rewards.get("reward") if isinstance(rewards, dict) else None
    return None, None


seen, trials = set(), []
for r in json.load(open(CENSUS)):
    m = norm(r["model"])
    key = (r["trial"], m, r.get("steps"), r.get("tool_calls"))
    if r["rp"] in seen or key in seen:
        continue
    seen.update((r["rp"], key))
    exc, reward = result(r["path"])
    trials.append(dict(model=m, family=family(m), bench="TB 2.x" if r["bench"].startswith("TB2") else r["bench"], harness=r["harness"],
                       task=r["trial"].split("__")[0], stop="Safety" in str(exc), ended=exc is not None or reward is not None,
                       rewarded=(reward or 0) > 0, incident=INCIDENT in r["path"]))

stops = [t for t in trials if t["stop"]]
by_model, stop_by_model = Counter(t["model"] for t in trials), Counter(t["model"] for t in stops)
models = [dict(model=display(m), family=family(m), stops=n, trials=by_model[m],
               tasks=len({t["task"] for t in stops if t["model"] == m and not t["incident"]}),
               harnesses=sorted({t["harness"] for t in stops if t["model"] == m}))
          for m, n in stop_by_model.most_common()]
clear_models = [dict(model=display(m), family=family(m), trials=n) for m, n in by_model.most_common()
                if not stop_by_model[m] and family(m) in ("Claude", "GPT") and n >= 400]
fam_trials, fam_models = Counter(), defaultdict(set)
for t in trials:
    if t["family"] not in ("Claude", "GPT"):
        fam_trials[t["family"]] += 1
        fam_models[t["family"]].add(t["model"])
clear_families = [dict(family=f, trials=n, models=len(fam_models[f])) for f, n in fam_trials.most_common()]
assert not any(t["stop"] for t in trials if t["family"] not in ("Claude", "GPT"))

# Model by task, fast-agent on Terminal-Bench 2.x only (one harness, so the columns compare), without the incident job.
COLS = ["claude-fable-5", "claude-opus-5.5", "claude-sonnet-5.5", "claude-haiku-5.5", "gpt-6-astra", "gpt-5.6-sol", "gpt-6.1-sol", "gpt-6.1-terra"]
grid = [t for t in trials if t["bench"] == "TB 2.x" and t["harness"] == "fast-agent" and t["model"] in COLS and t["ended"] and not t["incident"]]
cell_n, cell_s = Counter((t["model"], t["task"]) for t in grid), Counter((t["model"], t["task"]) for t in grid if t["stop"])
tasks = sorted({t for (m, t), n in cell_s.items() if n >= 4 and n / cell_n[(m, t)] >= 0.3}, key=lambda t: (-sum(cell_s[(m, t)] > 0 for m in COLS), -sum(cell_s[(m, t)] for m in COLS)))
matrix = dict(models=[display(m) for m in COLS],
              rows=[dict(task=t, cells=[dict(stops=cell_s[(m, t)], trials=cell_n[(m, t)]) for m in COLS]) for t in tasks],
              stops=sum(cell_s.values()), trials=len(grid))

# Fable 5 on Claude Code (public leaderboard row): trials where Opus 4.8 answered some steps, and how those tasks
# line up with the tasks where fast-agent records a Fable 5 safety stop.
fable_stop_tasks = {t["task"] for t in stops if t["model"] == "claude-fable-5"}
swapped = Counter(i["task"] for i in json.load(open(FABLE_ROW))["inputs"] if any("opus" in k for k in (i.get("step_models") or {})))
fable = dict(stopTasks=len(fable_stop_tasks), stops=stop_by_model["claude-fable-5"], swappedTrials=sum(swapped.values()), swappedTasks=len(swapped),
             overlap=len(fable_stop_tasks & set(swapped)),
             otherHarness={h: n for h, n in Counter(t["harness"] for t in trials if t["model"] == "claude-fable-5" and t["harness"] != "fast-agent").items()})

bench_n, bench_s = Counter(t["bench"] for t in trials), Counter(t["bench"] for t in stops)
out = dict(generated="2026-11-11", source="corpus census of 8 Nov 2026", trials=len(trials), stops=len(stops),
           rewarded=sum(t["rewarded"] for t in stops), incident=dict(date="6 Sep 2026", model="GPT-6 Astra", stops=sum(t["incident"] for t in stops),
                                                                    tasks=len({t["task"] for t in stops if t["incident"]})),
           labs=dict(Anthropic=sum(t["family"] == "Claude" for t in stops), OpenAI=sum(t["family"] == "GPT" for t in stops)),
           labTrials=dict(Anthropic=sum(t["family"] == "Claude" for t in trials), OpenAI=sum(t["family"] == "GPT" for t in trials)),
           benchmarks=[dict(bench=b, trials=n, stops=bench_s[b]) for b, n in bench_n.most_common()],
           models=models, clearModels=clear_models, clearFamilies=clear_families, matrix=matrix, fable=fable)
(DECK / "data/safety-corpus.json").write_text(json.dumps(out, indent=1))
print({k: out[k] for k in ("trials", "stops", "rewarded", "incident", "labs", "labTrials", "benchmarks", "fable")})
for m in models:
    print(m)
print(clear_models, clear_families, sep="\n")
print(" " * 34 + " ".join(f"{m[-9:]:>9}" for m in matrix["models"]), matrix["stops"], matrix["trials"])
for row in matrix["rows"]:
    print(f"{row['task']:34}" + " ".join(f"{c['stops']:>4}/{c['trials']:<4}" for c in row["cells"]))
