"""Build data/lookup-rates.json: how often benchmark material reaches the agent, as a rate per model, over the whole corpus.

The other lookup data (extract.py) covers the 65 public Terminal-Bench 2.1 runs. This uses every trajectory in the
private corpus scan of 8 Nov 2026 (corpus_scan_v2.jsonl, 181,732 rows; atif-scan d469890 with b344b40 merged in),
because a count over the corpus would mostly describe which models were run most. A rate per model does not.

One check only: lookup.benchmark_content_obtained (critical): benchmark content came back from a web request.
It needs no task-specific allowance, so the older scan and the public reports agree on what it means. A trial whose
web results were not recorded is "unknown" for this check and is counted as not flagged: rates are lower bounds.

Trials are deduplicated by trial folder, model and step count (publication and share copies). Counts only.
Run: python3 analysis/answer-lookup/corpus_rates.py
"""
import json
import re
from collections import Counter
from pathlib import Path

HOME = Path.home()
DECK = Path(__file__).resolve().parent.parent.parent
SCAN = HOME / ".cache/atif-scan/reviews/corpus-2026-11-08/corpus_scan_v2.jsonl"
CENSUS = HOME / ".cache/atif-scan/reviews/talk-hunt-2026-11-08/census/census.json"
CHECK = "lookup.benchmark_content_obtained"
MIN_TRIALS = 1000
PREFIX = (r"^(hf\.|huggingface/|openai/|xai/|x-ai/|deepseek/|deepseek\.|anthropic/|anthropic\.|google/|gemini/|vertex_ai/|zai-org/|zai/|z-ai/|zai\.|"
          r"moonshotai/|moonshot/|qwen/|meta-models/|meta/|harbor-endpoint/|copilot/|copilot\.|codexresponses\.|responses\.|openai-codex/|hlb/|cursor/|"
          r"openrouter/|together/|fireworks/|devin/|hai_console/|aws-bedrock/anthropic\.|recursivesi/|ricursiveai/)")
NAMES = {"flint": "Flint", "ripple-65b-preview": "Ripple 65B Preview", "swe-1.7-lightning": "SWE-1.7", "deepseek-flash": "DeepSeek Flash"}


def norm(m):
    m = (m or "").lower().split("?")[0]
    for _ in range(3):
        m = re.sub(PREFIX, "", m)
    m = re.sub(r":.*$", "", m).replace("deepseek-ai/", "")
    return re.sub(r"(claude-(?:opus|sonnet|haiku|fable)-\d)-(\d)$", r"\1.\2", m) or "unknown"


def display(m):
    if m in NAMES:
        return NAMES[m]
    m = re.sub(r"^claude-", "", m)
    if m.startswith("gpt-"):
        return "GPT-" + " ".join(w.capitalize() for w in m[4:].split("-"))
    if m.startswith("glm-"):
        return "GLM-" + " ".join(w.capitalize() for w in m[4:].split("-"))
    return " ".join(w.capitalize() if w != "deepseek" else "DeepSeek" for w in m.split("-"))


census = {r["path"]: r["bench"] for r in json.load(open(CENSUS))}


def bench(path):
    b = census.get(path)
    if b:
        return "Terminal-Bench 2.x" if b.startswith("TB2") else {"TB4": "Terminal-Bench 4", "HLB": "other"}.get(b, b)
    if "/deep-swe/" in path:
        return "DeepSWE"
    if "agents-last-exam" in path:
        return "TB-lite"
    return "Terminal-Bench 2.x" if "/terminal-bench-2" in path or "/tb21" in path else "other"


seen, trials = set(), []
for line in open(SCAN):
    r = json.loads(line)
    m = re.search(r"/([^/]+__[A-Za-z0-9]+)/", r["path"])
    model = norm(r.get("model"))
    key = (m.group(1) if m else r["path"], model, r.get("nsteps"))
    if key in seen:
        continue
    seen.add(key)
    status = r["checks"].get(CHECK)  # absent = no_match
    trials.append(dict(model=model, bench=bench(r["path"]), got=status == "match", unknown=status == "unknown"))


def tally(ts):
    n, got = len(ts), sum(t["got"] for t in ts)
    return dict(trials=n, obtained=got, unknown=sum(t["unknown"] for t in ts), per1000=round(1000 * got / n, 2) if n else 0)


order = ["Terminal-Bench 2.x", "TB-lite", "DeepSWE", "Terminal-Bench 4"]
tb = [t for t in trials if t["bench"] == "Terminal-Bench 2.x"]
by_model = Counter(t["model"] for t in tb)
models = sorted((dict(model=display(m), **tally([t for t in tb if t["model"] == m])) for m, n in by_model.items() if n >= MIN_TRIALS),
                key=lambda x: (-x["obtained"] / x["trials"], -x["trials"]))
small = sorted((dict(model=display(m), **tally([t for t in tb if t["model"] == m])) for m, n in by_model.items()
                if 400 <= n < MIN_TRIALS and any(t["got"] for t in tb if t["model"] == m)), key=lambda x: -x["obtained"] / x["trials"])
out = dict(generated="2026-11-11", source="corpus scan v2, 8 Nov 2026", check=CHECK, minTrials=MIN_TRIALS, all=tally(trials),
           benchmarks=[dict(bench=b, **tally([t for t in trials if t["bench"] == b])) for b in order],
           models=models, small=small, modelsCounted=len(by_model))
(DECK / "data/lookup-rates.json").write_text(json.dumps(out, indent=1))
print(out["all"], out["benchmarks"], sep="\n")
for m in models + small:
    print(m)
