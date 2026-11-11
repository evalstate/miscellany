"""Build data/zork-methods.json: how each model goes about TB 2.1 extract-moves-from-video on fast-agent.

Source: the trial list in talk-hunt-2026-11-08/strategies/rows.json (private), each trial's trajectory.
Method labels are pattern matches on the code and commands in the agent's own tool calls:
- fuzzy: approximate text matching (difflib, SequenceMatcher, rapidfuzz, Levenshtein and similar)
- crop: crops a region of a frame before reading it
- fixes: a named table of OCR corrections (fix/correct/repair/normalise... = {...})
- image: at least one frame read as an image
A label says the code is there, not that it is why the trial passed. Trials with no recorded reward are left out.
Counts only. Run: python3 analysis/zork-methods/extract.py
"""
import json
import os
import re
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DECK = Path(__file__).resolve().parent.parent.parent
ROWS = Path.home() / ".cache/atif-scan/reviews/talk-hunt-2026-11-08/strategies/rows.json"
PAT = {k: re.compile(v, re.I) for k, v in {
    "fuzzy": r"difflib|SequenceMatcher|rapidfuzz|fuzzywuzzy|thefuzz|levenshtein|get_close_matches|editdistance|jellyfish|edit_distance",
    "crop": r"\.crop\(|crop=|-crop |\[\s*\w+\s*:\s*\w+\s*,\s*\w+\s*:\s*\w+\s*\]",
    "fixes": r"(fix|correct|repair|normali[sz]|canonical|replac|typo|ocr_map|subst)\w*\s*=\s*[\{\[\(]",
}.items()}
VIEW = re.compile(r"attach_media|view_image|read_image|image_view|screenshot|read_media|look_at", re.I)
IMG = re.compile(r"\.(png|jpe?g|gif|webp|bmp)\b", re.I)
CLAUDE = [("fable-5", "Fable 5"), ("opus-5.5", "Opus 5.5"), ("sonnet-5.5", "Sonnet 5.5"), ("haiku-5.5", "Haiku 5.5")]
NAMES = {"gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6-astra": "GPT-6 Astra", "gpt-6.1-terra": "GPT-6.1 Terra", "gpt-6-luna": "GPT-6 Luna",
         "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-luna": "GPT-5.6 Luna", "glm*": "GLM", "grok*": "Grok", "deepseek-v4-flash": "DeepSeek V4 Flash"}
SHOWN = ["GPT-6.1 Sol", "Opus 5.5", "Sonnet 5.5", "GPT-6 Astra", "GPT-6.1 Terra", "GPT-6 Luna", "GLM", "Haiku 5.5", "DeepSeek V4 Flash"]


def has_image(x):
    return isinstance(x, list) and any(isinstance(a, dict) and a.get("type") in ("image", "image_url", "input_image") for a in x)


def one(row):
    path = row["path"]
    try:
        d = json.load(open(path if path.endswith(".json") else os.path.join(path, "agent/trajectory.json")))
        reward = float(row["reward"])
    except Exception:
        return None
    steps, text, image = d.get("steps") or [], [], False
    for i, s in enumerate(steps):
        image |= (i > 0 and has_image(s.get("message"))) or any(has_image(r.get("content")) for r in (s.get("observation") or {}).get("results") or [])
        for tc in s.get("tool_calls") or []:
            a = tc.get("arguments")
            a = "\n".join(str(v) for v in a.values())[:40000] if isinstance(a, dict) else str(a)[:40000]
            fn = tc.get("function_name") or ""
            text.append(a)
            image |= bool(VIEW.search(fn) or (fn.lower() in ("read", "view", "read_file", "read_text_file") and IMG.search(a)))
    t = "\n".join(text)
    model = str((d.get("agent") or {}).get("model_name")).lower().replace("-5-5", "-5.5")
    name = next((v for k, v in CLAUDE if k in model), None) if row["fam"] == "claude*" else NAMES.get(row["fam"])
    return dict(model=name, rewarded=reward > 0, steps=sum(s.get("source") == "agent" for s in steps), image=image, **{k: bool(rx.search(t)) for k, rx in PAT.items()})


if __name__ == "__main__":
    rows = [r for r in json.load(open(ROWS)) if r.get("task") == "extract-moves-from-video" and r["harness"] == "fast-agent"]
    with ProcessPoolExecutor(12) as ex:
        trials = [t for t in ex.map(one, rows, chunksize=10) if t]
    groups = defaultdict(list)
    for t in trials:
        groups[t["model"]].append(t)

    def tally(ts):
        return dict(n=len(ts), rewarded=sum(t["rewarded"] for t in ts))

    def split(ts, key):
        return {"with": tally([t for t in ts if t[key]]), "without": tally([t for t in ts if not t[key]])}

    models = [dict(model=m, **tally(groups[m]), steps=statistics.median(t["steps"] for t in groups[m]),
                   **{k: sum(t[k] for t in groups[m]) for k in ("fuzzy", "crop", "fixes", "image")}) for m in SHOWN]
    out = dict(generated="2026-11-11", trials=len(trials), rewarded=sum(t["rewarded"] for t in trials), unknownReward=len(rows) - len(trials),
               models=models, others=[dict(model=m, **tally(ts), fuzzy=sum(t["fuzzy"] for t in ts)) for m, ts in groups.items() if m and m not in SHOWN],
               all={k: split(trials, k) for k in ("fuzzy", "crop", "fixes", "image")},
               luna={k: split(groups["GPT-6 Luna"], k) for k in ("fuzzy", "crop", "fixes", "image")})
    (DECK / "data/zork-methods.json").write_text(json.dumps(out, indent=1))
    print(out["trials"], out["rewarded"], out["unknownReward"], out["all"], out["luna"], sep="\n")
    for m in models + out["others"]:
        print(m)
