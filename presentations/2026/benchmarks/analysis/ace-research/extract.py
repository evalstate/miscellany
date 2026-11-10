"""Build data/ace-research.json: when each model first wrote an *.ace-research host.

Sources (private): lab-host-timeline-2026-11-07/work/ace_traces.json (start time and model of
the 36 trials) and the report's "By model family" table (trials per family, trials before the
first hit), transcribed below. The two later harness-leaderboard sightings come from
ADDENDUM-2026-11-08.md and sit outside the 142,275-trajectory denominator.
Exports dates and counts only: no paths, hashes or trial names.

Run: python3 analysis/ace-research/extract.py
"""
import json
from collections import Counter
from pathlib import Path

SRC = Path.home() / ".cache/atif-scan/reviews/lab-host-timeline-2026-11-07/work/ace_traces.json"
DECK = Path(__file__).resolve().parent.parent.parent
# family: (lab, trials in the corpus, wrote the host, trials before the first hit) from the report table
TABLE = {
    "DeepSeek V4 Flash": ("DeepSeek", 5356, 12, 136),
    "DeepSeek V4 Pro": ("DeepSeek", 937, 18, 6),
    "GPT-5.6 Sol": ("OpenAI", 9977, 1, 6993),
    "GPT-6 Astra": ("OpenAI", 12706, 1, 11603),
    "GPT-6.1 Sol": ("OpenAI", 5261, 3, 2885),
    "DeepSeek V4.1 Flash": ("DeepSeek", 1446, 1, 963),
}
NEVER = [("GPT-6 Luna", 27626), ("GPT-5.6 Luna", 13102), ("Claude, all", 17464), ("Grok, all", 14763), ("GLM, all", 9871)]
LATER = [  # harness-leaderboard runs, 8 Nov addendum
    dict(family="DeepSeek V4.1 Flash", date="2026-10-30", note="rewrote resolv.conf, then looked the hosts up"),
    dict(family="Ripple-65B-Preview", date="2026-11-07", note="named it as OpenAI's; no request"),
]


def family(model):
    m = model.lower()
    for key, name in [("v4.1-flash", "DeepSeek V4.1 Flash"), ("v4-pro", "DeepSeek V4 Pro"), ("v4-flash", "DeepSeek V4 Flash"),
                      ("gpt-5.6-sol", "GPT-5.6 Sol"), ("gpt-6-astra", "GPT-6 Astra"), ("gpt-6.1-sol", "GPT-6.1 Sol")]:
        if key in m:
            return name
    raise ValueError(model)


traces = json.loads(SRC.read_text())
days = Counter((family(t["model"]), t["t0"][:10]) for t in traces)
hits = Counter(family(t["model"]) for t in traces)
rows = []
for name, (lab, trials, wrote, before) in TABLE.items():
    assert hits[name] == wrote, (name, hits[name], wrote)
    rows.append(dict(family=name, lab=lab, trials=trials, wrote=wrote, before=before,
                     days=[dict(date=d, n=n) for (f, d), n in sorted(days.items()) if f == name],
                     later=[x for x in LATER if x["family"] == name]))
rows.append(dict(family="Ripple-65B-Preview", lab="other", trials=None, wrote=0, before=None, days=[],
                 later=[x for x in LATER if x["family"] == "Ripple-65B-Preview"]))
rows.sort(key=lambda r: min([d["date"] for d in r["days"]] + [x["date"] for x in r["later"]]))
out = dict(corpus=142275, trials=len(traces), usedInCommand=23, rows=rows, never=[dict(family=f, trials=n) for f, n in NEVER])
(DECK / "data/ace-research.json").write_text(json.dumps(out, indent=1))
for r in rows:
    print(r["family"], r["wrote"], r["trials"], r["before"], [(d["date"], d["n"]) for d in r["days"]], r["later"])
