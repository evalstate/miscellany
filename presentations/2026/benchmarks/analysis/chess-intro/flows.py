"""Build data/chess-flows.json: how fast-agent trials approached chess-best-move.

Counts only, from the private strategies review (labels are regex matches over
the agent's own tool calls, plus whether an image was read). fast-agent only:
it is the one harness whose traces record image reads, so "no image read" means
that, not "unknown". No trace text, paths or IDs are written.

Run: python3 analysis/chess-intro/flows.py
"""
import collections
import json
import re
from pathlib import Path

REVIEW = Path.home() / ".cache/atif-scan/reviews/talk-hunt-2026-11-08/strategies"
DECK = Path(__file__).resolve().parent.parent.parent
NAMES = {
    "gpt-6-luna": "GPT-6 Luna", "gpt-6.1-terra": "GPT-6.1 Terra", "gpt-5.6-luna": "GPT-5.6 Luna", "grok*": "Grok",
    "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-6-astra": "GPT-6 Astra", "glm*": "GLM", "claude*": "Claude",
    "deepseek-v4-flash": "DeepSeek V4 Flash", "gpt-5.6-terra": "GPT-5.6 Terra", "gpt-6.1-sol": "GPT-6.1 Sol",
}
MIN_ARM = 8  # a model needs this many trials with and without the engine to be compared

art = {r["path"]: r["art_lines"] for r in json.loads((REVIEW / "asciiart_rows.json").read_text())}
rows = [r for r in json.loads((REVIEW / "rows.json").read_text())
        if r["task"] == "chess-best-move" and r["harness"] == "fast-agent" and r["reward"] is not None]

see = lambda r: "image" if r["vision_n"] > 0 else "noimage"
eng = lambda r: "stockfish" if "stockfish" in r["strat"] else ("pychess" if "python-chess" in r["strat"] else "none")
out = lambda r: "pass" if r["reward"] >= 1 else "fail"

links = collections.Counter()
for r in rows:
    links[(see(r), eng(r))] += 1
    links[(eng(r), out(r))] += 1
    links[(see(r), eng(r), out(r))] += 1

fam = collections.defaultdict(lambda: {"with": [0, 0], "without": [0, 0]})
for r in rows:
    arm = fam[r["fam"]]["with" if eng(r) == "stockfish" else "without"]
    arm[0] += 1
    arm[1] += r["reward"] >= 1
models = [dict(model=NAMES.get(k, k), n=v["with"][0] + v["without"][0], withEngine=v["with"], without=v["without"])
          for k, v in fam.items() if min(v["with"][0], v["without"][0]) >= MIN_ARM]
models.sort(key=lambda m: -(m["withEngine"][1] / m["withEngine"][0] - m["without"][1] / m["without"][0]))

# Within-model effect: Mantel-Haenszel style weighted difference across every family.
num = den = 0.0
for v in fam.values():
    (a, pa), (b, pb) = v["with"], v["without"]
    if a and b:
        w = a * b / (a + b)
        num += w * (pa / a - pb / b)
        den += w
crude = [sum(v[k][i] for v in fam.values()) for k in ("with", "without") for i in (0, 1)]

versions, dates = set(), []
for r in rows:
    head = Path(r["path"]).read_text()
    m = re.search(r'"version":\s*"([\d.]+)"', head)
    if m:
        versions.add(tuple(int(x) for x in m.group(1).split(".")))
    d = re.search(r"20\d\d-\d\d-\d\dT", head)
    if d:
        dates.append(d.group(0)[:10])
v = lambda t: ".".join(map(str, t))

data = dict(
    n=len(rows), families=len(fam), minArm=MIN_ARM,
    versions=[v(min(versions)), v(max(versions))], dates=[min(dates), max(dates)], dated=len(dates),
    ascii=sum(art.get(r["path"], 0) >= 12 for r in rows),
    asciiNoImage=sum(art.get(r["path"], 0) >= 12 and r["vision_n"] == 0 for r in rows),
    links=[dict(path=list(k), n=n) for k, n in sorted(links.items())],
    models=models,
    crude=dict(withEngine=crude[:2], without=crude[2:]),
    withinModelDiff=round(num / den, 3),
)
(DECK / "data/chess-flows.json").write_text(json.dumps(data, indent=1))
print(json.dumps({k: data[k] for k in data if k not in ("links",)}, indent=1))
