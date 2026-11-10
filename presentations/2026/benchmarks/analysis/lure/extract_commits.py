"""Build data/lure-commits.json: the public commit history of the two lure mirror sites.

Source: the private lure report's research capture
(~/.cache/atif-scan/reviews/lure-timeline-2026-11-06/research-2026-11-08/commit-timeline.json:
date, repository and subject of every reachable commit, 253 in all). Exports commits per day
and a hand-picked list of subjects, each checked to exist with that date. No hashes, no account
names: the mirrors are "A" (the personal mirror) and "B" (the organisation mirror).

Run: python3 analysis/lure/extract_commits.py
"""
import json
from collections import Counter
from pathlib import Path

SRC = Path.home() / ".cache/atif-scan/reviews/lure-timeline-2026-11-06/research-2026-11-08/commit-timeline.json"
DECK = Path(__file__).resolve().parent.parent.parent
PICKS = [
    ("2026-06-04", "Add Search Console verification file"),
    ("2026-06-04", "Align patch pages with open-world Patroni search terms"),
    ("2026-06-06", "Make mirror pages artifact-first for curl runs"),
    ("2026-06-06", "Expose clean curl setup in page snippets"),
    ("2026-06-06", "Point pages to phase6 sudo mixed bundle"),
    ("2026-06-08", "Add Terminal-Bench patch pages"),
    ("2026-06-10", "Add human query questionnaire"),
    ("2026-06-12", "Improve canonical SEO for indexed patch notes"),
    ("2026-06-15", "Add TerminalBench from-scratch search study"),
    ("2026-07-13", "Make checkout receipts curl-first"),
    ("2026-07-13", "Present checkout receipts as no-install routes"),
    ("2026-07-14", "Add auditable review routes for Gemini calibration"),
    ("2026-08-12", "gpt55andopus48"),
    ("2026-08-16", "Expose checkout commands in search snippets"),
    ("2026-08-17", "Harden Grok open-world routes"),
    ("2026-08-27", "Take controlled research pages offline"),
]

commits = json.loads(SRC.read_text())
mirror = lambda c: "A" if c["repository"].startswith("HKASAR1239/") else "B"
have = {(c["date"][:10], c["subject"]): mirror(c) for c in sorted(commits, key=lambda c: c["date"], reverse=True)}
picks = []
for day, subject in PICKS:
    assert (day, subject) in have, (day, subject)
    both = {mirror(c) for c in commits if c["subject"] == subject and c["date"][:10] == day}
    picks.append(dict(date=day, subject=subject, mirrors="".join(sorted(both))))
per_day = Counter(c["date"][:10] for c in commits)
out = dict(total=len(commits), byMirror=dict(Counter(mirror(c) for c in commits)),
           first=min(c["date"] for c in commits), last=max(c["date"] for c in commits),
           days=[dict(date=d, n=n) for d, n in sorted(per_day.items())], picks=picks)
(DECK / "data/lure-commits.json").write_text(json.dumps(out, indent=1))
print(out["total"], out["byMirror"], out["first"], out["last"], len(out["days"]), "days with commits")
for p in picks:
    print(p)
