"""Build data/evasion.json: how often each network workaround and side channel shows up.

Source: the private corpus scan of 8 Nov 2026
(~/.cache/atif-scan/reviews/corpus-2026-11-08/corpus_scan_v2.jsonl, 181,732 trajectories,
atif-scan d469890 / b344b40). Counts are trajectories whose check status is "match": the
shape of a command or request, not whether it got anywhere. Model families are a coarse
grouping of the configured model string. Counts only.

Run: python3 analysis/evasion/extract.py
"""
import json
from collections import Counter
from pathlib import Path

SCAN = Path.home() / ".cache/atif-scan/reviews/corpus-2026-11-08/corpus_scan_v2.jsonl"
DECK = Path(__file__).resolve().parent.parent.parent
CHECKS = {
    "archive": "network.web_archive",
    "reader": "side_channel.fetch_proxy",
    "pinned": "network.isolation_bypass",
    "noproxy": "network.proxy_bypass",
    "model": "side_channel.model_call",
    "modelAnswered": "side_channel.model_call_answered",
    "k8sCredentials": "side_channel.orchestrator_credentials",
    "k8sRequest": "side_channel.orchestrator_api_request",
}
FAMILIES = [("deepseek-v4-pro", "DeepSeek V4 Pro"), ("v4.1-flash", "DeepSeek V4.1 Flash"), ("deepseek-flash", "DeepSeek V4.1 Flash"),
            ("deepseek", "DeepSeek V4 Flash"), ("grok", "Grok"), ("glm", "GLM"), ("flint", "Flint"), ("ripple", "Ripple"),
            ("qwen", "Qwen"), ("gpt-6.1-sol", "GPT-6.1 Sol"), ("gpt-6-luna", "GPT-6 Luna"), ("gpt-5.6-luna", "GPT-5.6 Luna"),
            ("gpt", "other GPT"), ("claude", "Claude"), ("gemini", "Gemini"), ("muse", "Muse")]


def family(model):
    model = (model or "").lower()
    return next((name for key, name in FAMILIES if key in model), "other")


total, counts = 0, {k: Counter() for k in CHECKS}
k8s_tasks, k8s_answered = Counter(), Counter()
for line in SCAN.open():
    row = json.loads(line)
    total += 1
    checks = row.get("checks") or {}
    for key, check in CHECKS.items():
        if checks.get(check) == "match":
            counts[key][family(row.get("model"))] += 1
    if checks.get("side_channel.orchestrator_api_request") == "match":
        k8s_tasks[row.get("task")] += 1
        k8s_answered[checks.get("side_channel.orchestrator_api_answered", "no_match")] += 1

out = dict(source="corpus scan v2, 8 Nov 2026", trajectories=total,
           checks={k: dict(n=sum(c.values()), families=c.most_common(4)) for k, c in counts.items()},
           k8s=dict(tasks=k8s_tasks.most_common(), answered=dict(k8s_answered)))
(DECK / "data/evasion.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
