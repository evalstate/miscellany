# Companion chart — install-windows-3.11: "Single task cost variance"

**Current chart:** `same_task_different_attempts.png` (+ `.svg` editable, `.html`, `observations.csv`), built by
`python3 gen_companion_v2.py`. Terminus 2 (high) vs Claude Code (xhigh), one zero-based bar per attempt sorted by
cost: solid = pass, pale = fail. Claude Code's fallback attempt (zWKshqF) is shown at its **full** cost: a dashed
$4.05 Fable segment plus a hatched $1.66 Opus 4.8 extension = $5.71 total (unlike the main chart, which counts fallback
attempts as failures and excludes their cost). Summaries: "3 / 5 passed on Fable" plus "1 additional pass after Opus
fallback", and the cost range and spread (computed on full attempt cost). Effort badges (high / xhigh) sit beside
each harness name, and a legend-line note says it is not a controlled harness comparison. `observations.csv` has
total, Fable and Opus cost columns. Data, checks and pricing come from `gen_companion.py` / `gen_companion_cc.py` (details below).
Earlier dot-plot versions (fast-agent row and Claude Code row) are in `previous/`.

## Earlier version: fast-agent row (in `previous/`)

```bash
python3 gen_companion.py   # -> same_task_different_attempts.png (@2x), .svg (editable), .html, observations.csv
```

## Inventory (verified by asserts in `gen_companion.py`)
| Harness | Run | Attempts | Model / effort | Version | Route · environment | Task revision | Outcomes |
|---|---|---|---|---|---|---|---|
| Terminus 2 | TB2.1 leaderboard job #78 | 5 (all) | anthropic/claude-fable-5 · high | terminus-2 2.0.0 | Anthropic API (server-side Opus fallback enabled, **not triggered** on this task) · Daytona | sha256:1f1361b0… | 2 pass, 3 verifier fail |
| fast-agent | r9 | 1 | copilot/claude-fable-5 · high (all 41 LLM calls) | 0.10.32 | GitHub Copilot · HF Jobs hf-basic | same digest (pinned fork changes only qemu-alpine-ssh / qemu-startup) | 1 pass |
| fast-agent | r8 (earlier high run, 64/89 overall) | **not available** | copilot/claude-fable-5 · high | — | Copilot · HF Jobs | — | unknown |

- **r8 not located:** not in `hf://buckets/evalstate/fable-tb21-89x1`, other HF buckets, Harbor Hub (`harbor hub job list`), local job dirs or session histories. The r9 README says the adapter was patched after r8 to retain ATIF locally. r8 is shown as a "?" marker in an off-axis gutter: **cost unknown, not $0**. Add its trial record to plot it.
- **Excluded by design:** medium-effort runs (including medium r5 and its 4 setup replacements).
- **No timeouts, safety stops or infrastructure replacements** among the plotted attempts.
- **Cost basis:** $10/M uncached input, $1/M cached input, $50/M output, from each attempt's token counts. This is the main chart's basis and reproduces fast-agent's recorded cost exactly. Terminus's per-trial `cost_usd` was null for Fable trials, so it's priced from tokens, as the leaderboard total was.
- **"$7.36—and it failed."** refers to Terminus attempt `auErQfF` (group 4: 3.31M input tokens, 64.7k output, verifier fail). It's the highest-cost *available* attempt, but it isn't called the most expensive overall because r8 is unknown.
- `g1`–`g5` are the reconstructed groups from the main chart (attempt order by start time).

---

# Claude Code variant — `same_task_different_attempts_cc.*`

```bash
python3 gen_companion_cc.py   # -> same_task_different_attempts_cc.{png,svg,html} + observations_cc.csv
```
The second row is Claude Code in place of fast-agent. The fast-agent version above is kept.

| Harness | Run | Attempts | Model / effort | Version | Route · environment | Task revision | Outcomes |
|---|---|---|---|---|---|---|---|
| Claude Code | TB2.1 leaderboard job #75 (Hub job `11efb542…`) | 5 (all) | anthropic/claude-fable-5 · **xhigh** | claude-code 2.1.167 (`CLAUDE_CODE_SIMPLE=1`) | Anthropic API, with Claude Code's own `model_refusal_fallback` → claude-opus-4-8 · Daytona | sha256:1f1361b0… (same) | 3 pass, 1 verifier fail, 1 pass after Opus fallback |

- **Effort differs:** xhigh (Claude Code) vs high (Terminus). This is the only Claude Code + Fable 5 run with per-attempt data. The Strands-chart Claude Code figure has no per-attempt data.
- **`zWKshqF`:** Fable refused after 60 steps, Claude Code retried on Opus 4.8 for 17 steps, and the task passed. It's drawn as a hollow diamond at its **Fable-only** cost ($4.05). The Opus portion ($1.66 at the same rates) is excluded and disclosed. Under the main chart's rule it would count as a failure.
- **Costs re-priced from tokens** at $10/M uncached input (including cache writes), $1/M cached, $50/M output. Claude Code's self-reported `cost_usd` uses different, roughly Opus-class rates (e.g. $6.28 for the attempt that is $12.26 on our basis), so it isn't used.
- **The leaderboard total ($552.67) reconciles exactly** as $366.37 (Claude Code's own costs, 412 trials) + $186.30 (33 trials with no recorded cost, priced at $10/$1/$50). So that total mixes pricing bases.
- Source data: `../archive/source-data/claude-code-job/`. The five install-windows attempts are kept in full, plus every trial's `result.json`, the leaderboard submission and the row page.
