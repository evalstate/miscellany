# Fable 5 harness comparison — charts

Terminus 2 (TB2.1 leaderboard, re-scored **Fable-only**) and Strands / OpenCode / Oh-my-pi (as published in the
Strands chart) vs our fast-agent Fable 5 medium and high runs. Deepseek Harness and Claude Code excluded.

**Final chart:** `fable5_harness_comparison.png` (1500×900 @2x, 5:3) — sorted by cost; Terminus shown as its
highest- and lowest-scoring reconstructed 89-task groups; 89-cell accuracy waffles (passes first, Fable refusals /
fallbacks marked); cost as a dot plot on a labelled $50–$90 axis.

Disclosures on the chart:
- fast-agent high carries a **CHERRY-PICKED FROM 2** badge plus "Highest-scoring of two runs shown; other run: 65/89
  (73.0%), $68.60." The other run's cost is also drawn as a dashed "other run" marker on the cost axis. Its traces are
  not in the bucket; score and cost are operator-provided (`OTHER_RUNS` in `prepare_data.py`), corrected from an
  earlier 64/89 figure after an infra failure. Previous render: `archive/previous/`.
  A 20 px icon slot is reserved left of the badge: drop `assets/cherry.svg` (or `.png`) in and re-run `gen.py`.
- Terminus: fallback trials counted as failures; costs of those trials excluded. The "groups" are reconstructed by
  ordering each task's attempts by start time — they are not separately executed, fallback-disabled runs.

```bash
python3 prepare_data.py   # -> data.json  (syncs our runs; downloads the Terminus job via `harbor hub job download` if missing)
python3 gen.py            # -> fable5_harness_comparison.png
python3 gen.py --all      # + every candidate -> archive/candidates/ and archive/contact_sheet.jpg
```

| | Source | Accuracy (95% CI) | Total cost, 89 tasks |
|---|---|---|---|
| fast-agent 0.10.32 · medium | ours (1×89, incl. 4 setup replacements) | 75.3% ± 9.0 | $58.39 |
| fast-agent 0.10.32 · high | ours (1×89) | 75.3% ± 9.0 | $73.22 |
| Terminus 2 · high · best run | leaderboard job, Fable-only, run 5 of 5 | 69.7% ± 9.6 | $69.60 |
| Terminus 2 · high · 5-run avg | leaderboard job, Fable-only, 5×89 | 68.1% ± 8.8 | $73.69 |
| Strands | Strands chart | 69.7% ± 9.5 | $56.29 |
| Oh-my-pi | Strands chart | 69.7% ± 9.5 | $86.83 |
| OpenCode | Strands chart | 66.3% ± 9.8 | $73.42 |

## Terminus Fable-only re-scoring
- The leaderboard job ran with Anthropic's server-side fallback (`fallback: claude-opus-4-8`).
  79 of 445 trials were served by Opus 4.8 (Fable refused) across 16 tasks; 55 of them passed.
- Those trials score 0 and their $70.20 Opus cost is removed — the same treatment our runs get,
  where Fable safety stops score 0.
- Fable cost is priced from tokens at $10/M uncached, $1/M cached, $50/M output. This reproduces the leaderboard total
  exactly: $438.64 = $70.20 (Opus trials) + Fable tokens at these rates. Same rates as our runs.
- Groups 1–5 = the k-th attempt of each task by start time (ties: lower cost for highest-scoring, higher cost for lowest-scoring). Adjusted: 62/89, 61/89, 59/89, 59/89, 62/89.
- Published leaderboard figure for reference: 80.5% ± 2.3, $438.64 for 5×89.

## Error bars
95% CI from task sampling. Single 89-task runs: binomial ±1.96·√(p(1−p)/89). Terminus 5-run average: task-clustered
(SD of the 89 per-task mean scores ÷ √89, ×1.96). The leaderboard's own ±2.3 is run-to-run spread only
(SD of the 5 run scores ÷ √5), so it is not comparable with single-run intervals.

QEMU: our runs use pinned, infrastructure-only fixes for `qemu-alpine-ssh` and `qemu-startup` (needed on HF Jobs;
Terminus passed both 5/5 on the unmodified tasks). Stated in the chart smallprint.

Folders: `archive/candidates/` all other options · `archive/round1/` first round (pre-fallback-adjustment) ·
`archive/html/` SVG/HTML sources · `archive/source-data/` bucket runs, Terminus job, saved leaderboard pages.

## Third-party logos (`assets/third-party/`)
| Row | File | Source |
|---|---|---|
| Strands | `strands-logo-light.svg` (on a #0e0e0e tile) | github.com/strands-agents/docs `site/src/assets/` |
| OpenCode | `opencode-favicon.svg` | github.com/anomalyco/opencode `packages/app/public/favicon-v3.svg` |
| Oh-my-pi | `omp-favicon.svg` | github.com/can1357/oh-my-pi `packages/collab-web/public/favicon.svg` |
| Terminus 2 | `terminal-bench-fav.png` | tbench.ai `/fav.png` (Terminal-Bench org mark; no separate Terminus logo found) |

Sort order: accuracy descending (rounded to 0.1 pt), ties by lower cost.
