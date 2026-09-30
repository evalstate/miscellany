# Analysis and chart source

These analyses back the deck. They were copied from `~/source/general/` on 29 Sep 2026.
Each folder has its own README with rebuild steps, inclusion rules and caveats.

| Folder | What | Final output |
|---|---|---|
| `tb4-subset-calibration/` | A 19-task TB 4 subset calibrated against the public Terminal-Bench 4.0 leaderboard: score gap, rank correlation, cost multiplier | `subset_sheet.png`, `subset_scatter.png`, `subset_models.png`, `subset_cost.png` |
| `cc-comparison-charts/` | Claude Code default vs Minimal Mode vs fast-agent (Opus 5 / high, TB 2.1 7-task slice, 21 trials each) | `v14d_cost_hero_light_nochip.png`, `v15_dual_light_big.png` |
| `fable5-harness-charts/` | Fable 5 harness comparison on TB 2.1: Terminus 2 re-scored Fable-only, Strands, OpenCode, Oh-my-pi, fast-agent | `fable5_harness_comparison.png`, plus `install-windows-3.11/same_task_different_attempts.png` |

## Not copied (rebuildable)

- `tb4-subset-calibration/data/tasks/` (~500 MB): `harbor download terminal-bench/terminal-bench@4.0.0 -o data/tasks`
- `tb4-subset-calibration/data/{rows,jobs}/` (~21 MB): `python3 fetch.py` (Harbor Hub, ~20 s)
- `*/archive/` in the two chart folders (~280 MB of alternate candidates and synced source data):
  `prepare_data.py` re-syncs the source data, and `gen*.py --all` re-renders the alternates.

These paths are git-ignored in each folder, so re-running the pipelines won't stage them.

## Relevance to the deck

- Part 2 (What): the TB 4 cost multiplier and constraints. The 11 sidecar tasks were excluded
  as not runnable single-container on HF Jobs, which ties to the verifier-designs slide.
- Part 4 (Reading): cherry-pick disclosure ("CHERRY-PICKED FROM 2"), fallback re-scoring
  (Terminus served by Opus when Fable refused), CI vs run-to-run spread, "same task, different attempts".
