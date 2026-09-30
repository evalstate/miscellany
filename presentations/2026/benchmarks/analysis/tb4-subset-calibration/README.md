# TB4 19-task subset: calibration against the public Terminal-Bench 4.0 leaderboard

**Sheets (plain-language, current):** `subset_sheet.png` shows the 19 tasks and subset vs full score for every
leaderboard entry. `subset_models.png` is the per-entry deep dive (passes of 5 on every subset task, surprises vs similar-scoring entries). `subset_scatter.png` is the diagonal plot (subset vs full score, ±5-point band). `subset_cost.png` shows the full-run multiplier and why it's 6.5× rather than 3.5×. Built by
`python3 gen2.py`. The earlier analytical sheets (random-subset cloud, task map) are in `archive/` and are
rebuilt by `gen.py`.

```bash
python3 fetch.py     # every trial of all 27 leaderboard rows -> data/rows/, data/jobs/  (harbor hub, ~20 s)
harbor download terminal-bench/terminal-bench@4.0.0 -o data/tasks   # task.toml / compose files for the constraints
python3 analyse.py --all-tasks   # -> analysis_all66.json (random baseline from all 66 tasks)
python3 analyse.py               # -> analysis.json (random baseline from the 48 eligible tasks; used by the sheets)
python3 gen2.py      # -> subset_sheet.png, subset_scatter.png, subset_models.png, subset_cost.png  (gen.py: older analytical sheets)
```

## Inclusion
- Score: 26 of 27 rows. Excluded Opus 5 / max: trial records show 173 successes; the published figure is 171.
- Cost: 15 of 27 rows (7 models). All 330 trial costs are present and sum to the published total.
  - Excluded: 10 rows with missing trial costs.
  - Excluded: GPT-5.6 Sol / max and Opus 5 / max, whose trials don't sum to the published total.
- Subset tasks are matched by name. 21 rows ran exactly the digests in `subset.json`. 6 rows (GPT-6 Astra ×5,
  Gemini 3.8 Flash) ran a different revision (`data/subset_digests_seen.json`).

## Subset
19 tasks: the original frozen 18 plus hof-topology-interpenetration (digest 2534e62c…, the revision 21 leaderboard rows ran). Added after the candidate screen in `candidates.py`; distributed-dedup (#1635), ks-solver-cpp (#1633), vf2-speedup-networkx (#1770), embedding-drift-monitor (#1574 / #1636) rejected for open verifier defects.

## Selection constraints (the eligible pool, 48 of 66 tasks)
Derived from the task packages and leaderboard trials in `analyse.py`:
- **Runnable on HF Jobs (single container):** 11 tasks with a `docker-compose` sidecar excluded (ctr-optimization,
  cumulative-layout-shift, freight-dispatch-shift, heat-pump-warranty, intrastat-meldung, kv-live-surgery,
  legacy-utility-triage, live-database-cutover, medical-claims-processing, nextjs-performance, payments-pipeline-fix).
- **No GPU:** fp8-rmsnorm-gemm, jax-speedrun-gpu, math-eval-grader.
- **No safety refusals:** tasks with any `AgentSafetyRefusalError` on the leaderboard (batched-eval-parity,
  interleaved-vigenere, kv-live-surgery, shadow-relay, uefi-bootkit). If the selection used a different refusal list
  (e.g. from our own runs), update `REFUSED` in `analyse.py`.

All 18 subset tasks satisfy all three. The random baselines below draw from the 48 eligible tasks
(all-66 figures in brackets).

## Results
| | final18 | typical random 18 from 48 eligible (20k draws) [all 66] |
|---|---|---|
| Average abs score gap vs full | 4.26 pts (signed −0.83) | 5.36 [5.34]; final18 beats 75% |
| Within ±5 / ±10 pts | 17/26 · 24/26 | |
| Spearman vs full | 0.92 | 0.94 [0.95]; 73% of random picks rank better |
| Held out vs other 48 tasks only | 5.85 pts, Spearman 0.89 | |
| Cost share of a full run | 14.7% (wall time 15.2%) | 22.3% [26.7%]; 196/20,000 as cheap, 40 cheaper and closer [19 / 3] |
| Full ÷ subset cost | median 6.81× (5.75–8.99), pooled 6.78× | |
| Estimate error, target model held out | 9.2% avg, worst 24.3% | 17.0% [12.6%]; final18 beats 95% [81%] |
| Naive ×330/90 | 48% too low, every row | |

Cost decomposition (task cost relative to an average TB4 task): excluded tasks 1.43×, eligible pool 0.84×,
subset 0.52× (0.54× on the median row). The constraints account for part of the 6.8×; the selection for the rest.

Outliers: GPT-5.6 Luna / max +11.6 pts; Fable 5.1 / low −12.2 pts.
The subset also compresses the range: it underestimates strong rows and overestimates weak ones. A linear
recalibration (fitted with the target model held out) does not improve the average gap (4.16).

## Caveats
- This is retrospective. The subset was chosen with this leaderboard in view, so these numbers flatter it relative
  to a fresh model.
- Rows share models, so they aren't independent.
- The 18 tasks are part of the full score. The held-out comparison against the other 48 tasks is the stricter test.
- Some subset tasks barely separate models on this leaderboard: cargo-flight-dispatch is never passed, and
  vllm-deepseek-streaming, react-lead-form and atrx-vep-crispr show a negative item–rest correlation. That may be
  intentional (harness diagnostics), but it costs ranking precision.

## Visual style
The sheets follow the fast-agent "Forward" design guide (`~/Downloads/Fast-Agent benchmark splash/fast-agent-brand`)
but carry no fast-agent branding:
- Ivory `#FFF7E8` ground, petrol `#082C34` ink, muted text at 72% petrol (the lightest the guide allows).
- Figtree for text, with tabular numerals. Fraunces 900 (SOFT 100, −0.035em) for headlines and section titles.
  DM Mono for task names. 0.14em uppercase for small labels.
- Series: full benchmark = petrol; subset = teal `#277C80`; misses and surprises = orange `#F45125`, used only as a
  marker, never as text (3.3:1 contrast). Amber is reserved for fast-agent series, so it isn't used.
- Cards: paper fill with a 2px petrol keyline and radius 14. No zebra fills, KPI tiles, gradients or shadows. One flat
  tint band per chart (the ±5-point band).
- The previous Inter-styled generator is in `archive/gen2_inter_style.py`.
