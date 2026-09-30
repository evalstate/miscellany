# Claude Code Minimal Mode comparison — charts

**Publish candidates** (1500×900 @2x, 5:3):
- `v14d_cost_hero_light_nochip.png` — **preferred**: cost first; score + 7×3 waffle on the right (no +46% badge;
  the badged version is `archive/charts/v14_cost_hero_light.png`)
- `v15_dual_light_big.png` — score then cost; 7×3 waffle beneath the score bar

Publish-round changes (v14d and v15): "Pass rate" column heading; an **Exploratory: 7 tasks × 3 attempts per
configuration** pill under the subtitle; auth routes on the rows and in the smallprint; the confound note
"Mode, authentication route and concurrency differ; this is not an isolated estimate of the mode's effect." as the
first smallprint line. Zero-based cost bars and all numbers unchanged.

## Rebuild
```bash
python3 prepare_data.py        # (re)builds trials.json; syncs the bucket into archive/source-data if missing (--sync to force)
python3 gen2.py                # renders both publish candidates into this folder
python3 gen2.py --all          # + round-2/3 alternates  -> archive/charts
python3 gen.py                 # round-1 alternates      -> archive/charts
```
Needs: `hf` CLI (only for syncing), `chromium` (headless render), network for Google Fonts (Inter / JetBrains Mono).

## Files
| Path | What |
|---|---|
| `prepare_data.py` | reads ATIF traces + summaries → `trials.json`; Claude Code default = 18 originals + 3 replacements |
| `gen.py` | shared helpers (Clawd sprite, fast-agent logo, themes, footnotes) + round-1 variants |
| `gen2.py` | publish candidates (`v_cost_hero3`, `v_dual_big3`) + round-2/3 variants |
| `trials.json` | per-trial data for `oauth`, `bare`, `fa`, `oauth_orig` |
| `assets/` | fast-agent brand SVGs (from `fast-agent/docs/docs/assets/brand/`) |
| `archive/charts/` | all other candidate charts |
| `archive/html/` | HTML/SVG source for every render |
| `archive/contact_sheet.jpg` | overview of the archived candidates |
| `archive/source-data/` | local copy of `hf://buckets/evalstate/fable-tb21-89x1/cc-comparison/` |

## Numbers (21 trials each, Opus 5 / high, TB2.1 7-task slice)
| | Passed | Cost | $/pass |
|---|---|---|---|
| Claude Code 2.1.278 default (OAuth, replacement-selected) | 15/21 | $29.57 | $1.97 |
| Claude Code 2.1.278 Minimal Mode (API) | 16/21 | $20.29 | $1.27 |
| fast-agent 0.10.27 | 16/21 | $17.64 | $1.10 |

Score gaps are within noise (task-clustered 95% CI ≈ 52–95% for all arms). The cost gap is structural:
default's first call carries ~16k prompt tokens vs ~2k, and it writes 1h-TTL cache (505k tokens) where the others use 5m.
