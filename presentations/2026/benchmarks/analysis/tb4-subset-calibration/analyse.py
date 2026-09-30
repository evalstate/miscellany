"""Calibration of the frozen TB4 subset against the public Terminal-Bench 4.0 leaderboard.

    python3 fetch.py     # -> data/rows/*.json (every trial of every leaderboard row)
    python3 analyse.py   # -> analysis.json
"""
import json, pathlib, random, statistics as st, collections, math, sys

ROOT = pathlib.Path(__file__).resolve().parent
S = json.load(open(ROOT / 'subset.json'))
N_RANDOM = 20000
random.seed(4)


def tn(t): return t['task_name'].split('/')[-1]


def spearman(a, b):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    return pearson(rk(a), rk(b))


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


rows = []
for f in sorted((ROOT / 'data' / 'rows').glob('*.json')):
    d = json.load(open(f)); r = d['row']; m = r['metadata']; x = r['metrics']; T = d['trials']
    by = collections.defaultdict(list)
    for t in T: by[tn(t)].append(t)
    succ = sum(t['reward'] == 1 for t in T)
    tsum = sum(t['cost_usd'] or 0 for t in T)
    rows.append(dict(
        id=r['id'], rank=r['rank'], agent=m['agent_display']['label'], model=m['model_display']['label'],
        effort=m['reasoning_effort'], date=m['date'],
        full_score=x['accuracy'], full_cost=x['total_cost_usd'],
        trial_successes=succ, pub_successes=x['successes'], trial_cost=tsum,
        null_costs=sum(t['cost_usd'] is None for t in T),
        # per task: passes (of 5), cost (known trials), null-cost count
        tasks={k: dict(p=sum(t['reward'] == 1 for t in v), n=len(v), c=sum(t['cost_usd'] or 0 for t in v),
                       nulls=sum(t['cost_usd'] is None for t in v)) for k, v in by.items()}))

TASKS = sorted(rows[0]['tasks'])
assert all(sorted(r['tasks']) == TASKS for r in rows) and len(TASKS) == 66 and set(S) <= set(TASKS)
REST = [t for t in TASKS if t not in S]

# ---------------------------------------------------------------- selection constraints (the eligible pool)
# 1. runnable on HF Jobs: single container (no docker-compose sidecars)   2. no GPU   3. no safety refusals
import tomllib
TDIR = ROOT / 'data' / 'tasks' / 'terminal-bench'
EXCL = {}
for t in TASKS:
    toml = tomllib.loads((TDIR / t / 'task.toml').read_text())
    if toml.get('environment', {}).get('gpus', 0) or toml.get('verifier', {}).get('environment', {}).get('gpus', 0):
        EXCL.setdefault(t, []).append('gpu')
    if any((TDIR / t / 'environment').glob('docker-compose*.y*ml')):
        EXCL.setdefault(t, []).append('multi-container')
REFUSED = set()
for f in (ROOT / 'data' / 'rows').glob('*.json'):
    for tr in json.load(open(f))['trials']:
        if tr['error_type'] == 'AgentSafetyRefusalError':
            REFUSED.add(tn(tr))
for t in sorted(REFUSED):
    EXCL.setdefault(t, []).append('safety refusal')
ELIGIBLE = [t for t in TASKS if t not in EXCL]
assert set(S) <= set(ELIGIBLE), set(S) - set(ELIGIBLE)
POOL = ELIGIBLE if '--all-tasks' not in sys.argv else TASKS

# ---------------------------------------------------------------- inclusion rules
for r in rows:
    r['score_ok'] = r['trial_successes'] == r['pub_successes']
    r['cost_reconciled'] = abs(r['trial_cost'] - r['full_cost']) < 0.5
    r['cost_ok'] = r['cost_reconciled'] and r['null_costs'] == 0
SC = [r for r in rows if r['score_ok']]
CO = [r for r in rows if r['cost_ok']]


def sub_score(r, tasks):
    return 100 * sum(r['tasks'][t]['p'] for t in tasks) / sum(r['tasks'][t]['n'] for t in tasks)


def sub_cost(r, tasks):
    return sum(r['tasks'][t]['c'] for t in tasks)


def score_stats(tasks, rows_=SC):
    s = [sub_score(r, tasks) for r in rows_]; F = [r['full_score'] for r in rows_]
    g = [a - b for a, b in zip(s, F)]
    return dict(mae=st.mean(abs(v) for v in g), signed=st.mean(g), w5=sum(abs(v) <= 5 for v in g),
                w10=sum(abs(v) <= 10 for v in g), spearman=spearman(s, F), pearson=pearson(s, F),
                max_abs=max(abs(v) for v in g))


def cost_stats(tasks, rows_=CO):
    """Leave-one-model-out: multiplier = median(full/subset) over rows of *other* models."""
    ratio = {r['id']: r['full_cost'] / sub_cost(r, tasks) for r in rows_}
    errs, naive = [], []
    for r in rows_:
        others = [ratio[o['id']] for o in rows_ if o['model'] != r['model']]
        k = st.median(others)
        errs.append(abs(k * sub_cost(r, tasks) - r['full_cost']) / r['full_cost'])
        naive.append((len(TASKS) / len(tasks)) * sub_cost(r, tasks) / r['full_cost'] - 1)
    rs = list(ratio.values())
    return dict(mae=st.mean(errs), worst=max(errs), median_ratio=st.median(rs), min_ratio=min(rs), max_ratio=max(rs),
                pooled_ratio=sum(r['full_cost'] for r in rows_) / sum(sub_cost(r, tasks) for r in rows_),
                naive_mean=st.mean(naive), naive_all_low=all(v < 0 for v in naive),
                cv_ratio=st.stdev(rs) / st.mean(rs),
                spearman=spearman([sub_cost(r, tasks) for r in rows_], [r['full_cost'] for r in rows_]))


F18 = list(S)
score = score_stats(F18); cost = cost_stats(F18)
# Held-out view: subset vs the other 48 tasks (no shared trials)
s18 = [sub_score(r, F18) for r in SC]; s48 = [sub_score(r, REST) for r in SC]
held = dict(mae=st.mean(abs(a - b) for a, b in zip(s18, s48)), spearman=spearman(s18, s48),
            signed=st.mean(a - b for a, b in zip(s18, s48)))

# ---------------------------------------------------------------- random 18-task baselines
def cost_share(tasks):
    return st.median(sub_cost(r, tasks) / r['full_cost'] for r in CO)


rs_mae, rs_sp, rc_mae, rc_cv, rs_share = [], [], [], [], []
for _ in range(N_RANDOM):
    k = random.sample(POOL, len(F18))
    a = score_stats(k); rs_mae.append(a['mae']); rs_sp.append(a['spearman']); rs_share.append(cost_share(k))
for _ in range(N_RANDOM // 10):
    k = random.sample(POOL, len(F18))
    c = cost_stats(k); rc_mae.append(c['mae']); rc_cv.append(c['cv_ratio'])


def pct_better(vals, v, lower=True):
    return sum((x < v) if lower else (x > v) for x in vals) / len(vals)


def q(v, p): return sorted(v)[int(p * (len(v) - 1))]


random_base = dict(
    score_mae_median=st.median(rs_mae), score_mae_p10=q(rs_mae, .1), score_mae_p90=q(rs_mae, .9),
    score_mae_share_better=pct_better(rs_mae, score['mae']),
    spearman_median=st.median(rs_sp), spearman_share_better=pct_better(rs_sp, score['spearman'], lower=False),
    cost_mae_median=st.median(rc_mae), cost_mae_p10=q(rc_mae, .1), cost_mae_p90=q(rc_mae, .9),
    cost_mae_share_better=pct_better(rc_mae, cost['mae']),
    share_median=st.median(rs_share))
F_SHARE = cost_share(F18)
cloud = [[round(a, 4), round(b, 3)] for a, b in zip(rs_share, rs_mae)]
cloud_cheaper = sum(v <= F_SHARE for v in rs_share) / N_RANDOM
cloud_dom = sum(a <= F_SHARE and b <= score['mae'] for a, b in zip(rs_share, rs_mae)) / N_RANDOM

# wall-time share of the subset (trial start -> finish), all rows
import datetime as _dt
_tt = collections.Counter()
for f in (ROOT / 'data' / 'rows').glob('*.json'):
    for t in json.load(open(f))['trials']:
        if t['started_at'] and t['finished_at']:
            du = (_dt.datetime.fromisoformat(t['finished_at']) - _dt.datetime.fromisoformat(t['started_at'])).total_seconds()
            _tt[tn(t) in S] += du
time_share = _tt[True] / (_tt[True] + _tt[False])

# ---------------------------------------------------------------- per-task profile
task_profile = []
for t in TASKS:
    pr = [r['tasks'][t]['p'] / r['tasks'][t]['n'] for r in SC]
    rest = [(sum(r['tasks'][u]['p'] for u in TASKS if u != t)) for r in SC]
    disc = pearson(pr, rest) if len(set(pr)) > 1 else 0.0
    rel = []   # task cost relative to that row's average task cost
    for r in CO:
        mean_task = r['full_cost'] / len(TASKS)
        rel.append(r['tasks'][t]['c'] / mean_task)
    task_profile.append(dict(task=t, subset=t in S, pass_rate=st.mean(pr), discrimination=disc,
                             rel_cost=st.median(rel), rel_cost_mean=st.mean(rel)))

# ---------------------------------------------------------------- per-row output
out_rows = []
for r in rows:
    k_loo = None
    if r['cost_ok']:
        others = [o['full_cost'] / sub_cost(o, F18) for o in CO if o['model'] != r['model']]
        k_loo = st.median(others)
    out_rows.append(dict(
        {k: r[k] for k in ('id', 'rank', 'agent', 'model', 'effort', 'date', 'full_score', 'full_cost', 'score_ok',
                           'cost_ok', 'cost_reconciled', 'null_costs', 'trial_successes', 'pub_successes')},
        sub_score=sub_score(r, F18), rest_score=sub_score(r, REST), sub_passes=sum(r['tasks'][t]['p'] for t in F18),
        sub_cost=sub_cost(r, F18), sub_cost_nulls=sum(r['tasks'][t]['nulls'] for t in F18),
        cost_share=sub_cost(r, F18) / r['full_cost'] if r['full_cost'] else None,
        ratio=r['full_cost'] / sub_cost(r, F18) if sub_cost(r, F18) else None,
        k_loo=k_loo, est_cost=k_loo * sub_cost(r, F18) if k_loo else None,
        naive_cost=sub_cost(r, F18) * len(TASKS) / len(F18),
        per_task={t: r['tasks'][t] for t in F18}))

digests = json.load(open(ROOT / 'data' / 'subset_digests_seen.json'))
exact_rows = {rid for rid in {d[0] for d in digests}} - {d[0] for d in digests if d[2] != S[d[1]]}

OUTF = ROOT / ('analysis.json' if POOL is ELIGIBLE else 'analysis_all66.json')
json.dump(dict(
    n_rows=len(rows), n_score=len(SC), n_cost=len(CO), n_models_score=len({r['model'] for r in SC}),
    n_models_cost=len({r['model'] for r in CO}), n_tasks=len(TASKS), subset=F18,
    excluded_score=[f"{r['model']} / {r['effort']}" for r in rows if not r['score_ok']],
    excluded_cost_null=[f"{r['model']} / {r['effort']}" for r in rows if r['cost_reconciled'] and r['null_costs']],
    excluded_cost_mismatch=[f"{r['model']} / {r['effort']}" for r in rows if not r['cost_reconciled']],
    eligible=ELIGIBLE, n_eligible=len(ELIGIBLE), excluded_tasks=EXCL, pool='eligible' if POOL is ELIGIBLE else 'all',
    f_share=F_SHARE, cloud=cloud, cloud_median_share=random_base['share_median'], cloud_cheaper=cloud_cheaper,
    cloud_dominates=cloud_dom, time_share=time_share,
    score=score, held_out=held, cost=cost, random=random_base, n_random=N_RANDOM,
    exact_digest_rows=len(exact_rows), other_digest_rows=len({d[0] for d in digests}) - len(exact_rows),
    other_digest_row_names=sorted({f"{r['model']} / {r['effort']}" for r in rows if r['id'] not in exact_rows}),
    tasks=task_profile, rows=out_rows), open(OUTF, 'w'), indent=1)

print(f"pool: {len(POOL)} tasks; excluded {len(EXCL)}: " + ', '.join(f'{k} ({"/".join(v)})' for k, v in sorted(EXCL.items())))
print(f"random cheaper-than-final18 {cloud_cheaper*N_RANDOM:.0f}/{N_RANDOM}, dominating {cloud_dom*N_RANDOM:.0f}; share median {random_base['share_median']:.3f}")
print(f"score rows {len(SC)}  MAE {score['mae']:.2f}  signed {score['signed']:+.2f}  ±5 {score['w5']}/{len(SC)}  "
      f"±10 {score['w10']}  spearman {score['spearman']:.3f}  pearson {score['pearson']:.3f}")
print(f"held-out (subset vs other tasks): MAE {held['mae']:.2f}  spearman {held['spearman']:.3f}")
print(f"random 18: MAE median {random_base['score_mae_median']:.2f} (p10 {random_base['score_mae_p10']:.2f}, p90 "
      f"{random_base['score_mae_p90']:.2f}); final18 beats {100*(1-random_base['score_mae_share_better']):.1f}% ; "
      f"spearman median {random_base['spearman_median']:.3f}")
print(f"cost rows {len(CO)} models {len({r['model'] for r in CO})}  LOMO MAE {cost['mae']*100:.1f}% worst {cost['worst']*100:.1f}%  "
      f"median ratio {cost['median_ratio']:.2f} ({cost['min_ratio']:.2f}–{cost['max_ratio']:.2f}) pooled {cost['pooled_ratio']:.2f}  "
      f"naive {cost['naive_mean']*100:+.1f}% all low {cost['naive_all_low']}  spearman {cost['spearman']:.3f}")
print(f"random 18 cost LOMO MAE median {random_base['cost_mae_median']*100:.1f}% (p10 {random_base['cost_mae_p10']*100:.1f} "
      f"p90 {random_base['cost_mae_p90']*100:.1f}); final18 beats {100*(1-random_base['cost_mae_share_better']):.1f}%")
for r in sorted(out_rows, key=lambda r: r['sub_score'] - r['full_score']):
    print(f"  {r['model'][:16]:16} {r['effort']:6} full {r['full_score']:5.1f} sub {r['sub_score']:5.1f} "
          f"gap {r['sub_score']-r['full_score']:+6.1f} rest {r['rest_score']:5.1f} ratio {r['ratio'] or 0:5.2f} "
          f"share {100*(r['cost_share'] or 0):4.1f}% ok={r['score_ok']},{r['cost_ok']}")
