"""Score each eligible non-subset task as a +1 addition (and as a swap for the weakest subset tasks)."""
import sys, json
sys.argv = ['analyse.py']
src = open('analyse.py').read().split('# ---------------------------------------------------------------- random')[0]
exec(src)
A = json.load(open(ROOT / 'analysis.json')); tp = {t['task']: t for t in A['tasks']}
base_s, base_c = score_stats(F18), cost_stats(F18)
def held(tasks):
    rest = [t for t in TASKS if t not in tasks]
    a = [sub_score(r, tasks) for r in SC]; b = [sub_score(r, rest) for r in SC]
    return st.mean(abs(x - y) for x, y in zip(a, b)), spearman(a, b)
def row(name, tasks):
    s, c = score_stats(tasks), cost_stats(tasks); h = held(tasks)
    share = st.median(sub_cost(r, tasks) / r['full_cost'] for r in CO)
    return dict(name=name, mae=s['mae'], sp=s['spearman'], w5=s['w5'], held_mae=h[0], held_sp=h[1],
                cost_mae=c['mae'], ratio=c['median_ratio'], share=share)
out = [row('(current 18)', F18)]
cands = [t for t in ELIGIBLE if t not in S]
for t in cands:
    r = row('+ ' + t, F18 + [t]); r.update(pass_rate=tp[t]['pass_rate'], disc=tp[t]['discrimination'], rel=tp[t]['rel_cost'])
    out.append(r)
json.dump(out, open(ROOT / 'candidates.json', 'w'), indent=1)
b = out[0]
print(f"{'':34} {'gap':>5} {'rank':>5} {'±5':>3} {'hgap':>5} {'hrank':>5} {'costE':>6} {'share':>6} | pass  disc  relcost")
for r in [b] + sorted(out[1:], key=lambda r: (r['sp'] - b['sp']) * 10 - (r['mae'] - b['mae']) / 2, reverse=True):
    ex = f"| {r['pass_rate']:.2f} {r['disc']:+.2f} {r['rel']:.2f}" if 'disc' in r else ''
    print(f"{r['name'][:34]:34} {r['mae']:5.2f} {r['sp']:5.3f} {r['w5']:3d} {r['held_mae']:5.2f} {r['held_sp']:5.3f} {r['cost_mae']*100:5.1f}% {r['share']*100:5.1f}% {ex}")

# ---- greedy additions and robustness
def obj(tasks, rows_=SC):
    s = score_stats(tasks, rows_); return s['mae'] - 10 * (s['spearman'] - 0.9)
cur = list(F18); print('\ngreedy additions:')
for step in range(3):
    best = min((t for t in cands if t not in cur), key=lambda t: obj(cur + [t]))
    cur.append(best); r = row('', cur)
    print(f"  +{best:30} n={len(cur)} gap {r['mae']:.2f} rank {r['sp']:.3f} ±5 {r['w5']} held {r['held_mae']:.2f}/{r['held_sp']:.3f} costE {r['cost_mae']*100:.1f}% share {r['share']*100:.1f}% ratio {r['ratio']:.2f}")
# stability: best single addition when each model's rows are held out
picks = collections.Counter()
models = sorted({r['model'] for r in SC})
for m in models:
    rs = [r for r in SC if r['model'] != m]
    picks[min(cands, key=lambda t: obj(F18 + [t], rs))] += 1
print('\nbest single addition, leave-one-model-out selection:', picks.most_common())
# does the addition help the held-out model?
gain = collections.defaultdict(list)
for m in models:
    rs = [r for r in SC if r['model'] != m]; t = min(cands, key=lambda t: obj(F18 + [t], rs))
    for r in SC:
        if r['model'] == m:
            gain[t].append(abs(sub_score(r, F18 + [t]) - r['full_score']) - abs(sub_score(r, F18) - r['full_score']))
print('held-out change in |gap| for the held-out model (negative = better):', {k: round(st.mean(v), 2) for k, v in gain.items()},
      'overall', round(st.mean(x for v in gain.values() for x in v), 2))
# swap the never-passed task
print('\nswap cargo-flight-dispatch (never passed) ->')
base = [t for t in F18 if t != 'cargo-flight-dispatch']
for t in sorted(cands, key=lambda t: obj(base + [t]))[:5]:
    r = row('', base + [t]); print(f"  {t:30} gap {r['mae']:.2f} rank {r['sp']:.3f} ±5 {r['w5']} costE {r['cost_mae']*100:.1f}% share {r['share']*100:.1f}%")
