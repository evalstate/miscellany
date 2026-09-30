"""Build trials.json from the cc-comparison bucket traces.

- Syncs hf://buckets/evalstate/fable-tb21-89x1/cc-comparison/ into archive/source-data if missing
  (or pass --sync to force a re-sync).
- Arms: 'bare' (Minimal Mode), 'fa' (fast-agent), 'oauth' (Claude Code default, replacement-selected:
  18 originals + 3 replacements), 'oauth_orig' (the 21 originals, for reference).
- Prints a per-arm summary (score, cost, bootstrap CI, tokens, first-call prompt size, cache TTL mix).
"""
import collections, glob, json, pathlib, random, statistics as st, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / 'archive' / 'source-data'
BUCKET = 'hf://buckets/evalstate/fable-tb21-89x1/cc-comparison/'


def sync():
    SRC.mkdir(parents=True, exist_ok=True)
    subprocess.run(['hf', 'buckets', 'sync', BUCKET, str(SRC)], check=True)


def load_trial(summary_path):
    d = json.load(open(summary_path))
    steps = json.load(open(summary_path.replace('summary.json', 'trajectory.json'))).get('steps', [])
    ms = [s['metrics'] for s in steps if s.get('metrics')]
    w1 = w5 = 0
    for m in ms:
        e = m.get('extra', {})
        cc = e.get('cache_creation') or ((e.get('raw_usage') or [{}])[0].get('cache_creation')) or {}
        w1 += cc.get('ephemeral_1h_input_tokens', 0) or 0
        w5 += cc.get('ephemeral_5m_input_tokens', 0) or 0
    return dict(task=d['task_name'], trial=d['trial_name'], reward=d['reward'], err=d['error_type'],
                cost=d['cost_usd'] or 0, inp=d['input_tokens'], cached=d['cached_input_tokens'],
                outp=d['output_tokens'], secs=d['elapsed_seconds'], steps=len(steps),
                tools=sum(len(s.get('tool_calls') or []) for s in steps), calls=len(ms),
                first_prompt=ms[0]['prompt_tokens'] if ms else 0, cache_w_1h=w1, cache_w_5m=w5,
                replacement_for=None)


def arm(folder):
    return [load_trial(f) for f in sorted(glob.glob(str(SRC / folder / '*' / 'summary.json')))]


def boot_ci(trials, n=20000):
    random.seed(0)
    by = collections.defaultdict(list)
    for t in trials: by[t['task']].append(1 if t['reward'] == 1 else 0)
    groups = list(by.values()); bs = []
    for _ in range(n):
        s = [x for _ in groups for x in random.choice(groups)]
        bs.append(sum(s) / len(s))
    bs.sort(); return bs[int(n * .025)], bs[int(n * .975)]


def main():
    if '--sync' in sys.argv or not (SRC / 'manifest.json').exists():
        sync()
    T = {'bare': arm('claude-code-bare/trials'), 'fa': arm('fast-agent/trials'),
         'oauth_orig': arm('claude-code-oauth/originals')}
    reps = {t['trial']: t for t in arm('claude-code-oauth/replacements')}
    rmap = {r['original_trial']: r['replacement_trial']
            for r in json.load(open(SRC / 'claude-code-oauth' / 'replacement-map.json'))['replacements']}
    sel = []
    for t in T['oauth_orig']:
        if t['trial'] in rmap:
            r = dict(reps[rmap[t['trial']]]); r['replacement_for'] = t['trial']; sel.append(r)
        else:
            sel.append(t)
    T['oauth'] = sel
    json.dump(T, open(ROOT / 'trials.json', 'w'), indent=1)

    for k in ('oauth', 'bare', 'fa', 'oauth_orig'):
        t = T[k]; p = sum(1 for x in t if x['reward'] == 1); c = sum(x['cost'] for x in t)
        lo, hi = boot_ci(t); costs = [x['cost'] for x in t]
        print(f"{k:10s} {p}/{len(t)} ({p/len(t):.1%}, CI {lo:.0%}-{hi:.0%})  ${c:.2f}  ${c/p:.2f}/pass  "
              f"sd ${st.stdev(costs):.2f}  in {sum(x['inp'] for x in t)/1e6:.1f}M  "
              f"first-call median {st.median(x['first_prompt'] for x in t if x['first_prompt']):.0f} tok  "
              f"cache-write 1h/5m {sum(x['cache_w_1h'] for x in t)/1e3:.0f}k/{sum(x['cache_w_5m'] for x in t)/1e3:.0f}k")
    print('wrote', ROOT / 'trials.json')


if __name__ == '__main__':
    main()
