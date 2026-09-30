"""Assemble data.json for the Fable 5 harness comparison.

Sources
- OURS: fast-agent Fable 5 medium (r5, replacement-selected) and high (r9) runs from
  hf://buckets/evalstate/fable-tb21-89x1/  (synced into archive/source-data if missing; --sync to force)
- LEADERBOARD: Terminus 2 / Fable 5 / high, TB2.1 main leaderboard row ce0677b9-0fea-46ce-b8de-893c4d68e77a
  (headline metrics fetched live; falls back to the saved copy in archive/source-data).
  The full job (445 trials = 89 tasks x 5) is downloaded with `harbor hub job download` into
  archive/source-data/terminus-job and re-scored FABLE-ONLY:
    * a trial is a "fallback" trial if any agent step was served by claude-opus-4-8 (server-side fallback);
      fallback trials score 0 and carry $0 -- the same rule our runs get for Fable safety stops;
    * Fable cost is priced from each trial's tokens at $10/M uncached in, $1/M cached, $50/M out (the rates the
      leaderboard used for Fable: its $438.64 = $70.20 LiteLLM-priced Opus trials + Fable tokens at these rates);
    * "runs" 1..5 = the k-th attempt of each task by start time; best run = highest Fable-only score (ties -> cheaper).
- PUBLISHED (Strands chart): Strands / OpenCode / Oh-my-pi figures transcribed from the Strands
  "frontier accuracy on Terminal Bench 2.1 at lower cost" chart (Claude Fable 5, 89 trials each).
  Deepseek Harness and Claude Code rows intentionally dropped.
"""
import collections, json, math, pathlib, re, statistics as st, subprocess, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / 'archive' / 'source-data'
BUCKET = 'hf://buckets/evalstate/fable-tb21-89x1/'
RUNS = {
    'fa_medium': ('fable5-medium-copilot-hf-r5-1x-qemu-fixed-01032', '20260923-073703',
                  'replacements/20260923-100028/selected-trials.jsonl'),
    'fa_high': ('fable5-high-copilot-hf-r9-1x-qemu-fixed-01032', '20260924-114859', 'trials.jsonl'),
}
# Other runs of the same configuration that are NOT in the bucket (operator-provided). The chart discloses that the
# fast-agent high row is the higher-scoring of two runs.
OTHER_RUNS = {'fa_high': [dict(passes=65, n=89, cost=68.60, note='operator-provided (corrected after infra failure); earlier high run, traces not in the bucket')]}
ROW_URL = ('https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest/'
           'leaderboards/main/rows/ce0677b9-0fea-46ce-b8de-893c4d68e77a')
PUBLISHED = [  # transcribed from the Strands chart; accuracy %, cost USD for 89 trials
    dict(key='strands', name='Strands', score=0.697, cost=56.29),
    dict(key='opencode', name='OpenCode', score=0.663, cost=73.42),
    dict(key='ohmypi', name='Oh-my-pi', score=0.697, cost=86.83),
]


def ours(key):
    run, stamp, rel = RUNS[key]
    base = SRC / run / stamp
    if '--sync' in sys.argv or not (base / rel).exists():
        subprocess.run(['hf', 'buckets', 'sync', f'{BUCKET}{run}/', str(SRC / run)], check=True)
    trials = [json.loads(l) for l in open(base / rel)]
    conf = json.load(open(base / 'configuration.json'))
    p = sum(1 for t in trials if t['reward'] == 1)
    return dict(key=key, name='fast-agent', effort=conf['reasoning'], version=conf['version'],
                source='ours', n=len(trials), passes=p, score=p / len(trials),
                cost=sum(t['cost_usd'] for t in trials),
                out_tokens=sum(t['output_tokens'] for t in trials),
                errors={e: sum(1 for t in trials if t['error_type'] == e)
                        for e in sorted({t['error_type'] for t in trials if t['error_type']})},
                tasks={t['task_name']: t['reward'] for t in trials},
                task_refused={t['task_name']: t['error_type'] == 'AgentSafetyStopError' for t in trials},
                ci95=binom_ci(p / len(trials), len(trials)), ci_method='binomial, 1 x 89',
                other_runs=OTHER_RUNS.get(key, []))


TERM_JOB = '17f04e4f-1a75-4204-9b75-d042ef0333ec'
RATES = dict(uncached=10e-6, cached=1e-6, out=50e-6)


def binom_ci(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)


def terminus_leaderboard():
    try:
        html = urllib.request.urlopen(ROW_URL, timeout=30).read().decode()
        (SRC / 'terminus-leaderboard-row.html').write_text(html)
    except Exception as e:  # offline: use saved copy
        print('fetch failed, using saved copy:', e)
        html = (SRC / 'terminus-leaderboard-row.html').read_text()
    h = html.replace('\\"', '"')
    m = re.search(r'"agent_display":\{[^}]*"label":"([^"]+)"\},"model_display":\{[^}]*"label":"([^"]+)"\},'
                  r'"reasoning_effort":"([^"]+)"\},"metrics":(\{.*?\}),"status"', h)
    agent, model, effort, metrics = m.group(1), m.group(2), m.group(3), json.loads(m.group(4))
    pr = re.search(r'"pr_url":\{"url":"([^"]+)","label":"([^"]+)"', h)
    return dict(name=agent, model=model, effort=effort, pr=pr.group(2), pr_url=pr.group(1), metrics=metrics)


def terminus_trials():
    jobdir = SRC / 'terminus-job'
    if '--sync' in sys.argv or not any(jobdir.glob('*/*/result.json')):
        subprocess.run(['harbor', 'hub', 'job', 'download', TERM_JOB, '-o', str(jobdir), '--overwrite'], check=True)
    out = []
    for res in sorted(jobdir.glob('*/*/result.json')):
        r = json.load(open(res)); traj = json.load(open(res.parent / 'agent' / 'trajectory.json'))
        models = collections.Counter(s.get('model_name') for s in traj['steps'] if s.get('source') == 'agent')
        ar = r.get('agent_result') or {}
        reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward', 0) or 0
        inp, cache, outp = ar.get('n_input_tokens') or 0, ar.get('n_cache_tokens') or 0, ar.get('n_output_tokens') or 0
        fallback = any(m and 'opus' in m for m in models)
        out.append(dict(trial=res.parent.name, task=res.parent.name.split('__')[0], start=r['started_at'],
                        reward=1 if reward == 1 else 0, fallback=fallback, models=dict(models),
                        recorded_cost=ar.get('cost_usd') or 0,
                        fable_cost=0.0 if fallback else (inp - cache) * RATES['uncached'] + cache * RATES['cached'] + outp * RATES['out']))
    return out


def terminus():
    lb = terminus_leaderboard(); T = terminus_trials()
    by = collections.defaultdict(list)
    for t in T: by[t['task']].append(t)
    runs = collections.defaultdict(list)
    for v in by.values():
        for k, t in enumerate(sorted(v, key=lambda t: t['start'])): runs[k].append(t)
    run_rows = []
    for k in sorted(runs):
        v = runs[k]; p = sum(t['reward'] for t in v if not t['fallback'])
        run_rows.append(dict(run=k + 1, n=len(v), passes=p, score=p / len(v), cost=sum(t['fable_cost'] for t in v),
                             fallbacks=sum(t['fallback'] for t in v), passes_incl_opus=sum(t['reward'] for t in v)))
    n = len(T); p = sum(t['reward'] for t in T if not t['fallback']); attempts = n // 89
    task_means = [sum(t['reward'] for t in v if not t['fallback']) / len(v) for v in by.values()]
    se_task = st.stdev(task_means) / math.sqrt(len(task_means))
    common = dict(name=lb['name'], model=lb['model'], effort=lb['effort'], source='leaderboard', pr=lb['pr'],
                  pr_url=lb['pr_url'], lb_score=lb['metrics']['accuracy'] / 100, lb_ci95=lb['metrics']['accuracy_ci95_half_width'] / 100,
                  lb_cost_total=lb['metrics']['total_cost_usd'], n_total=n, attempts=attempts,
                  fallback_trials=sum(t['fallback'] for t in T),
                  fallback_passes=sum(t['reward'] for t in T if t['fallback']),
                  opus_cost_recorded=sum(t['recorded_cost'] for t in T if t['fallback']),
                  fallback_tasks=sorted({t['task'] for t in T if t['fallback']}), runs=run_rows)
    avg = dict(common, key='terminus_avg', variant='avg', n=n, passes=p, score=p / n,
               cost=sum(t['fable_cost'] for t in T) / attempts, ci95=1.96 * se_task,
               ci_method='task-clustered SE over 89 tasks (5 attempts each)')
    best = max(run_rows, key=lambda r: (r['score'], -r['cost']))
    worst = min(run_rows, key=lambda r: (r['score'], -r['cost']))  # lowest score; ties -> more expensive
    bst = dict(common, key='terminus_best', variant='best', run=best['run'], n=89, passes=best['passes'], score=best['score'],
               cost=best['cost'], ci95=binom_ci(best['score'], 89), ci_method='binomial, 1 x 89')
    wst = dict(common, key='terminus_worst', variant='worst', run=worst['run'], n=89, passes=worst['passes'],
               score=worst['score'], cost=worst['cost'], ci95=binom_ci(worst['score'], 89), ci_method='binomial, 1 x 89')
    json.dump(T, open(SRC / 'terminus-trials-rescored.json', 'w'), indent=0)
    avg['tasks'] = {task: sum(t['reward'] for t in v if not t['fallback']) / len(v) for task, v in by.items()}
    avg['task_refused'] = {task: sum(t['fallback'] for t in v) / len(v) for task, v in by.items()}
    bk = best['run'] - 1
    pick = {task: sorted(v, key=lambda t: t['start'])[bk] for task, v in by.items()}
    bst['tasks'] = {task: 0 if t['fallback'] else t['reward'] for task, t in pick.items()}
    bst['task_refused'] = {task: t['fallback'] for task, t in pick.items()}
    wk = worst['run'] - 1
    pick = {task: sorted(v, key=lambda t: t['start'])[wk] for task, v in by.items()}
    wst['tasks'] = {task: 0 if t['fallback'] else t['reward'] for task, t in pick.items()}
    wst['task_refused'] = {task: t['fallback'] for task, t in pick.items()}
    return [avg, bst, wst]


def main():
    rows = [ours('fa_medium'), ours('fa_high')] + terminus() + \
           [dict(r, source='strands_chart', model='Fable 5', n=89, ci95=binom_ci(r['score'], 89),
                 ci_method='binomial, 1 x 89 (assumed)') for r in PUBLISHED]
    for r in rows:
        r['cost_per_pass'] = r['cost'] / (r['score'] * 89)
    json.dump(rows, open(ROOT / 'data.json', 'w'), indent=1)
    for r in rows:
        print(f"{r['key']:14s} {r['source']:14s} {r['score']*100:5.1f}% ±{r['ci95']*100:.1f}  ${r['cost']:6.2f}/89  ${r['cost_per_pass']:.2f}/pass")
    t = next(r for r in rows if r['key'] == 'terminus_avg')
    print(f"terminus: {t['fallback_trials']} fallback trials ({t['fallback_passes']} passed on Opus, ${t['opus_cost_recorded']:.2f} Opus cost) excluded;",
          f"leaderboard {t['lb_score']*100:.1f}% ±{t['lb_ci95']*100:.1f} ${t['lb_cost_total']:.2f}")
    for r in t['runs']:
        print(f"   run {r['run']}: {r['passes']}/89 fable-only ({r['score']*100:.1f}%), ${r['cost']:.2f}, {r['fallbacks']} fallbacks")
    print('wrote', ROOT / 'data.json')


if __name__ == '__main__':
    main()
