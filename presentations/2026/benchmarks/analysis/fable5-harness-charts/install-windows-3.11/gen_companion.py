"""Companion chart: individual-attempt cost on install-windows-3.11, Fable 5 / high.

    python3 gen_companion.py      # -> same_task_different_attempts.{png,svg,html} + observations.csv

Inventory (verified by the asserts below; see README.md):
  * Terminus 2 v2.0.0 / anthropic/claude-fable-5 / high: 5 attempts from the TB2.1 leaderboard job (PR #78),
    Daytona, task digest sha256:1f1361b0...  No attempt on this task was served by the Opus fallback.
  * fast-agent 0.10.32 / copilot/claude-fable-5 / high: run r9, 1 attempt, HF Jobs, same task digest.
  * fast-agent high r8 (the earlier run omitted from the main chart): per-attempt data NOT available
    (not in hf://buckets/evalstate/fable-tb21-89x1, other buckets, Harbor Hub or local job dirs) -> shown as
    "not available, cost unknown", never as $0.
  * Medium-effort runs excluded by design.
Costs use the main chart's token-price basis: $10/M uncached input, $1/M cached input, $50/M output.
"""
import csv, datetime as dt, json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import gen  # noqa: E402  (shared helpers: text, logos, lockup, theme)

SRC = ROOT / 'archive' / 'source-data'
TASK = 'install-windows-3.11'
DIGEST = 'sha256:1f1361b0012e1ad24054e7cfc039829459e8ab0631b3573d9b93499bf6b3563e'
RATES = dict(uncached=10e-6, cached=1e-6, out=50e-6)
R9 = SRC / 'fable5-high-copilot-hf-r9-1x-qemu-fixed-01032' / '20260924-114859'


def price(uncached, cached, out):
    return uncached * RATES['uncached'] + cached * RATES['cached'] + out * RATES['out']


def secs(a, b):
    f = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    return (f(b) - f(a)).total_seconds()


def terminus_attempts():
    job = next((SRC / 'terminus-job').glob('*/'))
    rows = []
    for d in sorted(job.glob(f'{TASK}__*/')):
        r = json.load(open(d / 'result.json')); ar = r['agent_result']; cfg = r['config']['agent']
        traj = json.load(open(d / 'agent' / 'trajectory.json'))
        models = {s.get('model_name') for s in traj['steps'] if s.get('source') == 'agent'}
        assert cfg['model_name'] == 'anthropic/claude-fable-5' and cfg['kwargs']['reasoning_effort'] == 'high'
        assert r['agent_info']['name'] == 'terminus-2' and r['agent_info']['version'] == '2.0.0'
        assert r['task_id']['ref'] == DIGEST, r['task_id']['ref']
        assert models == {'claude-fable-5'}, f'fallback on {d.name}: {models}'   # no Opus-served steps
        unc = ar['n_input_tokens'] - ar['n_cache_tokens']
        exc = (r.get('exception_info') or {}).get('exception_type')
        rew = r['verifier_result']['rewards']['reward']
        rows.append(dict(harness='Terminus 2', run='TB2.1 #78', attempt_id=d.name.split('__')[1], start_utc=r['started_at'],
                         outcome='pass' if rew == 1 else 'fail', error_type=exc or '', cost_usd=price(unc, ar['n_cache_tokens'], ar['n_output_tokens']),
                         uncached_in=unc, cached_in=ar['n_cache_tokens'], output_tokens=ar['n_output_tokens'],
                         agent_seconds=round(secs(r['agent_execution']['started_at'], r['agent_execution']['finished_at'])),
                         model='anthropic/claude-fable-5', effort='high', harness_version='terminus-2 2.0.0',
                         route='Anthropic API (server-side Opus fallback enabled; not triggered on this task)', environment='Daytona',
                         task_digest=DIGEST, note=''))
    rows.sort(key=lambda r: r['start_utc'])
    for i, r in enumerate(rows): r['group'] = i + 1          # same group numbering as the main chart
    return rows


def fast_agent_attempts():
    conf = json.load(open(R9 / 'configuration.json'))
    assert conf['model'] == 'copilot/claude-fable-5' and conf['reasoning'] == 'high' and conf['version'] == '0.10.32'
    assert TASK not in conf['qemu_repair_tasks']
    t = next(json.loads(l) for l in open(R9 / 'trials.jsonl') if json.loads(l)['task_name'] == TASK)
    traj = json.load(open(R9 / 'trials' / t['trial_name'] / 'trajectory.json'))
    ms = [s['metrics'] for s in traj['steps'] if s.get('metrics')]
    assert {(m.get('extra') or {}).get('reasoning_effort') for m in ms} == {'high'}
    assert {(m.get('extra') or {}).get('provider') for m in ms} == {'copilot'}
    unc = t['input_tokens'] - t['cached_input_tokens']
    c = price(unc, t['cached_input_tokens'], t['output_tokens'])
    assert abs(c - t['cost_usd']) < 1e-6, (c, t['cost_usd'])            # recorded cost uses the same basis
    ok = [dict(harness='fast-agent', run='r9', attempt_id=t['trial_name'].split('__')[1], start_utc=t['started_at'],
               outcome='pass' if t['reward'] == 1 else 'fail', error_type=t['error_type'] or '', cost_usd=c,
               uncached_in=unc, cached_in=t['cached_input_tokens'], output_tokens=t['output_tokens'],
               agent_seconds=round(t['elapsed_seconds']), model='copilot/claude-fable-5', effort='high',
               harness_version='fast-agent 0.10.32', route='GitHub Copilot', environment='HF Jobs (hf-basic)',
               task_digest=DIGEST + ' (pinned TB2.1 fork; only QEMU tasks differ)', note='', group='')]
    missing = [dict(harness='fast-agent', run='r8', attempt_id='', start_utc='', outcome='not available', error_type='',
                    cost_usd=None, uncached_in=None, cached_in=None, output_tokens=None, agent_seconds=None,
                    model='copilot/claude-fable-5', effort='high', harness_version='fast-agent (earlier high run)',
                    route='GitHub Copilot', environment='HF Jobs', task_digest='',
                    note='earlier high run (64/89 overall); per-attempt data not located; cost unknown, not zero', group='')]
    return ok, missing


# ---------------------------------------------------------------- drawing
W, H = 1500, 690
th = gen.THEMES['light']
T = gen.text
AX0, AX1, CMAX = 330, 1250, 8.0
GUTTER = 1372   # unknown-cost markers live here, off the dollar axis
X = lambda c: AX0 + (AX1 - AX0) * c / CMAX
CHAR = th['lb']            # charcoal (Terminus)
AMB = gen.AMBER


def marker(x, y, outcome, col):
    if outcome == 'pass':
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{col}" stroke="{th["bg"]}" stroke-width="2"/>'
    d = 8.5
    return (f'<path d="M{x-d:.1f},{y-d:.1f} L{x+d:.1f},{y+d:.1f} M{x-d:.1f},{y+d:.1f} L{x+d:.1f},{y-d:.1f}" '
            f'stroke="{th["bg"]}" stroke-width="8" stroke-linecap="round"/>'
            f'<path d="M{x-d:.1f},{y-d:.1f} L{x+d:.1f},{y+d:.1f} M{x-d:.1f},{y+d:.1f} L{x+d:.1f},{y-d:.1f}" '
            f'stroke="{col}" stroke-width="4.5" stroke-linecap="round"/>')


def offsets(rows, min_dx=70, step=26):
    """vertical offsets so markers (and their labels) never overlap."""
    out, placed = {}, []
    for r in sorted(rows, key=lambda r: r['cost_usd']):
        x = X(r['cost_usd']); k = 0
        while any(abs(x - px) < min_dx and pk == k for px, pk in placed):
            k = -k if k > 0 else -k + 1      # 0, 1, -1, 2, -2 ...
        placed.append((x, k)); out[id(r)] = k * step
    return out


def build(term, fa, missing):
    b = T(60, 82, 'Same task. Different attempts.', 38, th['fg'], 700)
    b += T(60, 120, 'Installing Windows 3.11 · Fable 5 · high reasoning', 19, th['fg1'], 400)
    b += gen.fa_lockup(W - 60 - gen.LOCK_W(28), 54, 28, th['fa_text'])
    top, rowh = 178, 150
    rows = [('Terminus 2', 'terminus_best', term, CHAR, f'v2.0.0 · {len(term)} attempts · TB2.1 leaderboard job'),
            ('fast-agent', 'fa_high', fa, AMB, f'v0.10.32 · {len(fa)} available attempt · run r9')]
    ybot = top + len(rows) * rowh
    # grid + axis ($0 start, shared)
    for c in range(0, int(CMAX) + 1):
        b += f'<line x1="{X(c):.1f}" y1="{top}" x2="{X(c):.1f}" y2="{ybot}" stroke="{th["grid"]}" stroke-width="{1.6 if c == 0 else 1.1}"/>'
        b += T(X(c), ybot + 24, f'${c}', 14, th['fg2'], 500, 'middle')
    b += T(AX1, ybot + 50, 'Cost per attempt (token-price estimate) →', 15, th['fg2'], 600, 'end')
    for i, (name, key, obs, col, sub) in enumerate(rows):
        y = top + i * rowh; cy = y + rowh / 2
        if i: b += f'<line x1="60" y1="{y}" x2="{W-60}" y2="{y}" stroke="{th["line"]}"/>'
        b += gen.icon(key, 60, cy - 26, 44, th)
        b += T(118, cy - 6, name, 24, th['fg'], 700)
        b += T(118, cy + 16, sub, 13.5, th['fg2'], 500)
        off = offsets(obs)
        for r in obs:
            x = X(r['cost_usd']); py = cy + off[id(r)]
            hl = r['harness'] == 'Terminus 2' and abs(r['cost_usd'] - 7.3647) < 1e-3 and r['outcome'] == 'fail'
            if hl:
                b += f'<circle cx="{x:.1f}" cy="{py:.1f}" r="19" fill="{col}" opacity="0.10"/>'
            b += marker(x, py, r['outcome'], col)
            b += T(x, py - 20, '$7.36—and it failed.' if hl else f"${r['cost_usd']:.2f}", 18 if hl else 17, th['fg'], 800 if hl else 700, 'middle')
            ident = r['attempt_id'] + (f" · g{r['group']}" if r['group'] else f" · {r['run']}")
            b += T(x, py + 30, ident, 11.5, th['fg2'], 500, 'middle', family=gen.MONO)
        if key == 'fa_high':
            for m in missing:   # unknown cost: outside the dollar axis, never at $0
                gx = GUTTER
                b += (f'<rect x="{gx-15}" y="{cy-15}" width="30" height="30" rx="6" fill="none" stroke="{AMB}" '
                      f'stroke-width="2" stroke-dasharray="4 3"/>' + T(gx, cy + 7, '?', 19, AMB, 800, 'middle'))
                b += T(gx, cy - 24, 'cost unknown', 12.5, th['fg1'], 600, 'middle')
                b += T(gx, cy + 34, f"{m['run']} · not available", 11.5, th['fg2'], 500, 'middle', family=gen.MONO)
    b += f'<line x1="{AX1 + 44}" y1="{top + 10}" x2="{AX1 + 44}" y2="{ybot - 10}" stroke="{th["line"]}" stroke-width="1.5" stroke-dasharray="4 4"/>'
    b += T(GUTTER, ybot + 24, 'not on axis', 13, th['fg2'], 500, 'middle')
    # legend
    ly = ybot + 50; lx = 60
    b += marker(lx + 10, ly - 5, 'pass', th['fg1']) + T(lx + 28, ly, 'pass', 14, th['fg2'], 500)
    b += marker(lx + 92, ly - 5, 'fail', th['fg1']) + T(lx + 110, ly, 'fail (verifier)', 14, th['fg2'], 500)
    b += (f'<rect x="{lx+222}" y="{ly-15}" width="20" height="20" rx="4" fill="none" stroke="{th["fg1"]}" stroke-width="1.6" stroke-dasharray="3 2.5"/>'
          + T(lx + 232, ly, '?', 13, th['fg1'], 800, 'middle') + T(lx + 250, ly, 'attempt data not available', 14, th['fg2'], 500))
    # caption + footnotes
    cy0 = ybot + 92
    b += T(60, cy0, 'Every available qualifying attempt shown; this task was selected to illustrate variation. '
           'It does not establish typical variability across the benchmark.', 15, th['fg1'], 600)
    t = term; nfail = sum(r['outcome'] == 'fail' for r in t)
    foot = [
        f'Terminus 2 v2.0.0 · anthropic/claude-fable-5 · effort high · Daytona · all {len(t)} attempts from the TB2.1 leaderboard job (#78), '
        f'{len(t) - nfail} pass / {nfail} fail; none used the Opus fallback. "g" = reconstructed group, as in the main chart.',
        'fast-agent 0.10.32 · copilot/claude-fable-5 · effort high · Copilot route · HF Jobs · run r9, 1 attempt (pass). Earlier high run r8 (64/89 overall): '
        'per-attempt data not available, so its cost is unknown, not zero. Medium-effort runs excluded.',
        'Same task revision for all (sha256:1f1361b0…; our pinned TB2.1 fork changes only the two QEMU tasks). No timeouts, safety stops or infrastructure '
        'replacements among these attempts. Environments and inference routes differ.',
        'Cost = token-price estimate at $10/M uncached input, $1/M cached input, $50/M output (same basis as the main chart); not bills, excludes compute.',
    ]
    fy = cy0 + 36
    b += f'<line x1="60" y1="{fy-20}" x2="{W-60}" y2="{fy-20}" stroke="{th["line"]}"/>'
    for i, l in enumerate(foot):
        b += T(60, fy + i * 18, l, 11.5, th['fg2'], 400)
    return b


def write(body):
    font = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
            "&amp;family=JetBrains+Mono:wght@400;500;700&amp;display=swap');")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<style>{font}</style><defs>{gen.defs(th)}</defs><rect width="{W}" height="{H}" fill="{th["bg"]}"/>{body}</svg>')
    (HERE / 'same_task_different_attempts.svg').write_text(svg)
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;width:{W}px;height:{H}px;'
            f'overflow:hidden;background:{th["bg"]}}}</style></head><body>{svg}</body></html>')
    p = HERE / 'same_task_different_attempts.html'; p.write_text(html)
    png = HERE / 'same_task_different_attempts.png'
    subprocess.run(['chromium', '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--virtual-time-budget=4000', '--force-device-scale-factor=2', f'--screenshot={png}', f'file://{p}'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
    print('wrote', png)


def write_csv(rows):
    cols = ['harness', 'run', 'attempt_id', 'group', 'start_utc', 'outcome', 'error_type', 'cost_usd', 'uncached_in',
            'cached_in', 'output_tokens', 'agent_seconds', 'model', 'effort', 'harness_version', 'route', 'environment',
            'task_digest', 'note']
    with open(HERE / 'observations.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else (f'{r[k]:.4f}' if k == 'cost_usd' else r[k])) for k in cols})
    print('wrote', HERE / 'observations.csv')


if __name__ == '__main__':
    term = terminus_attempts()
    fa, missing = fast_agent_attempts()
    write_csv(term + fa + missing)
    write(build(term, fa, missing))
