"""Companion chart (Claude Code variant): individual-attempt cost on install-windows-3.11, Fable 5.

    python3 gen_companion_cc.py   # -> same_task_different_attempts_cc.{png,svg,html} + observations_cc.csv

Rows: Terminus 2 (high) vs Claude Code (xhigh) -- both TB2.1 leaderboard jobs on Daytona, same task digest.
  * Claude Code + Fable 5 exists on the leaderboard only at xhigh (PR #75, Claude Code 2.1.167); flagged, not hidden.
  * Claude Code attempt zWKshqF hit a mid-run refusal: Claude Code's `model_refusal_fallback` retried on
    claude-opus-4-8 (17 of 77 steps) and the task passed. Shown with its own marker, at its Fable-only cost;
    the Opus portion is disclosed and excluded (the main chart's rule would count the attempt as a failure).
  * Costs re-priced from tokens on the main chart's basis ($10/M uncached input incl. cache writes, $1/M cached,
    $50/M output). Claude Code's self-reported cost_usd uses different (Opus-class) rates and is NOT used; the
    leaderboard's $552.67 total mixes those self-reported costs (412 trials) with $10/$1/$50 pricing (33 trials).
"""
import collections, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_companion as base          # noqa: E402  Terminus loader, pricing, writers, shared gen helpers
gen, T, th = base.gen, base.T, base.th

CC_JOB = next((base.SRC / 'claude-code-job').glob('*/'))
CLAUDE = '#D77757'


def clawd(x, y, s, fill, eye):
    """Clawd sprite, as on the Claude Code comparison chart."""
    px = {(qx, qy) for qx in range(3, 15) for qy in range(4)} - {(5, 1), (12, 1)}
    px |= {(q, 2) for q in (1, 2, 15, 16)} | {(q, 4) for q in (4, 6, 11, 13)}
    out = [f'<g transform="translate({x:.1f},{y:.1f})" shape-rendering="crispEdges">']
    out += [f'<rect x="{qx*s}" y="{qy*2*s}" width="{s+0.05}" height="{2*s+0.05}" fill="{fill}"/>' for qx, qy in px]
    out += [f'<rect x="{qx*s}" y="{qy*2*s}" width="{s}" height="{2*s}" fill="{eye}"/>' for qx, qy in ((5, 1), (12, 1))]
    return ''.join(out) + '</g>'


def split_by_model(traj):
    agg = collections.defaultdict(collections.Counter)
    for s in traj['steps']:
        if s.get('source') != 'agent' or not s.get('metrics'): continue
        m = s['metrics']; c = agg[s.get('model_name')]
        c['prompt'] += m.get('prompt_tokens') or 0; c['cached'] += m.get('cached_tokens') or 0
        c['out'] += m.get('completion_tokens') or 0; c['steps'] += 1
    return agg


def claude_code_attempts():
    rows = []
    for d in sorted(CC_JOB.glob(f'{base.TASK}__*/')):
        r = json.load(open(d / 'result.json')); ar = r['agent_result']; cfg = r['config']['agent']
        traj = json.load(open(d / 'agent' / 'trajectory.json'))
        assert cfg['model_name'] == 'anthropic/claude-fable-5' and cfg['kwargs']['reasoning_effort'] == 'xhigh'
        assert r['agent_info']['name'] == 'claude-code' and r['agent_info']['version'] == '2.1.167'
        assert r['task_id']['ref'] == base.DIGEST and r['config']['environment']['type'] == 'daytona'
        agg = split_by_model(traj)
        assert sum(c['prompt'] for c in agg.values()) == ar['n_input_tokens']      # step split reconciles
        fab = agg['claude-fable-5']; opus = agg.get('claude-opus-4-8')
        fcost = base.price(fab['prompt'] - fab['cached'], fab['cached'], fab['out'])
        ocost = base.price(opus['prompt'] - opus['cached'], opus['cached'], opus['out']) if opus else 0.0
        rew = r['verifier_result']['rewards']['reward']
        exc = (r.get('exception_info') or {}).get('exception_type')
        outcome = ('pass after Opus fallback' if opus and rew == 1 else 'fail after Opus fallback' if opus
                   else 'pass' if rew == 1 else 'fail')
        rows.append(dict(harness='Claude Code', run='TB2.1 #75', attempt_id=d.name.split('__')[1], start_utc=r['started_at'],
                         outcome=outcome, error_type=exc or '', cost_usd=fcost,
                         uncached_in=fab['prompt'] - fab['cached'], cached_in=fab['cached'], output_tokens=fab['out'],
                         agent_seconds=round(base.secs(r['agent_execution']['started_at'], r['agent_execution']['finished_at'])),
                         model='anthropic/claude-fable-5', effort='xhigh', harness_version='claude-code 2.1.167',
                         route='Anthropic API (Claude Code model_refusal_fallback -> claude-opus-4-8)', environment='Daytona',
                         task_digest=base.DIGEST,
                         note=(f"Fable-only cost plotted; Opus 4.8 portion excluded: {opus['steps']} steps, ${ocost:.2f} at the same rates. "
                               f"Claude Code self-reported ${ar['cost_usd']:.2f} (different rates)") if opus else
                              f"Claude Code self-reported ${ar['cost_usd']:.2f} (different rates, not used)",
                         opus_cost=ocost, opus_steps=opus['steps'] if opus else 0))
    rows.sort(key=lambda r: r['start_utc'])
    for i, r in enumerate(rows): r['group'] = i + 1
    return rows


# ---------------------------------------------------------------- drawing
W, H = 1500, 756
AX0, AX1, CMAX = 330, 1400, 13.0
X = lambda c: AX0 + (AX1 - AX0) * c / CMAX
base.X = X                     # offsets() in the base module uses X


def marker(x, y, outcome, col):
    if outcome.endswith('after Opus fallback'):   # hollow diamond: passed/failed only with Opus assistance
        d = 11
        return (f'<path d="M{x},{y-d} L{x+d},{y} L{x},{y+d} L{x-d},{y} Z" fill="{th["bg"]}" stroke="{col}" stroke-width="3.2" '
                f'stroke-linejoin="round"/>')
    return base.marker(x, y, outcome, col)


def build(term, cc):
    b = T(60, 82, 'Same task. Different attempts.', 38, th['fg'], 700)
    b += T(60, 120, 'Installing Windows 3.11 · Fable 5 · Terminus 2 at high, Claude Code at xhigh reasoning', 19, th['fg1'], 400)
    b += gen.fa_lockup(W - 60 - gen.LOCK_W(28), 54, 28, th['fa_text'])
    top, rowh = 178, 150
    rows = [('Terminus 2', term, base.CHAR, f'v2.0.0 · high · {len(term)} attempts · TB2.1 leaderboard job #78'),
            ('Claude Code', cc, CLAUDE, f'v2.1.167 · xhigh · {len(cc)} attempts · TB2.1 leaderboard job #75')]
    ybot = top + len(rows) * rowh
    for c in range(0, int(CMAX) + 1):
        b += f'<line x1="{X(c):.1f}" y1="{top}" x2="{X(c):.1f}" y2="{ybot}" stroke="{th["grid"]}" stroke-width="{1.6 if c == 0 else 1.1}"/>'
        b += T(X(c), ybot + 24, f'${c}', 14, th['fg2'], 500, 'middle')
    b += T(AX1, ybot + 50, 'Cost per attempt (token-price estimate) →', 15, th['fg2'], 600, 'end')
    for i, (name, obs, col, sub) in enumerate(rows):
        y = top + i * rowh; cy = y + rowh / 2
        if i: b += f'<line x1="60" y1="{y}" x2="{W-60}" y2="{y}" stroke="{th["line"]}"/>'
        if name == 'Terminus 2':
            b += gen.icon('terminus_best', 60, cy - 26, 44, th)
        else:
            b += clawd(58, cy - 22, 2.7, CLAUDE, th['bg'])
        b += T(118, cy - 6, name, 24, th['fg'], 700)
        b += T(118, cy + 16, sub.split(' · ', 2)[0] + ' · ' + sub.split(' · ', 2)[1], 13.5, th['fg2'], 500)
        b += T(118, cy + 34, sub.split(' · ', 2)[2], 12.5, th['fg2'], 400)
        # two tiers: labels would collide within ~108 px, so the later marker drops to a lower tier
        off = base.offsets(obs, min_dx=100, step=1)
        for r in obs:
            tier = off[id(r)]
            x = X(r['cost_usd']); py = cy - 14 if tier == 0 else cy + 38
            hl = r['harness'] == 'Terminus 2' and abs(r['cost_usd'] - 7.3647) < 1e-3 and r['outcome'] == 'fail'
            if hl: b += f'<circle cx="{x:.1f}" cy="{py:.1f}" r="19" fill="{col}" opacity="0.10"/>'
            if tier:   # thin tick back to the row's baseline so the offset reads as an offset, not a value change
                b += f'<line x1="{x:.1f}" y1="{cy-14}" x2="{x:.1f}" y2="{py-12}" stroke="{col}" stroke-width="1.2" stroke-dasharray="2 3" opacity="0.6"/>'
            b += marker(x, py, r['outcome'], col)
            lab = '$7.36—and it failed.' if hl else f"${r['cost_usd']:.2f}" + ('*' if r.get('opus_steps') else '')
            ident = f"{r['attempt_id']} · g{r['group']}"
            if tier == 0:
                b += T(x, py - 20, lab, 18 if hl else 16.5, th['fg'], 800 if hl else 700, 'middle')
                b += T(x, py + 30, ident, 11, th['fg2'], 500, 'middle', family=gen.MONO)
            else:      # labels to the right of the lower-tier marker
                b += T(x + 18, py + 1, lab, 16.5, th['fg'], 700)
                b += T(x + 18, py + 17, ident, 11, th['fg2'], 500, family=gen.MONO)
    # legend
    ly = ybot + 50; lx = 60
    b += marker(lx + 10, ly - 5, 'pass', th['fg1']) + T(lx + 28, ly, 'pass', 14, th['fg2'], 500)
    b += marker(lx + 92, ly - 5, 'fail', th['fg1']) + T(lx + 110, ly, 'fail (verifier)', 14, th['fg2'], 500)
    b += marker(lx + 236, ly - 5, 'pass after Opus fallback', th['fg1']) + T(lx + 254, ly, 'passed after mid-run Opus fallback*', 14, th['fg2'], 500)
    cy0 = ybot + 92
    b += T(60, cy0, 'Every available qualifying attempt shown; this task was selected to illustrate variation. '
           'It does not establish typical variability across the benchmark.', 15, th['fg1'], 600)
    zw = next(r for r in cc if r.get('opus_steps'))
    ccf = sum(r['outcome'] == 'fail' for r in cc); ccp = sum(r['outcome'] == 'pass' for r in cc)
    tf = sum(r['outcome'] == 'fail' for r in term)
    foot = [
        f'Terminus 2 v2.0.0 · anthropic/claude-fable-5 · effort high · all {len(term)} attempts from TB2.1 leaderboard job #78 ({len(term)-tf} pass / {tf} fail); none used its Opus fallback.',
        f'Claude Code 2.1.167 · anthropic/claude-fable-5 · effort xhigh · all 5 attempts from TB2.1 leaderboard job #75 ({ccp} pass / {ccf} fail / 1 pass after fallback). '
        'Both Daytona; same task revision (sha256:1f1361b0…).',
        f"* {zw['attempt_id']}: Fable refused mid-run; Claude Code retried on Opus 4.8 ({zw['opus_steps']} of {zw['opus_steps'] + 60} steps) and the task passed. "
        f"Plotted at its Fable-only ${zw['cost_usd']:.2f}; the ${zw['opus_cost']:.2f} Opus portion is excluded. The main chart's rule would count it as a failure.",
        'Effort differs (xhigh is the only Claude Code + Fable 5 run with per-attempt data). No timeouts, safety stops or infrastructure replacements. '
        '"g" = reconstructed group (attempt order by start time).',
        'Cost = token-price estimate at $10/M uncached input (incl. cache writes), $1/M cached, $50/M output, the main chart\'s basis. '
        "Claude Code's self-reported costs use lower rates and are not used.",
    ]
    fy = cy0 + 36
    b += f'<line x1="60" y1="{fy-20}" x2="{W-60}" y2="{fy-20}" stroke="{th["line"]}"/>'
    for i, l in enumerate(foot):
        b += T(60, fy + i * 18, l, 11.5, th['fg2'], 400)
    return b


def write(body, stem='same_task_different_attempts_cc'):
    font = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
            "&amp;family=JetBrains+Mono:wght@400;500;700&amp;display=swap');")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<style>{font}</style><defs>{gen.defs(th)}</defs><rect width="{W}" height="{H}" fill="{th["bg"]}"/>{body}</svg>')
    (HERE / f'{stem}.svg').write_text(svg)
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;width:{W}px;height:{H}px;'
            f'overflow:hidden;background:{th["bg"]}}}</style></head><body>{svg}</body></html>')
    p = HERE / f'{stem}.html'; p.write_text(html)
    png = HERE / f'{stem}.png'
    base.subprocess.run(['chromium', '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                         '--virtual-time-budget=4000', '--force-device-scale-factor=2', f'--screenshot={png}', f'file://{p}'],
                        stdout=base.subprocess.DEVNULL, stderr=base.subprocess.DEVNULL, timeout=90)
    print('wrote', png)


if __name__ == '__main__':
    term = base.terminus_attempts()
    cc = claude_code_attempts()
    rows = term + cc
    import csv
    cols = ['harness', 'run', 'attempt_id', 'group', 'start_utc', 'outcome', 'error_type', 'cost_usd', 'uncached_in',
            'cached_in', 'output_tokens', 'agent_seconds', 'model', 'effort', 'harness_version', 'route', 'environment',
            'task_digest', 'note']
    with open(HERE / 'observations_cc.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.4f}' if k == 'cost_usd' else r.get(k, '')) for k in cols})
    print('wrote', HERE / 'observations_cc.csv')
    write(build(term, cc))
