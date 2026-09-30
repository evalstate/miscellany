"""Companion chart v2 — install-windows-3.11, one bar per attempt (Terminus 2 vs Claude Code, Fable 5).

    python3 gen_companion_v2.py   # -> same_task_different_attempts.{png,svg,html} + observations.csv

Same data, checks and pricing as gen_companion_cc.py (see README.md); only the presentation changes:
per harness, a summary (passes, cost range) and one zero-based bar per attempt, sorted by cost.
"""
import csv, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_companion as base          # noqa: E402
import gen_companion_cc as cc_mod     # noqa: E402
gen, T, th = base.gen, base.T, base.th

W, H = 1500, 830
CLAUDE = cc_mod.CLAUDE
BX0, BX1, CMAX = 470, 1210, 13.0            # bar area ($0 .. $13)
X = lambda c: BX0 + (BX1 - BX0) * c / CMAX
NR = W - 60                                  # right edge for cost values


def glyph(x, y, kind, col, s=1.0):
    if kind == 'pass':
        return f'<circle cx="{x}" cy="{y}" r="{7*s}" fill="{col}"/>'
    if kind == 'fail':
        d = 6 * s
        return (f'<path d="M{x-d},{y-d} L{x+d},{y+d} M{x-d},{y+d} L{x+d},{y-d}" stroke="{col}" '
                f'stroke-width="{3.2*s}" stroke-linecap="round"/>')
    d = 8 * s  # diamond: needed the Opus fallback
    return f'<path d="M{x},{y-d} L{x+d},{y} L{x},{y+d} L{x-d},{y} Z" fill="{th["bg"]}" stroke="{col}" stroke-width="{2.6*s}"/>'


OPUS = '#7b6f9e'   # Opus 4.8 usage (muted violet), hatched


def opus_defs():
    return (f'<pattern id="hatch_opus" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
            f'<rect width="6" height="6" fill="{th["bg"]}"/><rect width="2.6" height="6" fill="{OPUS}"/></pattern>')


def effort_badge(x, y, label, col):
    w = len(label) * 17 * 0.62 + 22
    return (f'<rect x="{x}" y="{y-21}" width="{w:.0f}" height="28" rx="14" fill="{col}" opacity="0.14"/>'
            f'<rect x="{x+0.75}" y="{y-20.25}" width="{w-1.5:.0f}" height="26.5" rx="13.25" fill="none" stroke="{col}" stroke-width="1.5"/>'
            + T(x + w / 2, y - 1, label, 17, col if col != base.CHAR else th['fg'], 800, 'middle'))


def kind(r):
    return 'fallback' if r['outcome'].endswith('Opus fallback') else r['outcome']


def bar(x0, y, w, h, k, col):
    if k == 'pass':
        return f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="{h}" rx="4" fill="{col}"/>'
    if k == 'fail':
        return (f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="{h}" rx="4" fill="{col}" opacity="0.22"/>'
                f'<rect x="{x0+0.75}" y="{y+0.75}" width="{w-1.5:.1f}" height="{h-1.5}" rx="3.5" fill="none" stroke="{col}" stroke-width="1.5" opacity="0.7"/>')
    return (f'<rect x="{x0+0.9}" y="{y+0.9}" width="{w-1.8:.1f}" height="{h-1.8}" rx="3.5" fill="{th["bg"]}" stroke="{col}" '
            f'stroke-width="1.8" stroke-dasharray="5 3"/>')


def group(b, y, name, effort, sub, icon, obs, col):
    n = len(obs); p = sum(kind(r) == 'pass' for r in obs); fb = sum(kind(r) == 'fallback' for r in obs)
    tot = lambda r: r['cost_usd'] + r.get('opus_cost', 0.0)          # full attempt cost (Fable + any Opus)
    lo, hi = min(tot(r) for r in obs), max(tot(r) for r in obs)
    step, bh = 36, 22
    gh = n * step
    # left: identity + summary
    b += icon
    b += T(128, y + 26, name, 26, th['fg'], 700)
    b += effort_badge(128 + len(name) * 26 * 0.56 + 14, y + 26, effort, col)
    b += T(128, y + 50, sub, 14.5, th['fg2'], 500)
    b += T(60, y + 106, f'{p} / {n}', 40, th['fg'], 800)
    b += T(60 + 108, y + 106, 'passed on Fable', 17, th['fg1'], 600)
    ry = y + 140
    if fb:
        b += T(60, y + 132, f'{fb} additional pass after Opus fallback', 15.5, th['fg1'], 600)
        ry = y + 170
    rng = f'${lo:.2f} – ${hi:.2f}'
    b += T(60, ry, rng, 24, th['fg'], 700)
    b += T(60 + len(rng) * 24 * 0.56 + 14, ry, f'{hi/lo:.1f}× spread', 16, th['fg1'], 600)
    # right: one bar per attempt, sorted by cost
    for i, r in enumerate(sorted(obs, key=tot)):
        k = kind(r); cy = y + 6 + i * step + bh / 2
        b += glyph(BX0 - 22, cy, k, col)
        b += bar(BX0, cy - bh / 2, X(r['cost_usd']) - BX0, bh, k, col)
        if k == 'fallback':   # distinct extension for the Opus 4.8 usage in the same attempt
            x1, x2 = X(r['cost_usd']), X(tot(r))
            b += (f'<rect x="{x1:.1f}" y="{cy-bh/2}" width="{x2-x1:.1f}" height="{bh}" rx="3" fill="url(#hatch_opus)"/>'
                  f'<rect x="{x1+0.75:.1f}" y="{cy-bh/2+0.75}" width="{x2-x1-1.5:.1f}" height="{bh-1.5}" rx="3" fill="none" stroke="{OPUS}" stroke-width="1.5"/>')
        hl = r['harness'] == 'Terminus 2' and abs(r['cost_usd'] - 7.3647) < 1e-3 and k == 'fail'
        if hl: lab = '$7.36—and it failed.'
        elif k == 'fallback': lab = f"${tot(r):.2f} total · ${r['cost_usd']:.2f} Fable*"
        else: lab = f"${r['cost_usd']:.2f}"
        b += T(X(tot(r)) + 12, cy + 6, lab, 17 if not hl else 18, th['fg'], 800 if hl else 700)
        if k == 'fallback':
            b += T(X(tot(r)) + 12 + len(lab) * 17 * 0.505 + 8, cy + 5, f"+ ${r['opus_cost']:.2f} Opus", 14, OPUS, 700)
        b += T(NR, cy + 4, f"{r['attempt_id']} · g{r['group']}", 11.5, th['fg2'], 500, 'end', family=gen.MONO)
    return b, max(gh, 150)


def build(term, cc):
    b = T(60, 82, 'Single task cost variance', 38, th['fg'], 700)
    b += T(60, 120, 'Installing Windows 3.11 · Fable 5 · Terminus 2 at high, Claude Code at xhigh reasoning', 19, th['fg1'], 400)
    b += gen.fa_lockup(W - 60 - gen.LOCK_W(28), 54, 28, th['fa_text'])
    top = 176
    # column heads
    b += T(BX0, top, 'Cost per attempt', 22, th['fg2'], 700)
    b += T(BX0 + 190, top, '· token-price estimate, $0 baseline', 15, th['fg2'], 400)
    b += T(NR, top, 'attempt · group', 13, th['fg2'], 500, 'end')
    y = top + 22
    groups = [('Terminus 2', 'high', 'v2.0.0 · Fable 5 · TB2.1 job #78', gen.icon('terminus_best', 60, y + 2, 50, th), term, base.CHAR),
              ('Claude Code', 'xhigh', 'v2.1.167 · Fable 5 · TB2.1 job #75', None, cc, CLAUDE)]
    ys = []
    for i, (name, effort, sub, icon, obs, col) in enumerate(groups):
        if i:
            b += f'<line x1="60" y1="{y-14}" x2="{W-60}" y2="{y-14}" stroke="{th["line"]}"/>'
        if icon is None:
            icon = cc_mod.clawd(57, y + 12, 3.1, CLAUDE, th['bg'])
        b, gh = group(b, y, name, effort, sub, icon, obs, col)
        ys.append((y, y + gh)); y += gh + (40 if i == 0 else 30)
    ybot = y - 24
    # grid behind the bars ($0..$13), drawn under everything else
    grid = ''
    for c in range(0, int(CMAX) + 1):
        grid += f'<line x1="{X(c):.1f}" y1="{top+14}" x2="{X(c):.1f}" y2="{ybot}" stroke="{th["grid"]}" stroke-width="{1.6 if c == 0 else 1}"/>'
        if c % 2 == 0: grid += T(X(c), ybot + 20, f'${c}', 13.5, th['fg2'], 500, 'middle')
    b = grid + b
    # legend
    ly = ybot + 54; lx = 60
    b += glyph(lx + 8, ly - 5, 'pass', th['fg1']) + T(lx + 22, ly, 'pass', 14, th['fg2'], 500)
    b += glyph(lx + 80, ly - 5, 'fail', th['fg1']) + T(lx + 94, ly, 'fail', 14, th['fg2'], 500)
    b += glyph(lx + 146, ly - 5, 'fallback', th['fg1']) + T(lx + 162, ly, 'passed after mid-run Opus fallback*', 14, th['fg2'], 500)
    b += (f'<rect x="{lx+430}" y="{ly-13}" width="26" height="14" rx="3" fill="url(#hatch_opus)" stroke="{OPUS}" stroke-width="1.2"/>'
          + T(lx + 464, ly, 'Opus 4.8 usage within that attempt', 14, th['fg2'], 500))
    b += T(W - 60, ly, 'effort differs: Terminus 2 high · Claude Code xhigh — not a controlled harness comparison', 14, th['fg1'], 600, 'end')
    cy0 = ly + 40
    b += T(60, cy0, 'Every available qualifying attempt shown; this task was selected to illustrate variation. '
           'It does not establish typical variability across the benchmark.', 15, th['fg1'], 600)
    zw = next(r for r in cc if kind(r) == 'fallback')
    foot = [
        'Terminus 2 v2.0.0 · anthropic/claude-fable-5 · effort high · all 5 attempts from TB2.1 leaderboard job #78; none used its Opus fallback. '
        'Claude Code 2.1.167 · effort xhigh · all 5 attempts from job #75.',
        'Effort differs: xhigh is the only Claude Code + Fable 5 run with per-attempt data. Both Daytona; same task revision (sha256:1f1361b0…). '
        'No timeouts, safety stops or infrastructure replacements.',
        f"* {zw['attempt_id']}: Fable refused mid-run; Claude Code retried on Opus 4.8 ({zw['opus_steps']} of {zw['opus_steps'] + 60} steps) and passed. "
        f"Full attempt shown: ${zw['cost_usd']:.2f} Fable + ${zw['opus_cost']:.2f} Opus = ${zw['cost_usd'] + zw['opus_cost']:.2f} (same rates). "
        "The main chart instead counts fallback attempts as failures and excludes their cost.",
        "Cost = token-price estimate at $10/M uncached input (incl. cache writes), $1/M cached, $50/M output, the main chart's basis; "
        "Claude Code's self-reported costs use lower rates and are not used. \"g\" = reconstructed group, as in the main chart.",
    ]
    fy = cy0 + 34
    b += f'<line x1="60" y1="{fy-20}" x2="{W-60}" y2="{fy-20}" stroke="{th["line"]}"/>'
    for i, l in enumerate(foot):
        b += T(60, fy + i * 18, l, 11.5, th['fg2'], 400)
    return b, fy + len(foot) * 18 + 10


if __name__ == '__main__':
    term = base.terminus_attempts(); cc = cc_mod.claude_code_attempts()
    body, h = build(term, cc)
    cc_mod.W, cc_mod.H = W, int(h)
    cc_mod.write(opus_defs() + body, stem='same_task_different_attempts')
    for r in term + cc:
        r['fable_cost_usd'] = r['cost_usd']; r['opus_cost_usd'] = r.get('opus_cost', 0.0)
        r['total_cost_usd'] = r['fable_cost_usd'] + r['opus_cost_usd']
        if r.get('opus_steps'):
            r['note'] = (f"Mixed-model attempt: total ${r['total_cost_usd']:.2f} = Fable ${r['fable_cost_usd']:.2f} + Opus 4.8 "
                         f"${r['opus_cost_usd']:.2f} ({r['opus_steps']} steps, same rates); token columns are the Fable portion only. "
                         + r['note'].split('. ', 1)[1])
    cols = ['harness', 'run', 'attempt_id', 'group', 'start_utc', 'outcome', 'error_type', 'total_cost_usd', 'fable_cost_usd',
            'opus_cost_usd', 'uncached_in',
            'cached_in', 'output_tokens', 'agent_seconds', 'model', 'effort', 'harness_version', 'route', 'environment',
            'task_digest', 'note']
    with open(HERE / 'observations.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for r in term + cc:
            w.writerow({k: (f'{r[k]:.4f}' if k.endswith('cost_usd') else r.get(k, '')) for k in cols})
    print('wrote', HERE / 'observations.csv')
