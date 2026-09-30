"""Round 2/3 charts. `python3 gen2.py` renders the publish candidates (v14d_cost_hero_light_nochip.png, v15_dual_light_big.png) into the root;
`python3 gen2.py --all` also re-renders the round-2/3 alternates into archive/charts."""
from gen import *  # noqa

def label_block(a, th, x, cy, family, s=5.0, name_size=28, cfg_size=22):
    b = mascot(a, x, cy - CLAWD_H(s) / 2 - 10, s, th)
    tx = x + CLAWD_W(s) + 26
    b += text(tx, cy - 14, a['name'], name_size, th['fg'], 700, family=family)
    cfg = a['cfg'] if a['kind'] == 'cc' else 'no edit tools'
    b += text(tx, cy + 16, cfg, cfg_size, a['color'], 700, family=family)
    sub = a['sub'] if a['kind'] == 'cc' else 'Copilot route'
    b += text(tx, cy + 42, f"v{a['ver']} · {sub}", 15, th['fg2'], 400, family=family)
    return b

def title_block(th, title, sub, family):
    b = text(60, 82, title, 38, th['fg'], 700, family=family)
    b += text(60, 120, sub, 19, th['fg1'], 400, family=family)
    b += fa_lockup(W - 60 - FA_LOCK_W(28), 54, 28, th['fa_text'])
    return b

def ink(a, th):
    """text colour for a label drawn on top of the bar."""
    return th['bg'] if a['solid'] else th['fg']

def halo(a, th, w=7):
    return f'paint-order="stroke" stroke="{th["bg"]}" stroke-width="{w}" stroke-linejoin="round"' if not a['solid'] else ''

# ---------------------------------------------------------------- dual bars, large labels
def v_dual_big(theme='light', name='v02c', family=SANS, val=44):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = title_block(th, 'Claude Code: default vs Minimal Mode',
                    f'Opus 5 / high · Terminal-Bench 2.1 slice · Claude Code {CC_VER} · fast-agent {FA_VER}', family)
    top, rowh = 225, 172
    sx0, sw, cx0, cw, cmax = 470, 320, 1025, 250, 32
    b += text(sx0, top - 20, 'Score', 18, th['fg2'], 600, family=family)
    b += text(sx0 + 58, top - 20, '· trials passed', 16, th['fg2'], 400, family=family)
    b += text(cx0, top - 20, 'Cost', 18, th['fg2'], 600, family=family)
    b += text(cx0 + 48, top - 20, '· 21 trials, token-price', 16, th['fg2'], 400, family=family)
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + i * rowh; cy = y + rowh / 2 - 6
        if i: b += f'<line x1="60" y1="{y}" x2="{W-60}" y2="{y}" stroke="{th["line"]}"/>'
        b += label_block(a, th, 60, cy, family)
        bh = 46; by = cy - bh / 2 - 12
        base = by + bh / 2 + val * 0.36
        b += f'<rect x="{sx0}" y="{by}" width="{sw}" height="{bh}" rx="5" fill="{th["track"]}"/>'
        b += f'<rect x="{sx0}" y="{by}" width="{sw*a["score"]:.1f}" height="{bh}" rx="5" fill="{bar_fill(a,"hatch")}"/>'
        b += text(sx0 + sw + 18, base, f"{a['score']*100:.1f}%", val, th['fg'], 800, family=family)
        d, dw = trial_dots(sx0 + 6, by + bh + 26, a, th, r=6, gap=14.2, group_gap=6)
        b += d
        b += text(sx0 + dw + 10, by + bh + 32, f"{a['passes']}/{a['n']}", 17, th['fg1'], 700, family=family)
        w = cw * a['cost'] / cmax
        b += f'<rect x="{cx0}" y="{by}" width="{w:.1f}" height="{bh}" rx="5" fill="{bar_fill(a,"hatch")}"/>'
        b += text(cx0 + w + 16, base, f"${a['cost']:.2f}", val, th['fg'], 800, family=family,
                  extra=f'paint-order="stroke" stroke="{th["bg"]}" stroke-width="6" stroke-linejoin="round"')
        b += text(cx0, by + bh + 32, f"${a['per_pass']:.2f} / pass  ·  ${a['per_trial']:.2f} / trial", 17, th['fg1'], 500, family=family)
    ly = top + 3 * rowh + 20
    b += f'<circle cx="{sx0+6}" cy="{ly-5}" r="6" fill="{th["fg1"]}"/>' + text(sx0 + 20, ly, 'pass', 14, th['fg2'], family=family)
    b += f'<circle cx="{sx0+76}" cy="{ly-5}" r="5" fill="none" stroke="{th["fail"]}" stroke-width="1.8"/>' + text(sx0 + 90, ly, 'fail', 14, th['fg2'], family=family)
    b += f'<circle cx="{sx0+138}" cy="{ly-5}" r="5" fill="none" stroke="{th["fg2"]}" stroke-width="1.8" stroke-dasharray="2 2"/>' + text(sx0 + 152, ly, 'replacement run (fail)', 14, th['fg2'], family=family)
    b += text(sx0 + 330, ly, '· dots grouped by task, 3 attempts each', 14, th['fg2'], family=family)
    b += footer(th, H - 88, footnote_lines(), 12.5, family)
    page(b, th, name, defs)

# ---------------------------------------------------------------- compact, cost as the hero
def v_cost_hero(theme='light', name='v13', family=SANS, deltas=True, ref_line=False,
                title='What each Claude Code configuration costs', score_bars=True):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = title_block(th, title, f'Opus 5 / high · Terminal-Bench 2.1 · 7 tasks × 3 attempts · Claude Code {CC_VER} vs fast-agent {FA_VER}', family)
    top, rowh = 215, 178
    bx, bw, cmax = 430, 540, 30
    nx = bx + bw + 26           # big cost number
    scx = 1250                  # score column
    b += text(bx, top - 14, 'Cost', 18, th['fg2'], 600, family=family)
    b += text(bx + 48, top - 14, '· 21 trials, token-price', 16, th['fg2'], 400, family=family)
    b += text(scx, top - 14, 'Score', 18, th['fg2'], 600, family=family)
    if ref_line:
        base = A['fa']['cost']; xr = bx + bw * base / cmax
        b += f'<line x1="{xr}" y1="{top+4}" x2="{xr}" y2="{top+3*rowh-24}" stroke="{th["fg2"]}" stroke-width="1.5" stroke-dasharray="5 5"/>'
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + 18 + i * rowh; cy = y + 56
        if i: b += f'<line x1="60" y1="{y-18}" x2="{W-60}" y2="{y-18}" stroke="{th["line"]}"/>'
        b += label_block(a, th, 60, cy, family, s=5.4, name_size=28, cfg_size=22)
        bh = 64; by = cy - bh / 2 - 8
        w = bw * a['cost'] / cmax
        b += f'<rect x="{bx}" y="{by}" width="{w:.1f}" height="{bh}" rx="7" fill="{bar_fill(a,"hatch")}"/>'
        b += text(nx, by + bh / 2 + 20, f"${a['cost']:.2f}", 56, th['fg'], 800, family=family)
        sub = f"${a['per_pass']:.2f} / pass  ·  ${a['per_trial']:.2f} / trial"
        b += text(bx, by + bh + 30, sub, 17, th['fg1'], 500, family=family)
        if deltas and k == 'oauth':
            dm = a['cost'] - A['bare']['cost']; df = a['cost'] - A['fa']['cost']
            chip = f"+{dm/A['bare']['cost']*100:.0f}% vs Minimal Mode  ·  +{df/A['fa']['cost']*100:.0f}% vs fast-agent"
            cwid = len(chip) * 8.6 + 28
            cxp = bx + 262
            b += f'<rect x="{cxp}" y="{by+bh+10}" width="{cwid}" height="30" rx="15" fill="{CLAUDE}" opacity="0.14"/>'
            b += text(cxp + cwid / 2, by + bh + 31, chip, 15.5, CLAUDE if theme == 'dark' else '#b5532f', 700, 'middle', family)
        # score column
        b += text(scx, by + 40, f"{a['score']*100:.1f}%", 36, th['fg'], 800, family=family)
        if score_bars:
            sw = 190
            b += f'<rect x="{scx}" y="{by+52}" width="{sw}" height="8" rx="4" fill="{th["track"]}"/>'
            b += f'<rect x="{scx}" y="{by+52}" width="{sw*a["score"]:.1f}" height="8" rx="4" fill="{bar_fill(a,"hatch")}"/>'
        b += text(scx, by + bh + 30, f"{a['passes']}/21 passed", 17, th['fg1'], 600, family=family)
    if ref_line:
        xr = bx + bw * A['fa']['cost'] / cmax
        b += text(xr + 8, top + 3 * rowh - 10, f"fast-agent ${A['fa']['cost']:.2f}", 13, th['fg2'], 500, family=family)
    b += footer(th, H - 88, footnote_lines(), 12.5, family)
    page(b, th, name, defs)



# ================================================================ round 3
TITLE3 = 'Claude Code Minimal Mode comparison'

def grid7x3(x, y, a, th, cell=14, gap=3.5):
    """7 task columns x 3 attempt rows; passes stacked from the bottom like mini bars."""
    by = collections.defaultdict(list)
    for t in a['trials']: by[t['task']].append(t)
    rep = {t['trial'] for t in a['infra']}
    out = []
    for j, task in enumerate(TASKS):
        ts = sorted(by[task], key=lambda t: (t['reward'] == 1, t['trial'] not in rep))  # fails/replacements on top
        for m, t in enumerate(ts):
            cx = x + j * (cell + gap); cy = y + m * (cell + gap)
            if t['reward'] == 1:
                out.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="2.5" fill="{a["color"]}"/>')
            elif t['trial'] in rep:
                out.append(f'<rect x="{cx+0.9}" y="{cy+0.9}" width="{cell-1.8}" height="{cell-1.8}" rx="2" fill="none" stroke="{th["fg2"]}" stroke-width="1.6" stroke-dasharray="2.4 2"/>')
            else:
                out.append(f'<rect x="{cx+0.9}" y="{cy+0.9}" width="{cell-1.8}" height="{cell-1.8}" rx="2" fill="none" stroke="{"#bdb7a9" if th["bg"].startswith("#f") else "#57534c"}" stroke-width="1.6"/>')
    return ''.join(out), 7 * (cell + gap) - gap, 3 * (cell + gap) - gap

def label_block3(a, th, x, cy, family, s=5.4):
    b = mascot(a, x, cy - CLAWD_H(s) / 2 - 10, s, th)
    tx = x + CLAWD_W(s) + 26
    b += text(tx, cy - 14, a['name'], 28, th['fg'], 700, family=family)
    cfg = a['cfg'] if a['kind'] == 'cc' else 'no edit tools'
    b += text(tx, cy + 16, cfg, 22, a['color'], 700, family=family)
    sub = f"v{a['ver']} · {a['sub']}" if a['kind'] == 'cc' else f"v{a['ver']}"
    b += text(tx, cy + 42, sub, 15, th['fg2'], 400, family=family)
    return b

def col_head(x, y, main, sub, th, family, anchor='start', size=26):
    s = text(x, y, main, size, th['fg2'], 700, anchor, family)
    if sub:
        if anchor == 'start':
            s += text(x + len(main) * size * 0.6 + 10, y, sub, 16, th['fg2'], 400, family=family)
        else:
            s += text(x, y + 22, sub, 15, th['fg2'], 400, anchor, family)
    return s

def grid_legend(x, y, th, family, a_demo):
    s = f'<rect x="{x}" y="{y-12}" width="13" height="13" rx="2.5" fill="{th["fg1"]}"/>' + text(x + 20, y, 'pass', 14, th['fg2'], family=family)
    s += f'<rect x="{x+72}" y="{y-11}" width="11" height="11" rx="2" fill="none" stroke="{th["fg2"]}" stroke-width="1.6"/>' + text(x + 92, y, 'fail', 14, th['fg2'], family=family)
    s += f'<rect x="{x+136}" y="{y-11}" width="11" height="11" rx="2" fill="none" stroke="{th["fg2"]}" stroke-width="1.6" stroke-dasharray="2.4 2"/>' + text(x + 156, y, 'replacement run (fail)', 14, th['fg2'], family=family)
    s += text(x + 330, y, '· grid = 7 tasks (columns) × 3 attempts, passes stacked from the bottom', 14, th['fg2'], family=family)
    return s


# ---------------------------------------------------------------- publish-round disclosures (v14d / v15)
SUB3 = f'Opus 5 / high · Terminal-Bench 2.1 · Claude Code {CC_VER} vs fast-agent {FA_VER}'
EXPLORATORY = 'Exploratory: 7 tasks × 3 attempts per configuration'
CONFOUND = 'Mode, authentication route and concurrency differ; this is not an isolated estimate of the mode\'s effect.'


def exploratory_pill(th, family, y=146):
    w = len(EXPLORATORY) * 17 * 0.56 + 32
    return (f'<rect x="60" y="{y}" width="{w:.0f}" height="34" rx="17" fill="{CLAUDE}" opacity="0.13"/>'
            f'<rect x="60.75" y="{y+0.75}" width="{w-1.5:.0f}" height="32.5" rx="16.25" fill="none" stroke="{CLAUDE}" stroke-width="1.5"/>' +
            text(60 + w / 2, y + 23, EXPLORATORY, 17, '#9a4a2c' if th['bg'].startswith('#f') else CLAUDE_HI, 700, 'middle', family))


def publish_footer(th, family, size=12):
    lines = [
        CONFOUND,
        '21 trials per configuration · HF Jobs cpu-basic · 0 Harbor retries · routes: default = subscription OAuth, '
        'Minimal Mode = direct Anthropic API, fast-agent = Copilot · concurrency: default 2, Minimal Mode 6, fast-agent 4',
        'Tasks: build-pov-ray, dna-assembly, gpt2-codegolf, mteb-leaderboard, mteb-retrieve, raman-fitting, video-processing',
        'Claude Code default: 18 originals + 3 replacement runs (2 originals blocked by 5-hour OAuth quota, 1 HF exec-stream error); '
        'cost includes replacements ($30.31 incl. discarded originals)',
        'Cost = token-price estimate from recorded usage; excludes HF compute. OAuth / Copilot figures are not subscription bills. Not a leaderboard submission.',
    ]
    y0 = H - 104
    out = f'<line x1="60" y1="{y0-22}" x2="{W-60}" y2="{y0-22}" stroke="{th["line"]}"/>'
    for i, l in enumerate(lines):
        out += text(60, y0 + i * (size + 7), l, size + (0.5 if i == 0 else 0), th['fg1'] if i == 0 else th['fg2'],
                    600 if i == 0 else 400, family=family)
    return out


def v_cost_hero3(theme='light', name='v14', family=SANS, deltas=True, ref_line=False, title=TITLE3):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = title_block(th, title, SUB3, family) + exploratory_pill(th, family)
    top, rowh = 222, 168
    bx, bw, cmax = 430, 460, 30
    nr = 1120                   # right edge of cost numbers
    scx = 1162                  # score column
    gx = W - 60 - (7 * 19 - 3.5)
    b += col_head(bx, top - 14, 'Cost', '· 21 trials, token-price', th, family)
    b += col_head(scx, top - 14, 'Pass rate', '', th, family)
    b += text(gx + (7 * 19 - 3.5) / 2, top - 14, 'by task', 15, th['fg2'], 500, 'middle', family)
    if ref_line:
        xr = bx + bw * A['fa']['cost'] / cmax
        b += f'<line x1="{xr}" y1="{top+4}" x2="{xr}" y2="{top+3*rowh-20}" stroke="{th["fg2"]}" stroke-width="1.5" stroke-dasharray="5 5"/>'
        b += text(xr + 8, top + 3 * rowh - 8, f"fast-agent ${A['fa']['cost']:.2f}", 13, th['fg2'], 500, family=family)
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + 18 + i * rowh; cy = y + 54
        if i: b += f'<line x1="60" y1="{y-16}" x2="{W-60}" y2="{y-16}" stroke="{th["line"]}"/>'
        b += label_block3(a, th, 60, cy, family)
        bh = 62; by = cy - bh / 2 - 8
        w = bw * a['cost'] / cmax
        b += f'<rect x="{bx}" y="{by}" width="{w:.1f}" height="{bh}" rx="7" fill="{bar_fill(a,"hatch")}"/>'
        b += text(nr, by + bh / 2 + 20, f"${a['cost']:.2f}", 56, th['fg'], 800, 'end', family)
        b += text(bx, by + bh + 30, f"${a['per_pass']:.2f} / pass  ·  ${a['per_trial']:.2f} / trial", 17, th['fg1'], 500, family=family)
        if deltas and k == 'oauth':
            chip = (f"+{(a['cost']/A['bare']['cost']-1)*100:.0f}% vs Minimal Mode  ·  "
                    f"+{(a['cost']/A['fa']['cost']-1)*100:.0f}% vs fast-agent")
            cwid = len(chip) * 8.6 + 28; cxp = bx + 250
            b += f'<rect x="{cxp}" y="{by+bh+10}" width="{cwid}" height="30" rx="15" fill="{CLAUDE}" opacity="0.14"/>'
            b += text(cxp + cwid / 2, by + bh + 31, chip, 15.5, CLAUDE if theme == 'dark' else '#b5532f', 700, 'middle', family)
        # score + grid
        b += text(scx, by + 44, f"{a['score']*100:.1f}%", 38, th['fg'], 800, family=family)
        b += text(scx, by + bh + 30, f"{a['passes']}/21 passed", 17, th['fg1'], 600, family=family)
        g, gw, gh = grid7x3(gx, by + 2, a, th, cell=15.5, gap=3.5)
        b += g
    b += grid_legend(bx, top + 3 * rowh + 22, th, family, A['oauth'])
    b += publish_footer(th, family)
    page(b, th, name, defs)

def v_dual_big3(theme='light', name='v15', family=SANS, val=44):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = title_block(th, TITLE3, SUB3, family) + exploratory_pill(th, family)
    top, rowh = 228, 170
    sx0, sw = 470, 300
    gx = sx0
    cx0, cw, cmax, nr = 1000, 230, 32, W - 60
    b += col_head(sx0, top - 16, 'Pass rate', '· trials passed', th, family)
    b += col_head(cx0, top - 16, 'Cost', '· 21 trials', th, family)
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + i * rowh; cy = y + rowh / 2 - 4
        if i: b += f'<line x1="60" y1="{y}" x2="{W-60}" y2="{y}" stroke="{th["line"]}"/>'
        b += label_block3(a, th, 60, cy, family)
        bh = 46; by = cy - bh / 2 - 14
        base = by + bh / 2 + val * 0.36
        b += f'<rect x="{sx0}" y="{by}" width="{sw}" height="{bh}" rx="5" fill="{th["track"]}"/>'
        b += f'<rect x="{sx0}" y="{by}" width="{sw*a["score"]:.1f}" height="{bh}" rx="5" fill="{bar_fill(a,"hatch")}"/>'
        b += text(sx0 + sw + 18, base, f"{a['score']*100:.1f}%", val, th['fg'], 800, family=family)
        g, gw, gh = grid7x3(sx0, by + bh + 12, a, th, cell=13.5, gap=3)
        b += g
        b += text(sx0 + gw + 16, by + bh + 12 + gh / 2 + 7, f"{a['passes']}/{a['n']} passed", 18, th['fg1'], 700, family=family)
        w = cw * a['cost'] / cmax
        b += f'<rect x="{cx0}" y="{by}" width="{w:.1f}" height="{bh}" rx="5" fill="{bar_fill(a,"hatch")}"/>'
        b += text(nr, base, f"${a['cost']:.2f}", val, th['fg'], 800, 'end', family)
        b += text(cx0, by + bh + 32, f"${a['per_pass']:.2f} / pass  ·  ${a['per_trial']:.2f} / trial", 17, th['fg1'], 500, family=family)
    b += grid_legend(sx0, top + 3 * rowh + 18, th, family, A['oauth'])
    b += publish_footer(th, family)
    page(b, th, name, defs)


if __name__ == '__main__':
    import sys
    v_cost_hero3('light', 'v14d_cost_hero_light_nochip', deltas=False)   # publish candidates -> root folder
    v_dual_big3('light', 'v15_dual_light_big')
    if '--all' in sys.argv:                                 # alternates -> archive/charts
        v_dual_big('light', 'v02c_dual_light_sans_big')
        v_dual_big('dark', 'v02d_dual_dark_sans_big')
        v_cost_hero('light', 'v13_cost_hero_light')
        v_cost_hero('dark', 'v13b_cost_hero_dark')
        v_cost_hero('light', 'v13c_cost_hero_light_refline', ref_line=True)
        v_cost_hero('light', 'v13d_cost_hero_light_headline', title='Default Claude Code costs ~46% more than Minimal Mode')
        v_cost_hero('light', 'v13e_cost_hero_light_plain', deltas=False, score_bars=False)
        v_cost_hero3('dark', 'v14b_cost_hero_dark')
        v_cost_hero3('light', 'v14c_cost_hero_light_refline', ref_line=True)
        v_cost_hero3('light', 'v14_cost_hero_light')        # with the +46% / +68% badge
        v_dual_big3('dark', 'v15b_dual_dark_big')
