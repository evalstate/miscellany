"""Round-1 chart candidates (1500x900, 5:3) + shared helpers. HTML+SVG rendered via headless chromium.

`python3 gen.py` re-renders the round-1 variants into archive/charts."""
import json, re, statistics as st, subprocess, pathlib, collections

W, H = 1500, 900
ROOT = pathlib.Path(__file__).resolve().parent
ARCHIVE = ROOT / 'archive' / 'charts'; HTML = ROOT / 'archive' / 'html'
FINAL = {'v14d_cost_hero_light_nochip', 'v15_dual_light_big'}   # publish candidates, rendered into the root folder
T = json.load(open(ROOT / 'trials.json'))
TASKS = sorted({t['task'] for t in T['bare']})

CC_VER, FA_VER = '2.1.278', '0.10.27'
CLAUDE = '#D77757'      # Claude Code terminal orange (rgb 215,119,87)
CLAUDE_HI = '#E8906F'
AMBER = '#f5a400'


def arm(key):
    t = T[key]
    p = sum(1 for x in t if x['reward'] == 1)
    c = sum(x['cost'] for x in t)
    return dict(key=key, trials=t, passes=p, n=len(t), score=p / len(t), cost=c,
                per_pass=c / p, per_trial=c / len(t),
                infra=[x for x in t if x.get('replacement_for')])

A = {
    'oauth': dict(arm('oauth'), name='Claude Code', cfg='default', sub='subscription OAuth', ver=CC_VER, conc=2,
                  color=CLAUDE, kind='cc', solid=True),
    'bare': dict(arm('bare'), name='Claude Code', cfg='Minimal Mode', sub='direct Anthropic API', ver=CC_VER, conc=6,
                 color=CLAUDE, kind='cc', solid=False),
    'fa': dict(arm('fa'), name='fast-agent', cfg='no edit tools', sub='Copilot · generic prompt', ver=FA_VER, conc=4,
               color=AMBER, kind='fa', solid=True),
}
ORDER = ['oauth', 'bare', 'fa']

def boot_ci(key, n=20000):
    import random
    random.seed(0)
    by = collections.defaultdict(list)
    for t in T[key]: by[t['task']].append(1 if t['reward'] == 1 else 0)
    groups = list(by.values()); bs = []
    for _ in range(n):
        s = []
        for _ in groups: s.extend(random.choice(groups))
        bs.append(sum(s) / len(s))
    bs.sort(); return bs[int(n*0.025)], bs[int(n*0.975)]
CI = {k: boot_ci(k) for k in ORDER}

# ---------------------------------------------------------------- themes
THEMES = {
    'dark': dict(bg='#0b0c0f', panel='#12141a', fg='#f0ece2', fg1='#bbb5a8', fg2='#7e796f', line='#2a2926',
                 grid='#1d1d1c', track='#1a1b20', fa_text='#eef4ff', pass_='#f0ece2', fail='#3a3834'),
    'light': dict(bg='#faf9f5', panel='#f0eee6', fg='#1f1e1d', fg1='#4a4843', fg2='#8a867d', line='#dedad0',
                  grid='#ebe8df', track='#ebe8df', fa_text='#111827', pass_='#1f1e1d', fail='#d6d2c7'),
}
MONO = "'JetBrains Mono','JetBrainsMono Nerd Font','IBM Plex Mono',monospace"
SANS = "'Inter','Noto Sans','Adwaita Sans',sans-serif"

# ---------------------------------------------------------------- assets
def clawd(x, y, s, fill, hollow=False, eye='#000', bg=None):
    """Clawd from the Claude Code banner glyphs ( ▐▛███▜▌ / ▝▜█████▛▘ / ▘▘ ▝▝ ).
    Quadrant pixels are s wide x 2s tall (terminal cell aspect). 18 x 5 quadrants."""
    px = set()
    for qx in range(3, 15):
        for qy in range(0, 4):
            px.add((qx, qy))
    px.discard((5, 1)); px.discard((12, 1))            # eyes
    for qx in (1, 2, 15, 16): px.add((qx, 2))          # arms
    for qx in (4, 6, 11, 13): px.add((qx, 4))          # legs
    out = [f'<g transform="translate({x:.1f},{y:.1f})" shape-rendering="crispEdges">']
    if hollow:
        body = px | {(5, 1), (12, 1)}
        edge = {(qx, qy) for (qx, qy) in body
                if any((qx + dx, qy + dy) not in body for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))}
        for (qx, qy) in edge | {(5, 1), (12, 1)}:
            out.append(f'<rect x="{qx*s}" y="{qy*2*s}" width="{s+0.05}" height="{2*s+0.05}" fill="{fill}"/>')
    else:
        for (qx, qy) in px:
            out.append(f'<rect x="{qx*s}" y="{qy*2*s}" width="{s+0.05}" height="{2*s+0.05}" fill="{fill}"/>')
        for (qx, qy) in ((5, 1), (12, 1)):
            out.append(f'<rect x="{qx*s}" y="{qy*2*s}" width="{s}" height="{2*s}" fill="{eye}"/>')
    out.append('</g>')
    return ''.join(out)

CLAWD_W = lambda s: 18 * s
CLAWD_H = lambda s: 10 * s

_lock = (ROOT / 'assets' / 'fast-agent-lockup-dark.svg').read_text()
LOCK_PATHS = re.findall(r'<path d="([^"]+)" fill="([^"]+)"', _lock)

def fa_icon(x, y, size, amber=AMBER, tile='#10151f', tile_on=True):
    sc = size / 256
    t = f'<rect width="256" height="256" rx="32" fill="{tile}"/>' if tile_on else ''
    return (f'<g transform="translate({x},{y}) scale({sc})">{t}'
            f'<path d="M94.340 208L131.300 208L161 128.140L131.300 47.400L94.340 47.400L124.040 127.700L94.340 208Z" fill="{amber}"/></g>')

def fa_lockup(x, y, h, text_fill):
    # lockup content spans x 25..528, y 56..118 in a 540x160 viewBox
    sc = h / 62
    parts = []
    for d, f in LOCK_PATHS:
        parts.append(f'<path d="{d}" fill="{AMBER if f.lower()=="#f5a400" else text_fill}"/>')
    parts.append(f'<rect x="494" y="56" width="34" height="62" fill="{AMBER}"/>')
    return f'<g transform="translate({x - 25*sc},{y - 56*sc}) scale({sc})">{"".join(parts)}</g>'

FA_LOCK_W = lambda h: (528 - 25) * h / 62

def hatch_defs(color, bg, idn='hatch', gap=7, w=3):
    return (f'<pattern id="{idn}" patternUnits="userSpaceOnUse" width="{gap}" height="{gap}" patternTransform="rotate(45)">'
            f'<rect width="{gap}" height="{gap}" fill="{bg}"/><rect width="{w}" height="{gap}" fill="{color}"/></pattern>')

def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def text(x, y, s, size=16, fill='#fff', weight=400, anchor='start', family=MONO, ls=0, extra=''):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" letter-spacing="{ls}" {extra}>{esc(s)}</text>')

def trial_dots(x, y, a, th, r=5.5, gap=15, group_gap=8, infra_mark=True, pass_fill=None, fail_fill=None):
    """21 dots grouped by task (3 each)."""
    out = []; cx = x
    by = collections.defaultdict(list)
    for t in a['trials']: by[t['task']].append(t)
    infra_ids = {t['trial'] for t in a['infra']}
    for ti, task in enumerate(TASKS):
        for t in by[task]:
            ok = t['reward'] == 1
            if t['trial'] in infra_ids and infra_mark:
                out.append(f'<circle cx="{cx}" cy="{y}" r="{r-1}" fill="none" stroke="{th["fg2"]}" stroke-width="1.5" stroke-dasharray="2 2"/>')
            elif ok:
                out.append(f'<circle cx="{cx}" cy="{y}" r="{r}" fill="{pass_fill or a["color"]}"/>')
            else:
                out.append(f'<circle cx="{cx}" cy="{y}" r="{r-1}" fill="none" stroke="{fail_fill or th["fail"]}" stroke-width="1.5"/>')
            cx += gap
        cx += group_gap
    return ''.join(out), cx - x

def footnote_lines(short=False):
    o = A['oauth']
    l1 = ('Terminal-Bench 2.1 · 7-task slice × 3 attempts = 21 trials per configuration · Opus 5 / high · HF Jobs cpu-basic · 0 Harbor retries')
    l2 = ('Tasks: build-pov-ray, dna-assembly, gpt2-codegolf, mteb-leaderboard, mteb-retrieve, raman-fitting, video-processing · '
          f'concurrency: default 2, Minimal Mode 6, fast-agent 4')
    l3 = ('Claude Code default: 18 originals + 3 replacement runs (2 originals blocked by 5-hour OAuth quota, 1 HF exec-stream error); cost includes replacements ($30.31 incl. discarded originals)')
    l4 = ('Cost = token-price estimate from recorded usage; excludes HF compute. OAuth / Copilot figures are not subscription bills. Exploratory, not a leaderboard submission.')
    return [l1, l2, l3, l4] if not short else [l1, l3, l4]

def page(svg_body, th, name, defs=''):
    html = f'''<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap">
<style>html,body{{margin:0;width:{W}px;height:{H}px;overflow:hidden;background:{th["bg"]}}}</style></head>
<body><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>{defs}</defs>
<rect width="{W}" height="{H}" fill="{th["bg"]}"/>{svg_body}</svg></body></html>'''
    HTML.mkdir(parents=True, exist_ok=True); ARCHIVE.mkdir(parents=True, exist_ok=True)
    p = HTML / f'{name}.html'; p.write_text(html)
    png = (ROOT if name in FINAL else ARCHIVE) / f'{name}.png'
    subprocess.run(['chromium', '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--virtual-time-budget=4000', '--force-device-scale-factor=2', f'--screenshot={png}', f'file://{p}'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
    print('wrote', png)

def mascot(a, x, y, s, th):
    if a['kind'] == 'cc':
        return clawd(x, y, s, a['color'], hollow=False, eye=th['bg'], bg=th['bg'])
    sz = CLAWD_H(s) + 8
    return fa_icon(x + (CLAWD_W(s) - sz) / 2, y - 4, sz, tile='#10151f') + (f'<rect x="{x + (CLAWD_W(s) - sz) / 2}" y="{y-4}" width="{sz}" height="{sz}" rx="{sz*32/256}" fill="none" stroke="#2c3444"/>' if th['bg'].startswith('#0') else '')

def cfg_badge(x, y, a, th, size=13, family=MONO):
    label = a['cfg'].upper() if a['kind'] == 'cc' else a['cfg'].upper()
    w = len(label) * size * 0.64 + 18
    col = a['color']
    if a['solid']:
        return (f'<rect x="{x}" y="{y-size-3}" width="{w}" height="{size+10}" rx="3" fill="{col}"/>' +
                text(x + w/2, y+1, label, size, th['bg'], 700, 'middle', family, 0.5)), w
    return (f'<rect x="{x+0.75}" y="{y-size-2.25}" width="{w-1.5}" height="{size+8.5}" rx="3" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="4 2.5"/>' +
            text(x + w/2, y+1, label, size, col, 700, 'middle', family, 0.5)), w

def header(th, title, subtitle, family=MONO, fa_logo=True, y=70):
    s = text(60, y, title, 34, th['fg'], 700, family=family)
    s += text(60, y + 34, subtitle, 17, th['fg1'], 400, family=family)
    if fa_logo:
        s += fa_lockup(W - 60 - FA_LOCK_W(30), y - 30, 30, th['fa_text'])
    return s

def footer(th, y0, lines, size=12.5, family=MONO):
    s = f'<line x1="60" y1="{y0-22}" x2="{W-60}" y2="{y0-22}" stroke="{th["line"]}"/>'
    for i, l in enumerate(lines):
        s += text(60, y0 + i * (size + 7), l, size, th['fg2'], 400, family=family)
    return s

def bar_fill(a, idn):
    return a['color'] if a['solid'] else f'url(#{idn})'

# ======================================================================== V1/V2/V3: dual horizontal bars
def v_dual(theme='dark', fa_neutral=False, name='v01', family=MONO, dots=True, order=ORDER, cc_only=False, big_cfg=False):
    th = THEMES[theme]
    arms = [A[k] for k in order if not (cc_only and A[k]['kind'] == 'fa')]
    if fa_neutral:
        A_fa = dict(A['fa']); A_fa['color'] = th['fg2']
        arms = [A_fa if a['kind'] == 'fa' else a for a in arms]
    defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = header(th, 'Claude Code: default vs Minimal Mode', f'Opus 5 / high · Terminal-Bench 2.1 slice · Claude Code {CC_VER} · fast-agent {FA_VER}', family)
    top, rowh = 215, (170 if not cc_only else 230)
    lx, sx0, sw, cx0, cw = 60, 500, 420, 1075, 270
    # column heads
    b += text(sx0, top - 18, 'SCORE  ·  trials passed', 13, th['fg2'], 600, family=family, ls=1.5)
    b += text(cx0, top - 18, 'COST  ·  21 trials, token-price', 13, th['fg2'], 600, family=family, ls=1.5)
    for i, a in enumerate(arms):
        y = top + i * rowh
        cy = y + rowh / 2 - 8
        if i: b += f'<line x1="60" y1="{y}" x2="{W-60}" y2="{y}" stroke="{th["line"]}"/>'
        # label block
        s = 4.6
        b += mascot(a, lx, cy - CLAWD_H(s) / 2 - 6, s, th)
        tx = lx + 110
        if big_cfg:
            big = a['cfg'] if a['kind'] == 'cc' else 'fast-agent'
            b += text(tx, cy + 4, big, 38, a['color'], 800, family=family)
            small = f"{a['name']} v{a['ver']}" if a['kind'] == 'cc' else f"v{a['ver']} · no edit tools"
            b += text(tx, cy + 32, small, 15, th['fg'], 600, family=family)
            b += text(tx, cy + 54, a['sub'], 13, th['fg2'], 400, family=family)
        else:
            b += text(tx, cy - 8, a['name'], 26, th['fg'], 700, family=family)
            bd, bw = cfg_badge(tx, cy + 24, a, th, 13, family)
            b += bd
            b += text(tx + bw + 10, cy + 24, f"v{a['ver']}", 14, th['fg1'], 500, family=family)
            b += text(tx, cy + 48, a['sub'], 13, th['fg2'], 400, family=family)
        # score bar
        bh = 40
        by = cy - bh / 2 - 4
        b += f'<rect x="{sx0}" y="{by}" width="{sw}" height="{bh}" rx="3" fill="{th["track"]}"/>'
        b += f'<rect x="{sx0}" y="{by}" width="{sw*a["score"]:.1f}" height="{bh}" rx="3" fill="{bar_fill(a,"hatch")}"/>'
        b += text(sx0 + sw + 14, by + 30, f"{a['score']*100:.1f}%", 28, th['fg'], 700, family=family)
        if dots:
            d, dw = trial_dots(sx0 + 6, by + bh + 24, a, th, r=5.5, gap=15, group_gap=10)
            b += d
            b += text(sx0 + dw + 8, by + bh + 29, f"{a['passes']}/{a['n']}", 14, th['fg1'], 600, family=family)
        # cost bar
        cmax = 32
        b += f'<rect x="{cx0}" y="{by}" width="{cw}" height="{bh}" rx="3" fill="{th["track"]}"/>'
        b += f'<rect x="{cx0}" y="{by}" width="{cw*a["cost"]/cmax:.1f}" height="{bh}" rx="3" fill="{bar_fill(a,"hatch")}" opacity="{1 if not fa_neutral or a["kind"]=="cc" else 1}"/>'
        b += text(cx0 + cw*a['cost']/cmax + 12, by + 30, f"${a['cost']:.2f}", 28, th['fg'], 700, family=family)
        b += text(cx0, by + bh + 29, f"${a['per_pass']:.2f} / pass  ·  ${a['per_trial']:.2f} / trial", 13, th['fg1'], 400, family=family)
    # legend for dots
    ly = top + len(arms) * rowh + 22
    b += f'<circle cx="{sx0+6}" cy="{ly-5}" r="5" fill="{th["fg1"]}"/>' + text(sx0 + 18, ly, 'pass', 12, th['fg2'], family=family)
    b += f'<circle cx="{sx0+70}" cy="{ly-5}" r="4" fill="none" stroke="{th["fail"]}" stroke-width="1.5"/>' + text(sx0 + 82, ly, 'fail', 12, th['fg2'], family=family)
    b += f'<circle cx="{sx0+130}" cy="{ly-5}" r="4" fill="none" stroke="{th["fg2"]}" stroke-width="1.5" stroke-dasharray="2 2"/>' + text(sx0 + 142, ly, 'replacement run (fail)', 12, th['fg2'], family=family)
    b += text(sx0 + 380, ly, 'dots grouped by task, 3 attempts each', 12, th['fg2'], family=family)
    b += footer(th, H - 92, footnote_lines(), 12, family)
    page(b, th, name, defs)

# ======================================================================== V4: butterfly
def v_butterfly(theme='dark', name='v04'):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = header(th, 'Score ← → Cost', f'Claude Code {CC_VER} (default · Minimal Mode) and fast-agent {FA_VER} · Opus 5 / high · 21 trials each')
    cxm = W / 2; lblw = 300; top = 200; rowh = 150
    sL = cxm - lblw / 2 - 20; sw = 470; cR = cxm + lblw / 2 + 20; cw = 470; cmax = 32
    b += text(sL, top - 25, 'SCORE', 14, th['fg2'], 700, 'end', ls=2) + text(cR, top - 25, 'COST', 14, th['fg2'], 700, ls=2)
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + i * rowh; cy = y + 55
        bh = 46
        # labels centre
        s = 3.6
        b += mascot(a, cxm - CLAWD_W(s) / 2 if a['kind']=='cc' else cxm - (CLAWD_H(s)+4)/2, cy - 62, s, th)
        b += text(cxm, cy + 12, a['name'], 22, th['fg'], 700, 'middle')
        b += text(cxm, cy + 36, f"{a['cfg']} · v{a['ver']}", 14, a['color'] if a['kind']=='cc' else AMBER, 600, 'middle')
        b += text(cxm, cy + 56, a['sub'], 12, th['fg2'], 400, 'middle')
        # score left
        wv = sw * a['score']
        b += f'<rect x="{sL-sw}" y="{cy-bh/2}" width="{sw}" height="{bh}" rx="3" fill="{th["track"]}"/>'
        b += f'<rect x="{sL-wv:.1f}" y="{cy-bh/2}" width="{wv:.1f}" height="{bh}" rx="3" fill="{bar_fill(a,"hatch")}"/>'
        lab = f"{a['score']*100:.1f}%"
        b += text(sL - wv + 16, cy + 11, lab, 30, th['bg'] if a['solid'] else th['fg'], 800,
                  extra=f'paint-order="stroke" stroke="{th["bg"] if not a["solid"] else "none"}" stroke-width="6"')
        b += text(sL - 14, cy + 9, f"{a['passes']}/{a['n']}", 18, th['bg'] if a['solid'] else th['fg'], 700, 'end',
                  extra=f'paint-order="stroke" stroke="{th["bg"] if not a["solid"] else "none"}" stroke-width="6"')
        # cost right
        wv = cw * a['cost'] / cmax
        b += f'<rect x="{cR}" y="{cy-bh/2}" width="{cw}" height="{bh}" rx="3" fill="{th["track"]}"/>'
        b += f'<rect x="{cR}" y="{cy-bh/2}" width="{wv:.1f}" height="{bh}" rx="3" fill="{bar_fill(a,"hatch")}"/>'
        b += text(cR + wv - 16, cy + 11, f"${a['cost']:.2f}", 30, th['bg'] if a['solid'] else th['fg'], 800, 'end',
                  extra=f'paint-order="stroke" stroke="{th["bg"] if not a["solid"] else "none"}" stroke-width="6"')
        b += text(cR + wv + 14, cy + 9, f"${a['per_pass']:.2f}/pass", 15, th['fg1'], 500)
    b += footer(th, H - 92, footnote_lines(), 12)
    page(b, th, name, defs)

# ======================================================================== V5: big-number cards
def v_cards(theme='dark', name='v05', family=MONO):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = header(th, 'Same model, three harness configurations', f'Opus 5 / high · Terminal-Bench 2.1 · 7 tasks × 3 attempts', family)
    gap = 30; cw = (W - 120 - 2 * gap) / 3; top = 165; chh = 520
    for i, k in enumerate(ORDER):
        a = A[k]; x = 60 + i * (cw + gap)
        stroke = a['color']
        dash = '' if a['solid'] else 'stroke-dasharray="8 5"'
        b += f'<rect x="{x}" y="{top}" width="{cw}" height="{chh}" rx="10" fill="{th["panel"]}" stroke="{stroke}" stroke-width="{2 if a["kind"]=="cc" else 1.2}" {dash}/>'
        s = 5
        b += mascot(a, x + 30, top + 34, s, th)
        b += text(x + 30 + 110, top + 62, a['name'], 26, th['fg'], 700, family=family)
        bd, bw = cfg_badge(x + 140, top + 94, a, th, 13, family); b += bd
        b += text(x + 140 + bw + 10, top + 94, f"v{a['ver']}", 14, th['fg1'], 500, family=family)
        b += text(x + 30, top + 140, a['sub'] + f" · concurrency {a['conc']}", 13, th['fg2'], family=family)
        # score
        b += text(x + 30, top + 205, 'SCORE', 12, th['fg2'], 700, family=family, ls=2)
        b += text(x + 30, top + 265, f"{a['score']*100:.1f}%", 60, th['fg'], 800, family=family)
        b += text(x + cw - 30, top + 265, f"{a['passes']}/{a['n']}", 22, th['fg1'], 600, 'end', family=family)
        bw_ = cw - 60
        b += f'<rect x="{x+30}" y="{top+285}" width="{bw_}" height="10" rx="2" fill="{th["track"]}"/>'
        b += f'<rect x="{x+30}" y="{top+285}" width="{bw_*a["score"]:.1f}" height="10" rx="2" fill="{bar_fill(a,"hatch")}"/>'
        d, dw = trial_dots(x + 36, top + 318, a, th, r=4.5, gap=(bw_ - 12 - 6 * 8) / 20, group_gap=8); b += d
        # cost
        b += text(x + 30, top + 380, 'COST', 12, th['fg2'], 700, family=family, ls=2)
        b += text(x + 30, top + 440, f"${a['cost']:.2f}", 60, a['color'] if a['kind']=='cc' else AMBER, 800, family=family)
        b += f'<rect x="{x+30}" y="{top+460}" width="{bw_}" height="10" rx="2" fill="{th["track"]}"/>'
        b += f'<rect x="{x+30}" y="{top+460}" width="{bw_*a["cost"]/32:.1f}" height="10" rx="2" fill="{bar_fill(a,"hatch")}"/>'
        b += text(x + 30, top + 498, f"${a['per_pass']:.2f} / pass · ${a['per_trial']:.2f} / trial", 14, th['fg1'], family=family)
    b += footer(th, H - 92, footnote_lines(), 12, family)
    page(b, th, name, defs)

# ======================================================================== V6: scatter cost vs score
def v_scatter(theme='dark', name='v06'):
    th = THEMES[theme]
    b = header(th, 'Cost vs score', f'Opus 5 / high · TB2.1 slice · 21 trials each · Claude Code {CC_VER} · fast-agent {FA_VER}')
    x0, x1, y0, y1 = 170, 1080, 715, 175   # plot box (y0 bottom)
    cmin, cmax, smin, smax = 15, 32, 0.45, 1.00
    X = lambda c: x0 + (c - cmin) / (cmax - cmin) * (x1 - x0)
    Y = lambda s: y0 - (s - smin) / (smax - smin) * (y0 - y1)
    for c in range(16, 33, 2):
        b += f'<line x1="{X(c)}" y1="{y1}" x2="{X(c)}" y2="{y0}" stroke="{th["grid"]}"/>'
        if c % 4 == 0: b += text(X(c), y0 + 26, f'${c}', 13, th['fg2'], anchor='middle')
    for sv in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        b += f'<line x1="{x0}" y1="{Y(sv)}" x2="{x1}" y2="{Y(sv)}" stroke="{th["grid"]}"/>'
        b += text(x0 - 14, Y(sv) + 5, f'{sv*100:.0f}%', 13, th['fg2'], anchor='end')
    b += text((x0 + x1) / 2, y0 + 56, 'total token-price cost, 21 trials →', 14, th['fg1'], anchor='middle')
    b += text(x0 - 80, (y0 + y1) / 2, 'score ↑', 14, th['fg1'], anchor='middle', extra=f'transform="rotate(-90 {x0-80} {(y0+y1)/2})"')
    # "better" arrow
    b += text(x0 + 16, y1 + 26, '↖ better: cheaper and higher', 13, th['fg2'])
    for k in ORDER:
        a = A[k]; lo, hi = CI[k]; px = X(a['cost']) + (0 if k != 'oauth' else 26)
        b += f'<line x1="{px}" y1="{Y(lo)}" x2="{px}" y2="{Y(hi)}" stroke="{a["color"]}" stroke-width="2" opacity="0.35"/>'
        b += f'<line x1="{px-7}" y1="{Y(lo)}" x2="{px+7}" y2="{Y(lo)}" stroke="{a["color"]}" stroke-width="2" opacity="0.35"/>'
        b += f'<line x1="{px-7}" y1="{Y(hi)}" x2="{px+7}" y2="{Y(hi)}" stroke="{a["color"]}" stroke-width="2" opacity="0.35"/>'
    b += text(x1 - 10, y0 - 14, 'whiskers: task-clustered bootstrap 95% CI on score', 12, th['fg2'], anchor='end')
    for k in ORDER:
        a = A[k]; px, py = X(a['cost']), Y(a['score'])
        s = 3.4
        if a['kind'] == 'cc':
            b += mascot(a, px - CLAWD_W(s) / 2, py - CLAWD_H(s) / 2, s, th)
        else:
            b += fa_icon(px - 20, py - 20, 40)
        lx = px + 44; ly = py - 4
        if k == 'bare': ly = py + 52; lx = px + 20
        if k == 'fa': ly = py - 44; lx = px - 150
        if k == 'oauth': ly = py + 52; lx = px - 250
        b += text(lx, ly, f"{a['name']} {a['cfg'] if a['kind']=='cc' else ''}".strip(), 19, th['fg'], 700)
        b += text(lx, ly + 22, f"{a['score']*100:.1f}% ({a['passes']}/21) · ${a['cost']:.2f} · v{a['ver']}", 13.5, a['color'], 600)
    # side panel
    sx = 1140
    b += text(sx, 210, 'NOTE', 12, th['fg2'], 700, ls=2)
    notes = ['1 trial = 4.8 pts.', 'Minimal Mode and fast-agent', 'tie at 16/21; default 15/21.', '',
             'Task-clustered bootstrap', '95% CI on score spans', '~40 pts for every arm:', 'score gaps here are noise.', '',
             'Cost spread is real:', 'default starts each call', 'with ~16k prompt tokens', 'vs ~2k for Minimal / fa,', 'and writes 1h-TTL cache.']
    for i, n in enumerate(notes):
        b += text(sx, 240 + i * 22, n, 13.5, th['fg1'])
    b += footer(th, H - 72, footnote_lines(short=True), 12)
    page(b, th, name)

# ======================================================================== V7: per-task matrix
def v_tasks(theme='dark', name='v07'):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = header(th, 'Where the trials land', f'Pass/fail per attempt · 7 TB2.1 tasks × 3 attempts · Opus 5 / high')
    top = 225; lx = 60; colx0 = 400; colw = 124; rowh = 135
    for j, task in enumerate(TASKS):
        cx = colx0 + j * colw + colw / 2
        p1, p2 = task.split('-', 1)
        b += text(cx, top - 38, p1 + '-', 13, th['fg1'], 600, 'middle') + text(cx, top - 20, p2, 13, th['fg1'], 600, 'middle')
    b += text(colx0 + 7 * colw + 40, top - 26, 'TOTAL', 13, th['fg2'], 700, ls=2)
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + i * rowh; cy = y + rowh / 2 - 6
        b += f'<line x1="60" y1="{y}" x2="{W-60}" y2="{y}" stroke="{th["line"]}"/>'
        s = 3.6
        b += mascot(a, lx, cy - CLAWD_H(s) / 2, s, th)
        b += text(lx + 88, cy - 4, a['name'], 21, th['fg'], 700)
        b += text(lx + 88, cy + 20, f"{a['cfg']} · v{a['ver']}", 14, a['color'], 600)
        by = collections.defaultdict(list)
        for t in a['trials']: by[t['task']].append(t)
        infra = {t['trial'] for t in a['infra']}
        for j, task in enumerate(TASKS):
            cx = colx0 + j * colw + colw / 2
            ts = by[task]
            p = sum(1 for t in ts if t['reward'] == 1)
            for m, t in enumerate(ts):
                xx = cx - 30 + m * 30; yy = cy - 10
                if t['trial'] in infra:
                    b += f'<rect x="{xx-11}" y="{yy-11}" width="22" height="22" rx="4" fill="none" stroke="{th["fg2"]}" stroke-dasharray="3 3" stroke-width="1.5"/>'
                elif t['reward'] == 1:
                    b += f'<rect x="{xx-11}" y="{yy-11}" width="22" height="22" rx="4" fill="{bar_fill(a,"hatch")}"/>'
                else:
                    b += f'<rect x="{xx-10}" y="{yy-10}" width="20" height="20" rx="4" fill="none" stroke="{th["fail"]}" stroke-width="2"/>'
            tc = sum(t['cost'] for t in ts)
            b += text(cx, cy + 30, f"${tc:.2f}", 12, th['fg2'], 400, 'middle')
        b += text(colx0 + 7 * colw + 40, cy + 2, f"{a['passes']}/21", 30, th['fg'], 800)
        b += text(colx0 + 7 * colw + 40, cy + 26, f"{a['score']*100:.1f}% · ${a['cost']:.2f}", 13, th['fg1'], 500)
    # highlight columns that differ
    for task in ('mteb-retrieve', 'raman-fitting', 'video-processing', 'dna-assembly'):
        j = TASKS.index(task); x = colx0 + j * colw + 6
        b += f'<rect x="{x}" y="{top-60}" width="{colw-12}" height="{3*rowh+60}" rx="8" fill="none" stroke="{th["line"]}" stroke-width="1"/>'
    ly = top + 3 * rowh + 34
    b += text(60, ly, 'Only build-pov-ray, gpt2-codegolf and mteb-leaderboard are 3/3 everywhere. fast-agent is the only arm to solve mteb-retrieve reliably (3/3 vs 1/3);', 13, th['fg1'])
    b += text(60, ly + 20, 'Claude Code default is weaker on raman-fitting (1/3) but stronger on dna-assembly (2/3). Dashed = replacement run for an infra-hit original (all 3 failed).', 13, th['fg1'])
    b += footer(th, H - 72, footnote_lines(short=True), 12)
    page(b, th, name, defs)

# ======================================================================== V8: per-trial cost strip (variance)
def v_strip(theme='dark', name='v08'):
    th = THEMES[theme]
    b = header(th, 'Per-trial cost spread', f'Each dot is one trial · filled = pass · Opus 5 / high · 21 trials per configuration')
    top = 210; rowh = 150; x0 = 430; x1 = 1360; cmax = 3.6
    X = lambda c: x0 + c / cmax * (x1 - x0)
    for c in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]:
        b += f'<line x1="{X(c)}" y1="{top-10}" x2="{X(c)}" y2="{top+3*rowh-10}" stroke="{th["grid"]}"/>'
        b += text(X(c), top + 3 * rowh + 14, f'${c:.2f}', 13, th['fg2'], anchor='middle')
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + i * rowh; cy = y + rowh / 2 - 20
        s = 3.6
        b += mascot(a, 60, cy - CLAWD_H(s) / 2 - 8, s, th)
        b += text(148, cy - 10, a['name'], 21, th['fg'], 700)
        b += text(148, cy + 14, f"{a['cfg']} · v{a['ver']}", 14, a['color'], 600)
        costs = [t['cost'] for t in a['trials']]
        b += text(148, cy + 36, f"mean ${st.mean(costs):.2f} · sd ${st.stdev(costs):.2f}", 12.5, th['fg2'])
        # box: IQR
        q = st.quantiles(costs, n=4)
        b += f'<rect x="{X(q[0])}" y="{cy-26}" width="{X(q[2])-X(q[0])}" height="52" rx="4" fill="{a["color"]}" opacity="0.12"/>'
        b += f'<line x1="{X(q[1])}" y1="{cy-30}" x2="{X(q[1])}" y2="{cy+30}" stroke="{a["color"]}" stroke-width="2"/>'
        # jitter by task index
        infra = {t['trial'] for t in a['trials'] if t in a['infra']}
        for t in sorted(a['trials'], key=lambda t: t['cost']):
            jj = (TASKS.index(t['task']) - 3) * 6.5
            xx = X(t['cost']); yy = cy + jj
            if t['trial'] in infra:
                b += f'<circle cx="{xx}" cy="{yy}" r="6" fill="none" stroke="{th["fg2"]}" stroke-dasharray="2 2" stroke-width="1.5"/>'
            elif t['reward'] == 1:
                b += f'<circle cx="{xx}" cy="{yy}" r="6.5" fill="{a["color"]}" stroke="{th["bg"]}" stroke-width="1.5"/>'
            else:
                b += f'<circle cx="{xx}" cy="{yy}" r="6" fill="{th["bg"]}" stroke="{a["color"]}" stroke-width="1.8"/>'
        b += text(x1 + 20, cy + 8, f"${a['cost']:.2f}", 22, th['fg'], 800)
    b += text(x0, top + 3 * rowh + 42, 'shaded = interquartile range · line = median · hollow = fail · dashed = replacement run (fail)', 12.5, th['fg2'])
    b += footer(th, H - 72, footnote_lines(short=True), 12)
    page(b, th, name)

# ======================================================================== V9: cost per pass focus
def v_value(theme='dark', name='v09', family=MONO):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = header(th, 'Cost per solved trial', f'Total token-price cost ÷ passed trials · Opus 5 / high · TB2.1 slice', family)
    top = 210; rowh = 140; bx = 560; bw = 700; vmax = 2.4
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + i * rowh; cy = y + 50
        s = 4
        b += mascot(a, 60, cy - CLAWD_H(s) / 2, s, th)
        b += text(160, cy - 6, a['name'], 24, th['fg'], 700, family=family)
        bd, bwd = cfg_badge(160, cy + 26, a, th, 13, family); b += bd
        b += text(160 + bwd + 10, cy + 26, f"v{a['ver']}", 14, th['fg1'], family=family)
        w = bw * a['per_pass'] / vmax
        b += f'<rect x="{bx}" y="{cy-24}" width="{bw}" height="48" rx="3" fill="{th["track"]}"/>'
        b += f'<rect x="{bx}" y="{cy-24}" width="{w:.1f}" height="48" rx="3" fill="{bar_fill(a,"hatch")}"/>'
        b += text(bx + w + 16, cy + 11, f"${a['per_pass']:.2f}", 32, th['fg'], 800, family=family)
        b += text(bx, cy + 50, f"${a['cost']:.2f} ÷ {a['passes']} passes  ·  {a['score']*100:.1f}% score", 14, th['fg1'], family=family)
    b += footer(th, H - 92, footnote_lines(), 12, family)
    page(b, th, name, defs)

# ======================================================================== V10: compact single-panel (score bar + cost as inline bar)
def v_compact(theme='dark', name='v10', family=SANS):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = ''
    b += text(60, 80, 'Claude Code configurations on Terminal-Bench 2.1', 36, th['fg'], 700, family=family)
    b += text(60, 116, f'Opus 5 / high · 7 tasks × 3 attempts · Claude Code {CC_VER} vs fast-agent {FA_VER}', 18, th['fg1'], 400, family=family)
    b += fa_lockup(W - 60 - FA_LOCK_W(28), 52, 28, th['fa_text'])
    top = 180; rowh = 180; x0 = 440; bw = 560
    b += text(x0, top, 'score', 15, th['fg2'], 600, family=family)
    b += text(x0 + bw + 70, top, 'cost', 15, th['fg2'], 600, family=family)
    for i, k in enumerate(ORDER):
        a = A[k]; y = top + 30 + i * rowh
        s = 5.4
        b += mascot(a, 60, y + 10, s, th)
        b += text(190, y + 36, a['name'], 28, th['fg'], 700, family=family)
        b += text(190, y + 66, f"{a['cfg']}", 20, a['color'], 700, family=family)
        b += text(190, y + 92, f"v{a['ver']} · {a['sub']}", 14, th['fg2'], family=family)
        # score bar thick
        b += f'<rect x="{x0}" y="{y+10}" width="{bw}" height="52" rx="6" fill="{th["track"]}"/>'
        b += f'<rect x="{x0}" y="{y+10}" width="{bw*a["score"]:.1f}" height="52" rx="6" fill="{bar_fill(a,"hatch")}"/>'
        b += text(x0 + bw * a['score'] - 16, y + 47, f"{a['score']*100:.0f}%", 30, th['bg'] if a['solid'] else th['fg'], 800, 'end', family,
                  extra=f'paint-order="stroke" stroke="{th["bg"] if not a["solid"] else "none"}" stroke-width="6"')
        # cost bar thin under
        cw = bw * a['cost'] / 32 * 0.9
        b += f'<rect x="{x0}" y="{y+74}" width="{cw:.1f}" height="14" rx="3" fill="{a["color"]}" opacity="0.45"/>'
        b += text(x0 + cw + 10, y + 87, f"${a['cost']:.2f} total", 15, th['fg1'], 600, family=family)
        b += text(x0 + bw + 70, y + 50, f"${a['cost']:.2f}", 40, th['fg'], 800, family=family)
        b += text(x0 + bw + 70, y + 80, f"{a['passes']}/21 passed · ${a['per_pass']:.2f}/pass", 15, th['fg1'], family=family)
    b += footer(th, H - 92, footnote_lines(), 12, family)
    page(b, th, name, defs)

# ======================================================================== V11: grouped bars, config-as-color legend strip on top
def v_grouped(theme='dark', name='v11', family=MONO):
    th = THEMES[theme]; defs = hatch_defs(CLAUDE, th['bg'], 'hatch')
    b = header(th, 'Score and cost by configuration', f'Opus 5 / high · TB2.1 slice · 21 trials each', family)
    # legend chips
    lx = 60; ly = 170
    for k in ORDER:
        a = A[k]; s = 2.8
        b += mascot(a, lx, ly - 10, s, th)
        lab = f"{a['name']} {a['cfg']} v{a['ver']}" if a['kind']=='cc' else f"fast-agent v{a['ver']}"
        b += text(lx + 64, ly + 12, lab, 16, th['fg'], 600, family=family)
        lx += 64 + len(lab) * 10 + 50
    # two metric groups, horizontal bars
    groups = [('SCORE', lambda a: a['score'], 1.0, lambda a: f"{a['score']*100:.1f}%  ({a['passes']}/21)"),
              ('COST', lambda a: a['cost'], 32, lambda a: f"${a['cost']:.2f}  (${a['per_pass']:.2f}/pass)")]
    gy = 250
    for gi, (gname, f, mx, lab) in enumerate(groups):
        y = gy + gi * 270
        b += text(60, y, gname, 14, th['fg2'], 700, family=family, ls=2)
        for i, k in enumerate(ORDER):
            a = A[k]; yy = y + 24 + i * 70; bw = 1000
            w = bw * f(a) / mx
            bx = 330; bw = 820; w = bw * f(a) / mx
            b += mascot(a, 60, yy + 12, 2.6, th)
            tag = f"CC {a['cfg']}" if a['kind'] == 'cc' else 'fast-agent'
            b += text(120, yy + 32, tag, 18, a['color'], 700, family=family)
            b += f'<rect x="{bx}" y="{yy}" width="{bw}" height="50" rx="3" fill="{th["track"]}"/>'
            b += f'<rect x="{bx}" y="{yy}" width="{w:.1f}" height="50" rx="3" fill="{bar_fill(a,"hatch")}"/>'
            b += text(bx + w + 16, yy + 34, lab(a), 22, th['fg'], 700, family=family)
    b += footer(th, H - 72, footnote_lines(short=True), 12, family)
    page(b, th, name, defs)


if __name__ == '__main__':
    v_dual('dark', name='v01_dual_dark')
    v_dual('light', name='v02_dual_light')
    v_dual('dark', fa_neutral=True, name='v03_dual_dark_fa_grey')
    v_dual('dark', name='v03b_dual_dark_cc_only', cc_only=True)
    v_dual('light', name='v02b_dual_light_sans', family=SANS)
    v_butterfly('dark', 'v04_butterfly_dark')
    v_butterfly('light', 'v04b_butterfly_light')
    v_cards('dark', 'v05_cards_dark')
    v_cards('light', 'v05b_cards_light', family=SANS)
    v_scatter('dark', 'v06_scatter_dark')
    v_tasks('dark', 'v07_tasks_dark')
    v_tasks('light', 'v07b_tasks_light')
    v_strip('dark', 'v08_cost_spread_dark')
    v_value('dark', 'v09_cost_per_pass_dark')
    v_compact('light', 'v10_compact_light_sans')
    v_compact('dark', 'v10b_compact_dark_sans')
    v_grouped('dark', 'v11_grouped_dark')
    v_grouped('light', 'v11b_grouped_light')
    v_dual('dark', name='v12_dual_dark_bigcfg', big_cfg=True)
    v_dual('light', name='v12b_dual_light_bigcfg', big_cfg=True)
    v_dual('dark', name='v12c_dual_dark_bigcfg_cc_only', big_cfg=True, cc_only=True)
    v_scatter('light', 'v06b_scatter_light')
    v_strip('light', 'v08b_cost_spread_light')
