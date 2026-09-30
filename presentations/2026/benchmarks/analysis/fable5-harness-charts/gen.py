"""Fable 5 harness comparison charts (1500x900 @2x, 5:3): Terminus 2 (leaderboard), Strands / OpenCode /
Oh-my-pi (as published in the Strands chart) vs our fast-agent medium / high runs.

    python3 gen.py            # render the final chart -> fable5_harness_comparison.png (root)
    python3 gen.py --all      # also re-render every candidate -> archive/candidates/ + archive/contact_sheet.jpg
"""
import json, math, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'archive' / 'candidates'; HTML = ROOT / 'archive' / 'html'
FINAL = {'fable5_harness_comparison'}   # rendered into the root folder
W, H = 1500, 900
AMBER = '#f5a400'; AMBER_TXT = '#b77a00'
SANS = "'Inter','Noto Sans','Adwaita Sans',sans-serif"
MONO = "'JetBrains Mono','JetBrainsMono Nerd Font',monospace"

THEMES = {
    'light': dict(bg='#faf9f5', fg='#1f1e1d', fg1='#4a4843', fg2='#8a867d', line='#dedad0', grid='#ebe8df',
                  track='#ebe8df', fa_text='#111827', lb='#4d5059', pub='#aaa597', pub_line='#8f8a7d', tile='#10151f'),
    'dark': dict(bg='#0b0c0f', fg='#f0ece2', fg1='#bbb5a8', fg2='#7e796f', line='#2a2926', grid='#1d1d1c',
                 track='#1a1b20', fa_text='#eef4ff', lb='#c9c4b8', pub='#6d6a63', pub_line='#8a867d', tile='#10151f'),
}

D = {r['key']: r for r in json.load(open(ROOT / 'data.json'))}
FA_VER = D['fa_high']['version']
TA, TB = D['terminus_avg'], D['terminus_best']




# ------------------------------------------------------------------ per-entry presentation
def meta(k, th):
    r = D[k]
    if r['source'] == 'ours':
        cherry = bool(r.get('other_runs'))
        return dict(r=r, title='fast-agent', line2=f"Fable 5 · {r['effort']}", line3=f"v{r['version']} · {r['passes']}/89 passed",
                    color=AMBER, txt=AMBER if th['bg'].startswith('#0') else AMBER_TXT, solid=r['effort'] == 'medium',
                    hatch='hatch_amber', tag=f"CHERRY-PICKED FROM {1 + len(r['other_runs'])}" if cherry else 'OUR RUN',
                    cherry=cherry, mono=None)
    if r['source'] == 'leaderboard':
        if r['variant'] == 'avg':
            l2, l3 = f"Fable 5 · {r['effort']} · 5-run average", f"Fable-only · {r['passes']}/{r['n']} trials"
        else:
            grp = {'best': 'highest-scoring group', 'worst': 'lowest-scoring group'}[r['variant']]
            l2, l3 = f"Fable 5 · {r['effort']} · {grp}", f"fallback-adjusted · {r['passes']}/89 · group {r['run']} of 5"
        return dict(r=r, title=r['name'], line2=l2, line3=l3, color=th['lb'], txt=th['fg1'],
                    solid=r['variant'] != 'avg', hatch='hatch_lb', tag='LEADERBOARD', mono='T2')
    mono = {'strands': 'S', 'opencode': 'OC', 'ohmypi': 'π'}[k]
    return dict(r=r, title=r['name'], line2='Fable 5', line3='as published by Strands',
                color=th['pub'], txt=th['fg1'], solid=True, hatch='hatch_pub', tag='STRANDS CHART', mono=mono)


BASE = ['fa_medium', 'fa_high', 'strands', 'opencode', 'ohmypi']
TERMS = {'both': ['terminus_avg', 'terminus_best'], 'avg': ['terminus_avg'], 'best': ['terminus_best'],
         'range': ['terminus_best', 'terminus_avg', 'terminus_worst'], 'bestworst': ['terminus_best', 'terminus_worst']}


# ------------------------------------------------------------------ svg helpers
def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, size=16, fill='#000', weight=400, anchor='start', family=SANS, ls=0, extra=''):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}" {extra}>{esc(s)}</text>')


_lock = (ROOT / 'assets' / 'fast-agent-lockup-dark.svg').read_text()
LOCK = re.findall(r'<path d="([^"]+)" fill="([^"]+)"', _lock)


def fa_lockup(x, y, h, fill):
    sc = h / 62
    p = ''.join(f'<path d="{d}" fill="{AMBER if f.lower() == "#f5a400" else fill}"/>' for d, f in LOCK)
    return (f'<g transform="translate({x - 25 * sc},{y - 56 * sc}) scale({sc})">{p}'
            f'<rect x="494" y="56" width="34" height="62" fill="{AMBER}"/></g>')


LOCK_W = lambda h: (528 - 25) * h / 62


def fa_icon(x, y, size, th):
    sc = size / 256
    ring = (f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{size*32/256}" fill="none" stroke="#2c3444"/>'
            if th['bg'].startswith('#0') else '')
    return (f'<g transform="translate({x},{y}) scale({sc})"><rect width="256" height="256" rx="32" fill="{th["tile"]}"/>'
            f'<path d="M94.340 208L131.300 208L161 128.140L131.300 47.400L94.340 47.400L124.040 127.700L94.340 208Z" fill="{AMBER}"/></g>' + ring)


def mono_tile(x, y, size, label, m, th):
    dash = '' if m['r']['source'] == 'leaderboard' else 'stroke-dasharray="4 3"'
    return (f'<rect x="{x+1}" y="{y+1}" width="{size-2}" height="{size-2}" rx="{size*0.14}" fill="none" '
            f'stroke="{m["color"]}" stroke-width="2" {dash}/>' +
            text(x + size / 2, y + size / 2 + size * 0.15, label, size * 0.40 if len(label) > 1 else size * 0.46,
                 m['color'] if m['r']['source'] == 'leaderboard' else th['fg2'], 800, 'middle'))


import base64
TP = ROOT / 'assets' / 'third-party'
LOGOS = {  # key -> (file, tile colour or None, inset fraction)
    'strands': ('strands-logo-light.svg', '#0e0e0e', 0.17),
    'opencode': ('opencode-favicon.svg', None, 0),
    'ohmypi': ('omp-favicon.svg', None, 0),
    'terminus_avg': ('terminal-bench-fav.png', None, 0),
    'terminus_best': ('terminal-bench-fav.png', None, 0),
    'terminus_worst': ('terminal-bench-fav.png', None, 0),
}


def data_uri(fn):
    p = TP / fn
    mime = 'image/svg+xml' if p.suffix == '.svg' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode()


def logo(k, x, y, size, th):
    fn, tile, inset = LOGOS[k]
    rx = size * 32 / 256
    clip = f'clip_{k}_{int(x)}_{int(y)}'
    out = f'<clipPath id="{clip}"><rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{rx}"/></clipPath>'
    if tile:
        out += f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{rx}" fill="{tile}"/>'
    pad = size * inset
    out += (f'<image href="{data_uri(fn)}" x="{x+pad}" y="{y+pad}" width="{size-2*pad}" height="{size-2*pad}" '
            f'preserveAspectRatio="xMidYMid meet" clip-path="url(#{clip})"/>')
    if th['bg'].startswith('#0'):
        out += f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{rx}" fill="none" stroke="#2c3444"/>'
    return out


def icon(k, x, y, size, th):
    m = meta(k, th)
    if m['mono'] is None: return fa_icon(x, y, size, th)
    return logo(k, x, y, size, th) if k in LOGOS else mono_tile(x, y, size, m['mono'], m, th)


def defs(th):
    out = ''
    for idn, col in (('hatch_amber', AMBER), ('hatch_lb', th['lb']), ('hatch_pub', th['pub'])):
        out += (f'<pattern id="{idn}" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">'
                f'<rect width="7" height="7" fill="{th["bg"]}"/><rect width="3" height="7" fill="{col}"/></pattern>')
    return out


def fill(m):
    return m['color'] if m['solid'] else f"url(#{m['hatch']})"


CHERRY_SLOT = 40   # px reserved left of the cherry-pick badge


def cherry_icon(x, y, s):
    retro = ROOT / 'assets' / 'retro-cherries.png'
    if retro.exists():
        uri = 'data:image/png;base64,' + __import__('base64').b64encode(retro.read_bytes()).decode()
        # Oversized, tilted sticker; the viewport removes transparent asset padding.
        side = s * 4.86
        left, top = x - 46, y - 21
        return (f'<g id="cherry-sticker" transform="rotate(-16 {left + side / 2} {top + side / 2})">'
                f'<svg x="{left}" y="{top}" width="{side}" height="{side}" viewBox="215 180 860 895">'
                f'<image href="{uri}" width="1254" height="1254"/></svg></g>')
    for fn in ('cherry.svg', 'cherry.png'):
        p = ROOT / 'assets' / fn
        if p.exists():
            mime = 'image/svg+xml' if p.suffix == '.svg' else 'image/png'
            uri = f'data:{mime};base64,' + __import__('base64').b64encode(p.read_bytes()).decode()
            return f'<image href="{uri}" x="{x}" y="{y}" width="{s}" height="{s}" preserveAspectRatio="xMidYMid meet"/>'
    return f'<g id="cherry-slot" data-x="{x}" data-y="{y}" data-size="{s}"></g>'   # empty until artwork lands


def tag(x, y, m, th, size=11.5):
    t = m['tag']; w = len(t) * size * 0.68 + 16
    if m.get('cherry'):
        s = size + 7
        slot = cherry_icon(x + 50, y - size - 3 + (size + 9 - s) / 2, s)
        mascot = ROOT / 'assets' / 'retro-wink-mischievous.png'
        if mascot.exists():
            uri = 'data:image/png;base64,' + __import__('base64').b64encode(mascot.read_bytes()).decode()
            left, top, width, height = 1044, y - 76, 134, 170
            cx, cy = left + width / 2, top + height / 2
            slot += (f'<g id="wink-sticker" transform="rotate(-9 {cx} {cy}) translate({2 * cx} 0) scale(-1 1)">'
                     f'<image href="{uri}" x="{left}" y="{top}" width="{width}" height="{height}" '
                     'preserveAspectRatio="xMidYMid meet"/></g>')
        return slot, 135
    if m['r']['source'] == 'ours':
        return (f'<rect x="{x}" y="{y-size-3}" width="{w}" height="{size+9}" rx="{(size+9)/2}" fill="{AMBER}"/>' +
                text(x + w / 2, y + 1.5, t, size, '#1b1400', 800, 'middle', ls=0.6)), w
    dash = '' if m['r']['source'] == 'leaderboard' else 'stroke-dasharray="3 2.5"'
    return (f'<rect x="{x+0.75}" y="{y-size-2.25}" width="{w-1.5}" height="{size+7.5}" rx="{(size+9)/2}" fill="none" '
            f'stroke="{th["fg2"]}" stroke-width="1.4" {dash}/>' +
            text(x + w / 2, y + 1.5, t, size, th['fg2'], 700, 'middle', ls=0.6)), w


def title_block(th, title, sub):
    b = text(60, 82, title, 38, th['fg'], 700)
    b += text(60, 120, sub, 19, th['fg1'], 400)
    b += fa_lockup(W - 60 - LOCK_W(28), 54, 28, th['fa_text'])
    return b


def col_head(x, y, main, sub, th, anchor='start', size=26):
    s = text(x, y, main, size, th['fg2'], 700, anchor)
    if sub:
        if anchor == 'start':
            s += text(x + len(main) * size * 0.6 + 10, y, sub, 16, th['fg2'], 400)
        else:
            s += text(x - len(main) * size * 0.62 - 10, y, sub, 16, th['fg2'], 400, 'end')
    return s


def footer(th, lines, size=12.5, y0=None):
    y0 = y0 or H - 88
    s = f'<line x1="60" y1="{y0-22}" x2="{W-60}" y2="{y0-22}" stroke="{th["line"]}"/>'
    for i, l in enumerate(lines):
        s += text(60, y0 + i * (size + 7), l, size, th['fg2'], 400)
    return s


FOOT = [
    'All rows: Claude Fable 5 on Terminal-Bench 2.1, 89 tasks. Cost = total for one 89-task run, as in the Strands chart.',
    f"Terminus 2 (TB2.1 leaderboard {TA['pr']}; published {TA['lb_score']*100:.1f}% / ${TA['lb_cost_total']:.2f} over 5×89) re-scored Fable-only: "
    f"{TA['fallback_trials']} trials served by the Opus 4.8 server-side fallback score 0 and their ${TA['opus_cost_recorded']:.2f} Opus cost is removed.",
    f'Strands / OpenCode / Oh-my-pi: as published in the Strands chart (89 trials; effort, cost method and run selection not stated). '
    f'fast-agent {FA_VER} (ours): Copilot route, HF Jobs, 1×89, 0 Harbor retries; medium incl. 4 setup replacements.',
    'Error bars: 95% CI from task sampling (binomial for single 89-task runs; task-clustered for the Terminus 5-run average). '
    'Fable safety stops score 0 in our runs and in Terminus. Token-price estimates, not bills.',
]


def page(body, th, name):
    # Place the sticker above grid lines, as a physical sticker would sit.
    stickers = re.findall(r'<g id="(?:wink|cherry)-sticker".*?</g>', body)
    body = re.sub(r'<g id="(?:wink|cherry)-sticker".*?</g>', '', body) + ''.join(stickers)
    HTML.mkdir(parents=True, exist_ok=True); OUT.mkdir(exist_ok=True)
    html = f'''<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap">
<style>html,body{{margin:0;width:{W}px;height:{H}px;overflow:hidden;background:{th["bg"]}}}</style></head>
<body><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>{defs(th)}</defs>
<rect width="{W}" height="{H}" fill="{th["bg"]}"/>{body}</svg></body></html>'''
    p = HTML / f'{name}.html'; p.write_text(html)
    png = (ROOT if name in FINAL else OUT) / f'{name}.png'
    subprocess.run(['chromium', '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--virtual-time-budget=4000', '--force-device-scale-factor=2', f'--screenshot={png}',
                    f'file://{p}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
    print('wrote', png)


TAG_X = 300


def label(k, x, cy, th, compact=False, tight=False):
    m = meta(k, th)
    if tight:   # 8+ rows
        isz, ns, s2, s3, d1, d2, d3 = 36, 20, 14.5, 12, -13, 7, 24
    elif compact:
        isz, ns, s2, s3, d1, d2, d3 = 40, 22, 15.5, 12.5, -10, 13, 32
    else:
        isz, ns, s2, s3, d1, d2, d3 = 46, 24, 17, 13.5, -10, 15, 36
    b = icon(k, x, cy - isz / 2 - 6, isz, th)
    tx = x + isz + 18
    b += text(tx, cy + d1, m['title'], ns, th['fg'], 700)
    b += tag(TAG_X, cy + d1 - 2, m, th, 10.5 if tight else (11 if compact else 11.5))[0]
    b += text(tx, cy + d2, m['line2'], s2, m['txt'], 700)
    b += text(tx, cy + d3, m['line3'], s3, th['fg2'], 400)
    return b


def keys(terms, by):
    ks = BASE + TERMS[terms]
    if by == 'cost': return sorted(ks, key=lambda k: D[k]['cost'])
    return sorted(ks, key=lambda k: (-round(D[k]['score'], 3), D[k]['cost']))


SUB = 'Claude Fable 5 · Terminal-Bench 2.1 · 89 tasks · total cost per run'
TITLE = 'Fable 5 harness comparison'


def whisker(x1, x2, y, th, h=8, color=None):
    c = color or th['fg2']
    return (f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{c}" stroke-width="1.8"/>'
            f'<line x1="{x1:.1f}" y1="{y-h/2}" x2="{x1:.1f}" y2="{y+h/2}" stroke="{c}" stroke-width="1.8"/>'
            f'<line x1="{x2:.1f}" y1="{y-h/2}" x2="{x2:.1f}" y2="{y+h/2}" stroke="{c}" stroke-width="1.8"/>')


def run_dots(r, X, y, th):
    if r.get('variant') != 'avg': return ''
    return ''.join(f'<circle cx="{X(rr["score"]):.1f}" cy="{y}" r="3.6" fill="{th["lb"]}" stroke="{th["bg"]}" stroke-width="1.2"/>'
                   for rr in r['runs'])


def layout(n, bottom=762):
    top = 206
    rowh = min(94, (bottom - top) // n)
    return top, rowh, n >= 7


# ================================================================== v15-style: accuracy (with CI) + total cost
def rows_dual(theme='light', name='h10', terms='both', by='score', title=TITLE):
    th = THEMES[theme]
    ks = keys(terms, by); top, rowh, compact = layout(len(ks))
    b = title_block(th, title, SUB)
    sx0, sw = 520, 330
    cx0, cw, cmax, nr = 1030, 230, 95, W - 60
    b += col_head(sx0, top - 12, 'Accuracy', '', th)
    b += whisker(sx0 + 138, sx0 + 160, top - 18, th) + text(sx0 + 168, top - 12, '95% CI', 15, th['fg2'], 500)
    if 'terminus_avg' in ks:
        b += f'<circle cx="{sx0+238}" cy="{top-18}" r="3.6" fill="{th["lb"]}"/>' + text(sx0 + 248, top - 12, 'Terminus runs', 15, th['fg2'], 500)
    b += col_head(cx0, top - 12, 'Cost', '· total, 89 tasks', th)
    X = lambda s: sx0 + sw * s
    y = top + 8
    for i, k in enumerate(ks):
        m = meta(k, th); r = m['r']
        if i: b += f'<line x1="60" y1="{y-1}" x2="{W-60}" y2="{y-1}" stroke="{th["line"]}"/>'
        cy = y + rowh / 2 + 2
        b += label(k, 60, cy, th, compact)
        bh = 34 if compact else 40; by_ = cy - bh / 2 - 8
        b += f'<rect x="{sx0}" y="{by_}" width="{sw}" height="{bh}" rx="6" fill="{th["track"]}"/>'
        b += f'<rect x="{sx0}" y="{by_}" width="{sw*r["score"]:.1f}" height="{bh}" rx="6" fill="{fill(m)}"/>'
        wy = by_ + bh + 10
        b += whisker(X(r['score'] - r['ci95']), X(min(1, r['score'] + r['ci95'])), wy, th)
        b += run_dots(r, X, wy, th)
        b += text(sx0 + sw + 18, by_ + bh / 2 + 14, f"{r['score']*100:.1f}%", 38 if compact else 40, th['fg'], 800)
        w = cw * r['cost'] / cmax
        b += f'<rect x="{cx0}" y="{by_}" width="{w:.1f}" height="{bh}" rx="6" fill="{fill(m)}"/>'
        b += text(nr, by_ + bh / 2 + 14, f"${r['cost']:.2f}", 38 if compact else 40, th['fg'], 800, 'end')
        y += rowh
    b += footer(th, FOOT, 11.5)
    page(b, th, name)


# ================================================================== forest plot: accuracy dot + CI on a zoomed axis, cost column
def rows_forest(theme='light', name='h12', terms='both', by='score', title=TITLE, lo=0.50, hi=0.90):
    th = THEMES[theme]
    ks = keys(terms, by); top, rowh, compact = layout(len(ks))
    b = title_block(th, title, SUB)
    ax0, ax1 = 520, 1010
    X = lambda s: ax0 + (s - lo) / (hi - lo) * (ax1 - ax0)
    nx = ax1 + 30; cx0, cw, cmax, nr = 1160, 110, 95, W - 60
    b += col_head(ax0, top - 12, 'Accuracy', '· 95% CI', th)
    b += col_head(nr, top - 12, 'Cost', '', th, anchor='end')
    ybot = top + 8 + len(ks) * rowh
    s = lo
    while s <= hi + 1e-9:
        b += f'<line x1="{X(s)}" y1="{top+4}" x2="{X(s)}" y2="{ybot-6}" stroke="{th["grid"]}"/>'
        b += text(X(s), ybot + 16, f'{s*100:.0f}%', 13, th['fg2'], 500, 'middle')
        s += 0.10
    y = top + 8
    for i, k in enumerate(ks):
        m = meta(k, th); r = m['r']
        if i: b += f'<line x1="60" y1="{y-1}" x2="{W-60}" y2="{y-1}" stroke="{th["line"]}"/>'
        cy = y + rowh / 2 + 2
        b += label(k, 60, cy, th, compact)
        py = cy - 8
        c1, c2 = X(r['score'] - r['ci95']), X(min(1, r['score'] + r['ci95']))
        b += f'<line x1="{c1:.1f}" y1="{py}" x2="{c2:.1f}" y2="{py}" stroke="{m["color"]}" stroke-width="6" stroke-linecap="round" opacity="0.35"/>'
        b += run_dots(r, X, py + 12, th)
        if r['source'] == 'ours':
            sz = 30; b += fa_icon(X(r['score']) - sz / 2, py - sz / 2, sz, th)
        elif m['solid']:
            b += f'<circle cx="{X(r["score"])}" cy="{py}" r="11" fill="{m["color"]}"/>'
        else:
            b += f'<circle cx="{X(r["score"])}" cy="{py}" r="10" fill="{th["bg"]}" stroke="{m["color"]}" stroke-width="3.5"/>'
        b += text(nx, py + 13, f"{r['score']*100:.1f}%", 36, th['fg'], 800)
        b += text(nr, py + 13, f"${r['cost']:.2f}", 36, th['fg'], 800, 'end')
        y += rowh
    b += footer(th, FOOT, 11.5)
    page(b, th, name)


# ================================================================== Pareto with CI whiskers
def pareto(theme='light', name='h14', terms='both', title='Fable 5 harness comparison: cost vs accuracy',
           xr=(50, 95), yr=(0.50, 0.90), ci=True, side=True):
    th = THEMES[theme]
    ks = BASE + TERMS[terms]
    b = title_block(th, title, SUB)
    x0, x1, y0, y1 = 150, (1060 if side else W - 90), 712, 178
    X = lambda c: x0 + (c - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
    Y = lambda s: y0 - (s - yr[0]) / (yr[1] - yr[0]) * (y0 - y1)
    pts = sorted(ks, key=lambda k: (D[k]['cost'], -D[k]['score']))
    front, best = [], -1
    for k in pts:
        if D[k]['score'] > best: front.append(k); best = D[k]['score']
    path = f'M{X(D[front[0]]["cost"])},{y0} '
    for i, k in enumerate(front):
        path += f'L{X(D[k]["cost"])},{Y(D[k]["score"])} '
        nxt = X(D[front[i+1]]['cost']) if i + 1 < len(front) else x1
        path += f'L{nxt},{Y(D[k]["score"])} '
    b += f'<path d="{path}L{x1},{y0} Z" fill="{th["fg2"]}" opacity="0.07"/>'
    for c in range(50, 96, 5):
        b += f'<line x1="{X(c)}" y1="{y1}" x2="{X(c)}" y2="{y0}" stroke="{th["grid"]}"/>'
        if c % 10 == 0: b += text(X(c), y0 + 28, f'${c}', 15, th['fg2'], 500, 'middle')
    s = yr[0]
    while s <= yr[1] + 1e-9:
        b += f'<line x1="{x0}" y1="{Y(s)}" x2="{x1}" y2="{Y(s)}" stroke="{th["grid"]}"/>'
        b += text(x0 - 16, Y(s) + 5, f'{s*100:.0f}%', 15, th['fg2'], 500, 'end')
        s += 0.05
    b += text((x0 + x1) / 2, y0 + 62, 'Total cost, 89 tasks →', 20, th['fg2'], 700, 'middle')
    b += text(x0 - 78, (y0 + y1) / 2, 'Accuracy →', 20, th['fg2'], 700, 'middle', extra=f'transform="rotate(-90 {x0-78} {(y0+y1)/2})"')
    fl = ''
    for i, k in enumerate(front):
        fl += f'{"M" if i == 0 else "L"}{X(D[k]["cost"])},{Y(D[k]["score"])} '
        if i + 1 < len(front): fl += f'L{X(D[front[i+1]]["cost"])},{Y(D[k]["score"])} '
    fl += f'L{x1},{Y(D[front[-1]]["score"])}'
    b += f'<path d="{fl}" fill="none" stroke="{th["fg1"]}" stroke-width="2" stroke-dasharray="6 5"/>'
    b += text(x1 - 8, Y(D[front[-1]]['score']) - 10, 'Pareto frontier', 14, th['fg1'], 600, 'end')
    lab = {'fa_medium': (-30, -34, 'end'), 'fa_high': (0, -40, 'middle'), 'terminus_avg': (24, 6, 'start'),
           'terminus_best': (-20, -34, 'end'), 'strands': (-22, 40, 'end'), 'opencode': (0, 40, 'middle'), 'ohmypi': (0, -48, 'middle')}
    for k in ks:
        m = meta(k, th); r = m['r']; px, py = X(r['cost']), Y(r['score'])
        if ci:
            b += f'<line x1="{px}" y1="{Y(r["score"]-r["ci95"])}" x2="{px}" y2="{Y(min(yr[1], r["score"]+r["ci95"]))}" stroke="{m["color"]}" stroke-width="2.5" opacity="0.4"/>'
        if r.get('variant') == 'avg':
            for rr in r['runs']:
                b += f'<circle cx="{X(rr["cost"])}" cy="{Y(rr["score"])}" r="4" fill="{th["lb"]}" opacity="0.45"/>'
        if r['source'] == 'ours':
            sz = 40
            b += fa_icon(px - sz / 2, py - sz / 2, sz, th)
            dash = '' if m['solid'] else 'stroke-dasharray="4 3"'
            b += f'<rect x="{px-sz/2-5}" y="{py-sz/2-5}" width="{sz+10}" height="{sz+10}" rx="9" fill="none" stroke="{AMBER}" stroke-width="2.5" {dash}/>'
        elif m['solid']:
            b += f'<circle cx="{px}" cy="{py}" r="12" fill="{m["color"]}"/>' if r['source'] == 'leaderboard' else \
                 f'<circle cx="{px}" cy="{py}" r="11" fill="{th["bg"]}" stroke="{m["color"]}" stroke-width="3"/>'
        else:
            b += f'<circle cx="{px}" cy="{py}" r="11" fill="{th["bg"]}" stroke="{m["color"]}" stroke-width="3.5"/>'
        dx, dy, an = lab[k]
        if r['source'] == 'ours': nm = f"fast-agent {r['effort']}"
        elif r['source'] == 'leaderboard': nm = f"Terminus 2 {'5-run avg' if r.get('variant') == 'avg' else 'best run'}"
        else: nm = m['title']
        b += text(px + dx, py + dy, nm, 19, th['fg'], 700, an)
        b += text(px + dx, py + dy + 21, f"{r['score']*100:.1f}% · ${r['cost']:.2f}", 15.5, m['txt'], 600, an)
    if side:
        sx = 1110
        b += text(sx, 200, 'Key', 22, th['fg2'], 700)
        yy = 236
        b += fa_icon(sx, yy - 16, 24, th) + text(sx + 36, yy + 2, 'our runs (fast-agent)', 15, th['fg1'], 600); yy += 34
        b += f'<circle cx="{sx+12}" cy="{yy-4}" r="10" fill="{th["lb"]}"/>' + text(sx + 36, yy + 2, 'Terminus 2, best run', 15, th['fg1'], 600); yy += 34
        if 'terminus_avg' in ks:
            b += f'<circle cx="{sx+12}" cy="{yy-4}" r="9" fill="{th["bg"]}" stroke="{th["lb"]}" stroke-width="3.5"/>' + text(sx + 36, yy + 2, 'Terminus 2, 5-run avg', 15, th['fg1'], 600); yy += 34
            b += f'<circle cx="{sx+12}" cy="{yy-4}" r="4" fill="{th["lb"]}" opacity="0.45"/>' + text(sx + 36, yy + 2, 'individual Terminus runs', 15, th['fg1'], 600); yy += 34
        b += f'<circle cx="{sx+12}" cy="{yy-4}" r="9" fill="{th["bg"]}" stroke="{th["pub"]}" stroke-width="3"/>' + text(sx + 36, yy + 2, 'published by Strands', 15, th['fg1'], 600); yy += 34
        if ci:
            b += f'<line x1="{sx+12}" y1="{yy-16}" x2="{sx+12}" y2="{yy+8}" stroke="{th["fg2"]}" stroke-width="2.5" opacity="0.5"/>' + text(sx + 36, yy + 2, '95% CI (task sampling)', 15, th['fg1'], 600); yy += 34
        yy += 16
        b += text(sx, yy, 'Notes', 22, th['fg2'], 700); yy += 30
        notes = ['Terminus is Fable-only: its', f"{TA['fallback_trials']} Opus-fallback trials score 0,", f"as Fable refusals do for us.", '',
                 'fast-agent medium: highest', f"accuracy, ${D['fa_medium']['cost']-D['strands']['cost']:.2f} more than Strands."]
        for n in notes:
            b += text(sx, yy, n, 15, th['fg1'], 500); yy += 21
    b += footer(th, FOOT, 11.5)
    page(b, th, name)


# ================================================================== 89-trial waffle instead of the accuracy bar
TASKS = sorted(D['fa_high']['tasks'])
_per = [k for k in ('fa_medium', 'fa_high', 'terminus_avg') if D[k].get('tasks')]
# task order: easiest first (mean Fable-only pass rate across rows with per-task data), then name
TASK_ORDER = sorted(TASKS, key=lambda t: (-sum(float(D[k]['tasks'][t] or 0) for k in _per), t))


def waffle(x, y, r, m, th, cols=30, cell=10.5, gap=3, aligned=True, refusals=True):
    """89 cells, column-major (3 per column) so passes read left-to-right like a bar."""
    rows = math.ceil(89 / cols)
    has = bool(r.get('tasks'))
    if has and aligned:
        seq = [(float(r['tasks'][t] or 0), float(r['task_refused'][t] or 0)) for t in TASK_ORDER]
    elif has:
        seq = sorted(((float(r['tasks'][t] or 0), float(r['task_refused'][t] or 0)) for t in TASKS), key=lambda v: (-v[0], v[1]))
    else:  # count only (per-task results not published)
        p = round(r['score'] * 89); seq = [(1.0, 0.0)] * p + [(0.0, 0.0)] * (89 - p)
    out = []
    fail_c = '#c9c4b8' if th['bg'].startswith('#f') else '#4a4740'
    for i, (v, ref) in enumerate(seq):
        cx = x + (i // rows) * (cell + gap); cy = y + (i % rows) * (cell + gap)
        if not has:  # published count only: outlined-fill style
            if v:
                out.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="2" fill="{m["color"]}" opacity="0.75"/>')
            else:
                out.append(f'<rect x="{cx+0.8}" y="{cy+0.8}" width="{cell-1.6}" height="{cell-1.6}" rx="2" fill="none" stroke="{fail_c}" stroke-width="1.4" stroke-dasharray="2 1.6"/>')
            continue
        if v > 0:
            op = 1 if v >= 1 else 0.25 + 0.75 * v
            out.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="2" fill="{m["color"]}" opacity="{op:.2f}"/>')
        if v < 1:
            if refusals and ref >= 0.5:
                out.append(f'<rect x="{cx+0.8}" y="{cy+0.8}" width="{cell-1.6}" height="{cell-1.6}" rx="2" fill="none" stroke="{REFUSE}" stroke-width="1.6"/>'
                           f'<line x1="{cx+2.2}" y1="{cy+cell-2.2}" x2="{cx+cell-2.2}" y2="{cy+2.2}" stroke="{REFUSE}" stroke-width="1.6"/>')
            elif v == 0:
                out.append(f'<rect x="{cx+0.8}" y="{cy+0.8}" width="{cell-1.6}" height="{cell-1.6}" rx="2" fill="none" stroke="{fail_c}" stroke-width="1.4"/>')
    return ''.join(out), math.ceil(89 / rows) * (cell + gap) - gap, rows * (cell + gap) - gap


REFUSE = '#c2410c'


def rows_waffle(theme='light', name='h20', terms='both', by='score', title=TITLE, aligned=True, refusals=True,
                cols=30, cell=10.5, gap=3, ci=False, cost_min=0, cost_max=95, cost_ticks=None,
                cost_style='bar', publish=False):
    th = THEMES[theme]
    ks = keys(terms, by); top, rowh, compact = layout(len(ks), 742 if publish else 762)
    b = title_block(th, title, SUB)
    sx0 = 520
    cx0, cw, nr = (1120, 150, W - 60) if not cost_min else (1112, 168, W - 60)
    CX = lambda c: cx0 + cw * (c - cost_min) / (cost_max - cost_min)
    ybot = top + 8 + len(ks) * rowh
    if cost_ticks:
        for c in cost_ticks:
            b += f'<line x1="{CX(c):.1f}" y1="{top+6}" x2="{CX(c):.1f}" y2="{ybot-4}" stroke="{th["grid"]}" stroke-width="1.2"/>'
            b += text(CX(c), top + len(ks) * rowh + 26, f'${c}', 13.5, th['fg2'], 500, 'middle')
    b += col_head(sx0, top - 12, 'Accuracy', '· one cell per task, ' + ('easiest → hardest' if aligned else 'passes first'), th)
    if cost_style == 'dot':
        b += col_head(cx0, top - 12, 'Cost', '· total per run', th)
    else:
        b += col_head(cx0, top - 12, 'Cost', '· total' + (f' · axis from ${cost_min}' if cost_min else ''), th)
    y = top + 8
    for i, k in enumerate(ks):
        m = meta(k, th); r = m['r']
        if i: b += f'<line x1="60" y1="{y-1}" x2="{W-60}" y2="{y-1}" stroke="{th["line"]}"/>'
        cy = y + rowh / 2 + 2
        b += label(k, 60, cy, th, compact, tight=len(ks) >= 8)
        g, gw, gh = waffle(0, 0, r, m, th, cols, cell, gap, aligned, refusals)
        gy = cy - gh / 2 - 6
        b += f'<g transform="translate({sx0},{gy:.1f})">{g}</g>'
        if ci:
            X = lambda s: sx0 + gw * s
            b += whisker(X(r['score'] - r['ci95']), X(min(1, r['score'] + r['ci95'])), gy + gh + 8, th, h=6)
        b += text(sx0 + gw + 22, cy + 8, f"{r['score']*100:.1f}%", 36 if compact else 40, th['fg'], 800)
        if cost_style == 'dot':
            py = cy - 6; px = CX(r['cost'])
            b += f'<line x1="{cx0}" y1="{py}" x2="{cx0+cw}" y2="{py}" stroke="{th["line"]}" stroke-width="1.5"/>'
            for o in (r.get('other_runs') or []):
                if o.get('cost') is not None:
                    ox = CX(o['cost'])
                    b += (f'<circle cx="{ox:.1f}" cy="{py}" r="7" fill="{th["bg"]}" stroke="{AMBER}" stroke-width="2" stroke-dasharray="3 2.5" opacity="0.9"/>'
                          + text(ox, py - 14, 'other run', 11.5, th['fg2'], 600, 'middle'))
            if r['source'] == 'ours' and not m['solid']:
                b += f'<circle cx="{px:.1f}" cy="{py}" r="9" fill="{th["bg"]}" stroke="{AMBER}" stroke-width="4"/>'
            else:
                b += f'<circle cx="{px:.1f}" cy="{py}" r="10" fill="{m["color"]}" stroke="{th["bg"]}" stroke-width="2"/>'
        else:
            w = CX(r['cost']) - cx0
            bh = 30 if compact else 36
            rx = 6 if not cost_min else 3
            b += f'<rect x="{cx0}" y="{cy - bh/2 - 6}" width="{w:.1f}" height="{bh}" rx="{rx}" fill="{fill(m)}"/>'
        if m.get('cherry'):
            o = r['other_runs'][0]
            oc = f", ${o['cost']:.2f}" if o.get('cost') is not None else ''
            b += text(sx0, cy + 31, f"Highest-scoring of two runs shown; other run: {o['passes']}/{o['n']} ({o['passes']/o['n']*100:.1f}%){oc}.",
                      14, th['fg1'], 600)

        b += text(nr, cy + 8, f"${r['cost']:.2f}", 36 if compact else 40, th['fg'], 800, 'end')
        y += rowh
    # legend
    ly = top + len(ks) * rowh + 26; lx = sx0; c = 11
    fail_c = '#c9c4b8' if th['bg'].startswith('#f') else '#4a4740'
    items = [(f'<rect x="{lx}" y="{ly-c+1}" width="{c}" height="{c}" rx="2" fill="{th["fg1"]}"/>', 'pass'),
             (f'<rect x="{lx}" y="{ly-c+1}" width="{c}" height="{c}" rx="2" fill="none" stroke="{fail_c}" stroke-width="1.4"/>', 'fail')]
    if refusals:
        items.append((f'<rect x="{lx}" y="{ly-c+1}" width="{c}" height="{c}" rx="2" fill="none" stroke="{REFUSE}" stroke-width="1.6"/>'
                      f'<line x1="{lx+2}" y1="{ly-1}" x2="{lx+c-2}" y2="{ly-c+3}" stroke="{REFUSE}" stroke-width="1.6"/>', 'Fable refused / fallback (scored 0)'))
    if 'terminus_avg' in ks:
        items.append((f'<rect x="{lx}" y="{ly-c+1}" width="{c}" height="{c}" rx="2" fill="{th["lb"]}" opacity="0.45"/>', 'avg: shade = passes / 5' if cost_ticks else '5-run avg: shade = passes / 5'))
    if not cost_ticks:
        items.append((f'<rect x="{lx+0.8}" y="{ly-c+1.8}" width="{c-1.6}" height="{c-1.6}" rx="2" fill="none" stroke="{fail_c}" stroke-dasharray="2 1.6"/>', 'count only (no per-task data published)'))
    off = 0
    for sw, lab in items:
        b += f'<g transform="translate({off},0)">{sw}</g>' + text(lx + off + c + 7, ly, lab, 13.5, th['fg2'], 500)
        off += c + 7 + len(lab) * 6.9 + 26
    foot = FOOT[:3] + [('Cells: same easiest→hardest task order in every row with per-task data; Strands-chart rows (dashed empties) show counts only. ' if aligned else
                        'Cells: passes sorted first; Strands-chart rows (dashed empties) show counts only. ') +
                       'Single 89-task runs ≈ ±9 pts 95% CI (task sampling). Token-price estimates, not bills.']
    if ci: foot = FOOT
    if publish:
        foot = [
            'All rows: Claude Fable 5, Terminal-Bench 2.1, 89 tasks; cost = total for one 89-task run, as in the Strands chart. '
            'Strands / OpenCode / Oh-my-pi: as published by Strands (effort, cost method, run selection not stated).',
            f"Terminus 2: TB2.1 leaderboard {TA['pr']}, {TA['n_total']} trials run with an Opus 4.8 server-side fallback (published {TA['lb_score']*100:.1f}%, ${TA['lb_cost_total']:.2f}). "
            f"Fallback trials counted as failures; costs of those trials excluded ({TA['fallback_trials']} trials, ${TA['opus_cost_recorded']:.2f}).",
            "Five 89-task groups reconstructed by ordering each task's attempts by start time. Score ties resolved by lower cost for the highest-scoring group "
            f"and higher cost for the lowest-scoring group. Adjusted average over all {TA['n_total']} trials: {TA['score']*100:.1f}%, ${TA['cost']:.2f} per 89 tasks.",
            f'fast-agent {FA_VER} (ours): Copilot route, HF Jobs, 1×89, 0 Harbor retries; medium incl. 4 setup replacements; high = higher-scoring of two runs. '
            'Pinned infrastructure-only QEMU fixes for qemu-alpine-ssh and qemu-startup (Terminus passed both 5/5 unmodified).',
            'Cells: passes sorted first; Strands-chart rows (dashed empties) show counts only. Single 89-task runs ≈ ±9 pts 95% CI (task sampling). '
            'Costs are token-price estimates, not bills.']
    elif terms == 'bestworst':
        foot = [
            'All rows: Claude Fable 5, Terminal-Bench 2.1, 89 tasks; cost = total for one 89-task run, as in the Strands chart. '
            'Strands / OpenCode / Oh-my-pi: as published by Strands (effort, cost method, run selection not stated).',
            '',
            f'fast-agent {FA_VER} (ours): Copilot route, HF Jobs, 1×89, 0 Harbor retries; medium incl. 4 setup replacements. Pinned infrastructure-only '
            'QEMU fixes for qemu-alpine-ssh and qemu-startup on HF Jobs (Terminus passed both 5/5 unmodified).',
            ('Cells: passes sorted first; Strands-chart rows (dashed empties) show counts only. ' if not aligned else
             'Cells: same easiest→hardest task order in every row with per-task data; Strands-chart rows (dashed empties) show counts only. ') +
            'Single 89-task runs ≈ ±9 pts 95% CI (task sampling). Token-price estimates, not bills.']
        foot[1] = (f"Terminus 2 (TB2.1 leaderboard {TA['pr']}; published {TA['lb_score']*100:.1f}% / ${TA['lb_cost_total']:.2f} over 5×89), Fable-only: {TA['fallback_trials']} Opus-fallback trials score 0, "
                   f"${TA['opus_cost_recorded']:.2f} Opus cost removed. Best / worst of 5 runs shown; 5-run average {TA['score']*100:.1f}%, ${TA['cost']:.2f}.")
    b += footer(th, foot, 11 if publish else 11.5, H - 100 if publish else None)
    page(b, th, name)


if __name__ == '__main__':
    import sys
    # the chosen chart -> root folder
    rows_waffle('light', 'fable5_harness_comparison', 'bestworst', by='cost', aligned=False,
                cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90], cost_style='dot', publish=True)
    if '--all' in sys.argv:   # every candidate -> archive/candidates (+ archive/contact_sheet.jpg)
        rows_waffle('light', 'h26_bestworst_waffle_sorted_bycost_light', 'bestworst', by='cost', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('dark', 'h26b_bestworst_waffle_sorted_bycost_dark', 'bestworst', by='cost', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('light', 'h25_range_waffle_sorted_cost50_light', 'range', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('dark', 'h25b_range_waffle_sorted_cost50_dark', 'range', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('light', 'h25c_bestworst_waffle_sorted_cost50_light', 'bestworst', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('light', 'h25d_range_waffle_aligned_cost50_light', 'range', aligned=True, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('light', 'h24_both_waffle_sorted_cost50_light', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('light', 'h24b_both_waffle_sorted_cost50_noticks_light', aligned=False, cost_min=50, cost_max=90)
        rows_waffle('light', 'h24c_both_waffle_sorted_cost40_light', aligned=False, cost_min=40, cost_max=90, cost_ticks=[40, 50, 60, 70, 80, 90])
        rows_waffle('dark', 'h24d_both_waffle_sorted_cost50_dark', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('light', 'h24e_best_waffle_sorted_cost50_light', 'best', aligned=False, cost_min=50, cost_max=90, cost_ticks=[50, 60, 70, 80, 90])
        rows_waffle('light', 'h20_both_waffle_aligned_light')
        rows_waffle('dark', 'h20_both_waffle_aligned_dark')
        rows_waffle('light', 'h20_best_waffle_aligned_light', 'best')
        rows_waffle('light', 'h21_both_waffle_sorted_light', aligned=False)
        rows_waffle('light', 'h21_both_waffle_sorted_norefusal_light', aligned=False, refusals=False)
        rows_waffle('light', 'h22_both_waffle_aligned_ci_light', ci=True)
        rows_waffle('light', 'h23_both_waffle_45x2_light', cols=45, cell=8, gap=2.2)
        for terms in ('both', 'best', 'avg'):
            rows_dual('light', f'h10_{terms}_dual_by_accuracy_light', terms)
        rows_dual('dark', 'h10_both_dual_by_accuracy_dark', 'both')
        rows_dual('light', 'h11_both_dual_by_cost_light', 'both', by='cost')
        rows_dual('light', 'h11_best_dual_by_cost_light', 'best', by='cost')
        for terms in ('both', 'best'):
            rows_forest('light', f'h12_{terms}_forest_light', terms)
        rows_forest('dark', 'h12_both_forest_dark', 'both')
        for terms in ('both', 'best'):
            pareto('light', f'h14_{terms}_pareto_ci_light', terms)
        pareto('dark', 'h14_both_pareto_ci_dark', 'both')
        pareto('light', 'h14_both_pareto_noci_light', 'both', ci=False, yr=(0.60, 0.80))
        subprocess.run(['magick', 'montage', *sorted(str(p) for p in OUT.glob('h*.png')), '-tile', '3x', '-geometry',
                        '750x450+6+6', '-background', '#333', '-label', '%t', '-pointsize', '14', '-fill', 'white',
                        str(ROOT / 'archive' / 'contact_sheet.jpg')])
