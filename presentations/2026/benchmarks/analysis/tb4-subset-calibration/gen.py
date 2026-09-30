"""Two information sheets for the TB4 subset, built from analysis.json.

    python3 gen.py      # -> sheet1_score.png, sheet2_cost.png (+ html/ sources)
"""
import json, math, pathlib, random, subprocess, statistics as st
from html import escape

ROOT = pathlib.Path(__file__).resolve().parent
A = json.load(open(ROOT / 'analysis.json'))
W, H = 1600, 1240
SANS = "'Inter','Noto Sans',sans-serif"
BG, FG, FG1, FG2, LINE, GRID = '#faf9f5', '#1f1e1d', '#4a4843', '#8a867d', '#dedad0', '#ebe8df'
DOT, ACC, ACC_T, BAND, CLOUD = '#4d5059', '#d9774f', '#b8552d', '#f3e6dd', '#b9b4a8'

ROWS = A['rows']; SC = [r for r in ROWS if r['score_ok']]; CO = [r for r in ROWS if r['cost_ok']]
SCORE, COST, RB = A['score'], A['cost'], A['random']
N = len(A['subset']); NTR = 5 * N; NF = A['n_tasks']; NREST = NF - N; NAIVE = NF / N
A_ALL = json.load(open(ROOT / 'analysis_all66.json'))
ELIG = set(A['eligible'])


def T(x, y, s, size=16, fill=FG, weight=400, anchor='start', extra=''):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>')


def L(x1, y1, x2, y2, c=LINE, w=1, extra=''):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" {extra}/>'


def C(x, y, r, fill=DOT, stroke=BG, sw=2, extra=''):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def header(title, sub):
    return T(60, 84, title, 40, FG, 800) + T(60, 124, sub, 19, FG1)


def kpis(y, items):
    """items: (big, big_colour, label, note). Four tiles across the page."""
    n = len(items); gap = 28; w = (W - 120 - gap * (n - 1)) / n; s = ''
    for i, (big, col, lab, note) in enumerate(items):
        x = 60 + i * (w + gap)
        s += f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="150" rx="14" fill="#f3f1ea"/>'
        s += T(x + 24, y + 66, big, 50, col, 800)
        s += T(x + 24, y + 102, lab, 17.5, FG, 700)
        s += T(x + 24, y + 128, note, 15, FG2, 500)
    return s


def panel_head(x, y, title, sub):
    return T(x, y, title, 22, FG, 750) + T(x, y + 26, sub, 15, FG2, 500)


def footer(lines, y0, takeaway):
    s = L(60, y0 - 70, W - 60, y0 - 70)
    s += T(60, y0 - 30, takeaway, 25, FG, 800)
    for i, l in enumerate(lines):
        s += T(60, y0 + 6 + i * 21, l, 13, FG2)
    return s


def page(body, name):
    (ROOT / 'html').mkdir(exist_ok=True)
    html = f'''<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap">
<style>html,body{{margin:0;width:{W}px;height:{H}px;overflow:hidden;background:{BG}}}</style></head>
<body><svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs><clipPath id="plotclip"><rect x="0" y="0" width="{W}" height="{H}"/></clipPath></defs>
<rect width="{W}" height="{H}" fill="{BG}"/>{body}</svg></body></html>'''
    p = ROOT / 'html' / f'{name}.html'; p.write_text(html)
    (ROOT / 'html' / f'{name}.svg').write_text(html.split('<body>')[1].split('</body>')[0])
    png = ROOT / f'{name}.png'
    subprocess.run(['chromium', '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--window-size={W},{H}',
                    '--virtual-time-budget=5000', '--force-device-scale-factor=2', f'--screenshot={png}', f'file://{p}'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print('wrote', png)


def lab(r): return f"{r['model']} / {r['effort']}"


# ====================================================================== sheet 1: score
def sheet_score():
    b = header(f'Can {N} tasks stand in for Terminal-Bench 4.0?',
               f"A frozen {N}-task subset × 5 attempts, checked against {A['n_score']} public TB 4.0 leaderboard rows "
               f"({A['n_models_score']} models) · {NTR} trials instead of 330")
    share = st.median(r['cost_share'] for r in CO)
    b += kpis(160, [
        (f"{share*100:.0f}%", ACC, 'of the full-run cost', f"{A['time_share']*100:.0f}% of wall time · random eligible {N}: {A['cloud_median_share']*100:.0f}%"),
        (f"{SCORE['mae']:.1f} pts", FG, 'average score gap', f"typical random eligible pick: {RB['score_mae_median']:.1f} pts"),
        (f"{SCORE['w5']} / {A['n_score']}", FG, 'rows within ±5 points', f"{SCORE['w10']} / {A['n_score']} within ±10 points"),
        (f"{SCORE['spearman']:.2f}", FG, 'rank correlation', f"random eligible: {RB['spearman_median']:.2f} · close ranks can flip"),
    ])

    # ---------------- left: subset vs full scatter
    x0, y0, pw, ph = 130, 420, 620, 560           # plot box (top-left)
    lo, hi = 0, 65
    X = lambda v: x0 + pw * (v - lo) / (hi - lo); Y = lambda v: y0 + ph - ph * (v - lo) / (hi - lo)
    b += panel_head(60, 372, 'Subset score vs leaderboard score', 'Each dot is one public model / effort row (5 attempts per task)')
    band = f'M{X(lo):.1f},{Y(lo+5):.1f} L{X(hi-5):.1f},{Y(hi):.1f} L{X(hi):.1f},{Y(hi):.1f} L{X(hi):.1f},{Y(hi-5):.1f} L{X(lo+5):.1f},{Y(lo):.1f} L{X(lo):.1f},{Y(lo):.1f} Z'
    b += f'<path d="{band}" fill="{BAND}"/>'
    for v in range(0, 61, 10):
        b += L(X(v), y0, X(v), y0 + ph, GRID) + L(x0, Y(v), x0 + pw, Y(v), GRID)
        b += T(X(v), y0 + ph + 26, f'{v}%', 14, FG2, 500, 'middle') + T(x0 - 12, Y(v) + 5, f'{v}%', 14, FG2, 500, 'end')
    b += L(X(lo), Y(lo), X(hi), Y(hi), '#b3ada2', 1.5)
    b += L(x0, y0 + ph, x0 + pw, y0 + ph, '#b3ada2', 1.5) + L(x0, y0, x0, y0 + ph, '#b3ada2', 1.5)
    b += T(x0 + pw / 2, y0 + ph + 60, 'Full leaderboard score (66 tasks × 5)', 17, FG, 700, 'middle')
    b += T(x0 - 58, y0 + ph / 2, f'Subset score ({N} tasks × 5)', 17, FG, 700, 'middle',
           f'transform="rotate(-90 {x0-58} {y0+ph/2})"')
    b += T(X(4), Y(60), 'Above the line: subset flatters', 14, FG2, 600)
    b += T(X(63), Y(3), 'Below: subset is harsher', 14, FG2, 600, 'end')
    b += T(X(47.5), Y(56.5) - 2, '±5 pts', 13, '#b99a86', 700, 'middle', f'transform="rotate(-42.2 {X(47.5)} {Y(56.5)})"')
    out = sorted(SC, key=lambda r: abs(r['sub_score'] - r['full_score']))[-2:]
    pts = {}
    for r in SC:
        k = (round(r['full_score'], 2), round(r['sub_score'], 2)); pts.setdefault(k, []).append(r)
    for (fx, sy), rs in pts.items():
        hot = any(r in out for r in rs)
        b += C(X(fx), Y(sy), 8 if len(rs) == 1 else 10, ACC if hot else DOT)
        if len(rs) > 1:
            b += T(X(fx), Y(sy) + 4, len(rs), 11, '#fff', 800, 'middle')
    for r in out:
        g = r['sub_score'] - r['full_score']; up = g > 0
        tx, ty = (X(r['full_score']) - 10, Y(r['sub_score']) - 70) if up else (X(r['full_score']) + 30, Y(r['sub_score']) + 80)
        b += L(X(r['full_score']), Y(r['sub_score']), tx + (40 if up else -4), ty + (14 if up else -24), ACC, 1.5)
        anchor = 'end' if up else 'start'
        tx2 = tx + (40 if up else 0)
        b += T(tx2, ty - 8, lab(r), 17, ACC_T, 750, anchor) + T(tx2, ty + 12, f"{g:+.1f} pts", 15, ACC_T, 600, anchor)

    # ---------------- right: random-subset cloud (cost share vs score gap)
    rx0, ry0, rw, rh = 900, 420, 620, 560
    b += panel_head(840, 372, f'Why these {N} tasks? Cheap and still accurate',
                    f"vs {A['n_random']:,} random {N}-task picks from the {A['n_eligible']} tasks that meet the same constraints")
    xs_lo, xs_hi, ys_lo, ys_hi = 0.10, 0.45, 2, 13
    RX = lambda v: rx0 + rw * (v - xs_lo) / (xs_hi - xs_lo); RY = lambda v: ry0 + rh - rh * (v - ys_lo) / (ys_hi - ys_lo)
    for v in (0.1, 0.2, 0.3, 0.4):
        b += L(RX(v), ry0, RX(v), ry0 + rh, GRID) + T(RX(v), ry0 + rh + 26, f'{v*100:.0f}%', 14, FG2, 500, 'middle')
    for v in range(2, 14, 2):
        b += L(rx0, RY(v), rx0 + rw, RY(v), GRID) + T(rx0 - 12, RY(v) + 5, f'{v}', 14, FG2, 500, 'end')
    b += L(rx0, ry0 + rh, rx0 + rw, ry0 + rh, '#b3ada2', 1.5) + L(rx0, ry0, rx0, ry0 + rh, '#b3ada2', 1.5)
    b += T(rx0 + rw / 2, ry0 + rh + 60, 'Subset cost as a share of a full run (median over rows)', 17, FG, 700, 'middle')
    b += T(rx0 - 50, ry0 + rh / 2, 'Average score gap (pts)', 17, FG, 700, 'middle', f'transform="rotate(-90 {rx0-50} {ry0+rh/2})"')
    cloud = A['cloud']
    dots = ''.join(f'<circle cx="{RX(s):.1f}" cy="{RY(m):.1f}" r="2.1"/>' for s, m in cloud
                   if xs_lo <= s <= xs_hi and ys_lo <= m <= ys_hi)
    b += f'<g fill="{CLOUD}" opacity="0.38">{dots}</g>'
    ms, mm = A['cloud_median_share'], RB['score_mae_median']
    b += L(RX(ms), ry0, RX(ms), ry0 + rh, FG2, 1.2, 'stroke-dasharray="5 4"') + L(rx0, RY(mm), rx0 + rw, RY(mm), FG2, 1.2, 'stroke-dasharray="5 4"')
    b += T(RX(ms) + 8, ry0 + 20, f'typical random pick: {ms*100:.0f}% of cost', 14, FG1, 600)
    b += T(rx0 + rw - 6, RY(mm) - 8, f'typical gap: {mm:.1f} pts', 14, FG1, 600, 'end')
    fx, fy = RX(share), RY(SCORE['mae'])
    b += C(fx, fy, 11, ACC, BG, 3)
    ly = fy + 44 if fy + 80 < ry0 + rh else fy - 52
    b += T(fx + 20, ly, 'this subset', 20, ACC_T, 800)
    b += T(fx + 20, ly + 24, f"{share*100:.0f}% of cost · {SCORE['mae']:.1f} pts gap", 15, ACC_T, 600)
    n_cheap = round(A['cloud_cheaper'] * A['n_random']); n_dom = round(A['cloud_dominates'] * A['n_random'])
    bx0, bx1, byt = RX(0.215), rx0 + rw - 8, RY(12.25)
    b += f'<rect x="{bx0:.1f}" y="{byt:.1f}" width="{bx1-bx0:.1f}" height="100" rx="10" fill="{BG}" stroke="{LINE}"/>'
    b += T(bx0 + 16, byt + 30, f"Only {n_cheap} of {A['n_random']:,} random picks are this cheap,", 15, FG, 700)
    b += T(bx0 + 16, byt + 54, f"and only {n_dom} are both cheaper and closer.", 15, FG, 700)
    b += T(bx0 + 16, byt + 80, f"Typical accuracy at {A['f_share']/A['cloud_median_share']*100:.0f}% of a random pick’s cost.", 14, FG2, 500)

    ex = A['excluded_score'][0]
    b += footer([
        f"Source: public Terminal-Bench 4.0 leaderboard, every associated trial ({A['n_rows']} rows × 330 trials). {A['n_score']} rows used; {ex} excluded (trial records ≠ published successes).",
        f"Subset score = passes / {NTR} over the {N} tasks; full score = published leaderboard accuracy (the subset tasks are part of it). Held out against only the other {NREST} tasks: "
        f"gap {A['held_out']['mae']:.1f} pts, rank correlation {A['held_out']['spearman']:.2f}.",
        f"Random baseline: {A['n_random']:,} draws of {N} from the {A['n_eligible']} eligible tasks (the selection constraints: single container for HF Jobs, no GPU, no safety refusals on the leaderboard — {len(A['excluded_tasks'])} of 66 excluded).",
        f"Scored the same way. Cost share uses the {A['n_cost']} rows with complete trial costs; wall time from trial start/finish. Against all 66 tasks, a random pick costs {A_ALL['random']['share_median']*100:.0f}% and gaps {A_ALL['random']['score_mae_median']:.1f} pts.",
        f"Retrospective, not an independent validation: the subset was chosen with leaderboard data in view, and rows share models. {A['other_digest_rows']} rows (GPT-6 Astra ×5, Gemini 3.8 Flash) ran a different revision of the subset tasks (matched by name).",
    ], H - 122, f"Close enough to rank models broadly — at about {['','','a half','a third','a quarter','a fifth','a sixth','a seventh','an eighth'][round(1/A['f_share'])]} of the price.")
    page(b, 'sheet1_score')


# ====================================================================== sheet 2: cost
def sheet_cost():
    b = header(f"Budgeting a full run: multiply the subset by {COST['median_ratio']:.1f}",
               f"{A['n_cost']} public TB 4.0 rows with complete trial costs ({A['n_models_cost']} models) · subset = {N} tasks × 5 · full run = 66 tasks × 5")
    b += kpis(160, [
        (f"{COST['median_ratio']:.1f}×", ACC, 'full run ÷ subset cost', f"median · every row between {COST['min_ratio']:.1f}× and {COST['max_ratio']:.1f}×"),
        (f"{COST['mae']*100:.0f}%", FG, 'average estimate error', f"target model held out · random eligible {N}: {RB['cost_mae_median']*100:.0f}%"),
        (f"{abs(COST['naive_mean'])*100:.0f}% low", FG, 'naive trial-count scaling', f"× 330/{NTR} = {NAIVE:.1f} underestimates all {A['n_cost']} rows"),
        (f"{COST['spearman']:.3f}", FG, 'cost rank correlation', 'rows keep their cost order'),
    ])

    # ---------------- left: estimated vs reported
    x0, y0, pw, ph = 150, 420, 600, 560; hi = 8500
    X = lambda v: x0 + pw * v / hi; Y = lambda v: y0 + ph - ph * v / hi
    b += panel_head(60, 372, 'Estimated vs reported full-run cost', f'Each row estimated from its {NTR} subset trials')
    band = f'M{X(0):.1f},{Y(0):.1f} L{X(hi/1.2):.1f},{Y(hi):.1f} L{X(hi):.1f},{Y(hi):.1f} L{X(hi):.1f},{Y(hi*0.8):.1f} Z'
    b += f'<path d="{band}" fill="{BAND}"/>'
    for v in range(0, 8001, 2000):
        b += L(X(v), y0, X(v), y0 + ph, GRID) + L(x0, Y(v), x0 + pw, Y(v), GRID)
        lbl = '$0' if v == 0 else f'${v//1000}k'
        b += T(X(v), y0 + ph + 26, lbl, 14, FG2, 500, 'middle') + T(x0 - 12, Y(v) + 5, lbl, 14, FG2, 500, 'end')
    b += L(X(0), Y(0), X(hi), Y(hi), '#b3ada2', 1.5)
    b += L(x0, y0 + ph, x0 + pw, y0 + ph, '#b3ada2', 1.5) + L(x0, y0, x0, y0 + ph, '#b3ada2', 1.5)
    b += T(x0 + pw / 2, y0 + ph + 60, 'Reported full-run cost (published leaderboard total)', 17, FG, 700, 'middle')
    b += T(x0 - 66, y0 + ph / 2, 'Estimated full-run cost', 17, FG, 700, 'middle', f'transform="rotate(-90 {x0-66} {y0+ph/2})"')
    for r in sorted(CO, key=lambda r: r['full_cost']):
        b += L(X(r['full_cost']), Y(r['naive_cost']), X(r['full_cost']), Y(r['est_cost']), '#d9d4c9', 1.5)
    for r in CO:
        b += C(X(r['full_cost']), Y(r['naive_cost']), 6.5, ACC, BG, 1.5)
    for r in CO:
        b += C(X(r['full_cost']), Y(r['est_cost']), 8.5, DOT)
    b += T(X(hi) - 4, Y(hi * 0.8) + 34, '±20%', 13, '#b99a86', 700, 'end')
    b += C(x0 + 30, y0 + 22, 8.5, DOT) + T(x0 + 46, y0 + 28, f"subset × {COST['median_ratio']:.1f} (other models' median)", 15, FG, 700)
    b += C(x0 + 30, y0 + 50, 6.5, ACC, BG, 1.5) + T(x0 + 46, y0 + 56, f'subset × {NAIVE:.1f} (trial count)', 15, ACC_T, 700)
    b += T(X(5200), Y(900), 'Scaling by trial count is', 16, ACC_T, 700) + T(X(5200), Y(900) + 22, 'too low for every row', 16, ACC_T, 700)

    # ---------------- right: task map
    rx0, ry0, rw, rh = 900, 420, 620, 560
    b += panel_head(840, 372, f"Why {COST['median_ratio']:.1f}× and not {NAIVE:.1f}×? The expensive tasks aren’t in it",
                    'All 66 tasks: leaderboard pass rate vs cost relative to an average task')
    ylo, yhi = math.log10(0.07), math.log10(14)
    RX = lambda v: rx0 + rw * v; RY = lambda v: ry0 + rh - rh * (math.log10(v) - ylo) / (yhi - ylo)
    for v in (0, .25, .5, .75, 1):
        b += L(RX(v), ry0, RX(v), ry0 + rh, GRID) + T(RX(v), ry0 + rh + 26, f'{v*100:.0f}%', 14, FG2, 500, 'middle')
    for v in (0.1, 0.25, 0.5, 1, 2, 4):
        b += L(rx0, RY(v), rx0 + rw, RY(v), GRID) + T(rx0 - 12, RY(v) + 5, f'{v:g}×', 14, FG2, 500, 'end')
    b += L(rx0, RY(1), rx0 + rw, RY(1), '#b3ada2', 1.5, 'stroke-dasharray="6 4"')
    b += T(rx0 + rw - 6, RY(1) - 8, 'average task', 13, FG1, 600, 'end')
    b += L(rx0, ry0 + rh, rx0 + rw, ry0 + rh, '#b3ada2', 1.5) + L(rx0, ry0, rx0, ry0 + rh, '#b3ada2', 1.5)
    b += T(rx0 + rw / 2, ry0 + rh + 60, 'Pass rate across leaderboard rows  (hard → easy)', 17, FG, 700, 'middle')
    b += T(rx0 - 58, ry0 + rh / 2, 'Task cost vs average task (log)', 17, FG, 700, 'middle', f'transform="rotate(-90 {rx0-58} {ry0+rh/2})"')
    tasks = A['tasks']
    for t in tasks:
        if not t['subset'] and t['task'] in ELIG:
            b += C(RX(t['pass_rate']), RY(max(t['rel_cost'], 0.075)), 6.5, '#cdc8bc', BG, 1.5)
        elif not t['subset']:
            b += C(RX(t['pass_rate']), RY(max(t['rel_cost'], 0.075)), 5.5, BG, '#9d978a', 1.8)
    for t in tasks:
        if t['subset']:
            b += C(RX(t['pass_rate']), RY(max(t['rel_cost'], 0.075)), 8, ACC, BG, 2)
    sub = [t for t in tasks if t['subset']]; oth = [t for t in tasks if not t['subset']]
    rel = A['f_share'] * A['n_tasks'] / len(A['subset'])   # subset task cost vs average task (median row)
    exp = sorted(oth, key=lambda t: -t['rel_cost'])[:1][0]
    b += T(RX(exp['pass_rate']), RY(exp['rel_cost']) + 26, f"{exp['task']} · {exp['rel_cost']:.1f}×", 13, FG2, 600, 'middle')
    lx = rx0 + 24
    ne = A['n_eligible'] - len(A['subset']); nx = A['n_tasks'] - A['n_eligible']
    b += C(lx, ry0 + 22, 8, ACC) + T(lx + 16, ry0 + 28, f"subset ({len(A['subset'])})", 15, ACC_T, 700)
    b += C(lx + 120, ry0 + 22, 6.5, '#cdc8bc') + T(lx + 136, ry0 + 28, f'other eligible ({ne})', 15, FG1, 700)
    b += C(lx + 310, ry0 + 22, 5.5, BG, '#9d978a', 1.8) + T(lx + 326, ry0 + 28, f'GPU / multi-container / refusal ({nx})', 13, FG2, 600)
    bx, by = RX(0.40), ry0 + 150
    b += f'<rect x="{bx:.1f}" y="{by-92:.1f}" width="{rx0+rw-bx-10:.1f}" height="104" rx="10" fill="{BG}" stroke="{LINE}"/>'
    xm = st.mean(t['rel_cost_mean'] for t in tasks if t['task'] not in ELIG)
    b += T(bx + 16, by - 62, f'Excluded tasks average {xm:.1f}× an average task;', 15, FG, 700)
    b += T(bx + 16, by - 38, f'the subset spans hard → easy at {rel:.2f}×.', 15, FG, 700)
    b += T(bx + 16, by - 12, f"So: 330/{NTR} ÷ {rel:.2f} ≈ {NAIVE/rel:.1f}× — the multiplier.", 15, ACC_T, 700)

    b += footer([
        f"Source: public Terminal-Bench 4.0 leaderboard trial cost records. {A['n_cost']} of {A['n_rows']} rows used: all 330 trial costs present and summing to the published total. "
        f"Excluded: {len(A['excluded_cost_null'])} rows with missing trial costs, {len(A['excluded_cost_mismatch'])} whose trials don't sum to the published total.",
        f"Estimate = subset cost × median(full ÷ subset) over rows of other models (every effort of the target model held out). Error = |estimate − reported| ÷ reported, averaged over rows. "
        f"Pooled ratio {COST['pooled_ratio']:.2f}×.",
        f"Random {N}-task picks from the {A['n_eligible']} eligible tasks, same method, average {RB['cost_mae_median']*100:.0f}% error (this subset: {COST['mae']*100:.0f}%; beats {100*(1-RB['cost_mae_share_better']):.0f}%). Task cost = median over rows of the task's 5-trial cost ÷ the row's average task cost.",
        "Retrospective and model-mix dependent: the multiplier reflects these agents and models. Reported leaderboard costs, not invoices; sandbox compute excluded.",
    ], H - 108, f"Budget a full run at ≈{COST['median_ratio']:.1f}× the subset cost; expect about ±10%, occasionally ±25%.")
    page(b, 'sheet2_cost')


if __name__ == '__main__':
    sheet_score(); sheet_cost()
