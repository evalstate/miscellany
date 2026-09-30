"""Plain-language information sheets for the TB4 subset ("here's our subset, here's why it's not terrible").

    python3 gen2.py     # -> subset_sheet.png, subset_cost.png (+ html/)
"""
import json, math, statistics as st, tomllib
import gen
from gen import T, L, C, ROOT, BG, FG, FG1, FG2, LINE, GRID, DOT, ACC, ACC_T, BAND

A = gen.A; N = gen.N; NTR = gen.NTR; NF = gen.NF; NAIVE = gen.NAIVE
SC, CO, SCORE, COST, RB = gen.SC, gen.CO, gen.SCORE, gen.COST, gen.RB
TILE = '#f3f1ea'; PALE = '#e9e5db'; SOFT = '#9d978a'


def short(r):
    m = r['model'].replace('Gemini ', 'Gemini ').replace('GPT-', 'GPT-')
    return f"{m} · {r['effort']}"


def kpis(y, items, h=150):
    n = len(items); gap = 28; w = (gen.W - 120 - gap * (n - 1)) / n; s = ''
    for i, (big, col, lab, note) in enumerate(items):
        x = 60 + i * (w + gap)
        s += f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="14" fill="{TILE}"/>'
        s += T(x + 26, y + 68, big, 52, col, 800) + T(x + 26, y + 104, lab, 18, FG, 700) + T(x + 26, y + 130, note, 15, FG2, 500)
    return s


def section(x, y, title, sub):
    return T(x, y, title, 23, FG, 800) + T(x, y + 26, sub, 15, FG2, 500)


def footer(y0, takeaway, lines):
    s = L(60, y0, gen.W - 60, y0) + T(60, y0 + 44, takeaway, 24, FG, 800)
    for i, l in enumerate(lines):
        s += T(60, y0 + 80 + i * 20, l, 12.5, FG2)
    return s


# =================================================================== sheet 1: the subset
def sheet_subset():
    gen.W, gen.H = W, H = 1600, 1414
    tasks = sorted([t for t in A['tasks'] if t['subset']], key=lambda t: t['pass_rate'])
    meta = {t: tomllib.load(open(ROOT / 'data' / 'tasks' / 'terminal-bench' / t / 'task.toml', 'rb'))['metadata'] for t in A['subset']}
    domains = sorted({meta[t]['category'] for t in A['subset']})
    b = T(60, 60, 'TERMINAL-BENCH 4.0 · TASK SELECTION', 15, ACC_T, 800, extra='letter-spacing="1.5"')
    b += T(60, 104, f'A {N}-task subset of Terminal-Bench 4.0', 42, FG, 800)
    b += T(60, 142, f'Made for harness testing: {NTR} trials instead of 330, about {A["f_share"]*100:.0f}% of the cost. {SCORE["w5"]} of {A["n_score"]} leaderboard entries land within 5 points of their full score.', 19, FG1)

    # ---------------- left: the tasks
    y0 = 228
    b += section(60, y0, f'The {N} tasks', f'Sorted hard → easy by pass rate across all {A["n_score"]} leaderboard entries')
    hy = y0 + 64
    bx0, bw = 480, 170
    b += T(60, hy, 'TASK', 12, FG2, 700, extra='letter-spacing="1.2"') + T(318, hy, 'DOMAIN', 12, FG2, 700, extra='letter-spacing="1.2"')
    b += T(bx0, hy, 'LEADERBOARD PASS RATE', 12, FG2, 700, extra='letter-spacing="1.2"')
    b += T(740, hy, 'COST', 12, FG2, 700, 'end', extra='letter-spacing="1.2"')
    rh = 36; ty = hy + 14
    for i, t in enumerate(tasks):
        y = ty + i * rh
        if i % 2 == 0:
            b += f'<rect x="50" y="{y:.1f}" width="700" height="{rh}" rx="6" fill="#f4f2ec"/>'
        cy = y + rh / 2 + 5
        b += T(62, cy, t['task'], 15.5, FG, 650)
        b += T(318, cy, meta[t['task']]['category'], 14, FG1, 500)
        b += f'<rect x="{bx0}" y="{cy-11:.1f}" width="{bw}" height="12" rx="6" fill="{PALE}"/>'
        if t['pass_rate'] > 0.004:
            b += f'<rect x="{bx0}" y="{cy-11:.1f}" width="{max(bw*t["pass_rate"], 6):.1f}" height="12" rx="6" fill="{ACC}"/>'
        b += T(bx0 + bw + 10, cy, f"{t['pass_rate']*100:.0f}%", 14, FG1, 600)
        b += T(740, cy, f"{t['rel_cost']:.1f}×", 14, FG1, 600, 'end')
    ly = ty + N * rh + 22
    b += T(60, ly, 'Cost = typical cost of the task relative to an average Terminal-Bench 4.0 task.', 13, FG2)

    # how picked
    py = ly + 30; ph = 176
    b += f'<rect x="50" y="{py}" width="700" height="{ph}" rx="14" fill="{TILE}"/>'
    b += T(74, py + 36, 'How the tasks were picked', 19, FG, 800)
    bullets = ['Runs on Hugging Face Jobs: single container, fits 8 vCPU / 32 GB',
               'No GPU tasks, and no tasks that drew safety refusals on the leaderboard',
               'Dropped candidates with known faulty instructions or forged-result bugs',
               f'Spread from hard to easy, across {len(domains)} domains']
    for i, s_ in enumerate(bullets):
        b += C(84, py + 67 + i * 30, 3.5, ACC, ACC, 0) + T(98, py + 72 + i * 30, s_, 15, FG1, 500)

    # ---------------- right: every leaderboard entry
    rx = 820
    b += section(rx, y0, 'Subset score vs full score', f'Every public leaderboard entry ({A["n_score"]} model / effort combinations)')
    lab_r, ax0, ax1, gx = 1060, 1080, 1440, 1540
    lo, hi = 0, 65
    X = lambda v: ax0 + (ax1 - ax0) * (v - lo) / (hi - lo)
    ly2 = y0 + 62
    b += C(rx + 8, ly2 - 5, 6.5, BG, DOT, 2.5) + T(rx + 22, ly2, f'full benchmark ({NF} tasks)', 14, FG1, 600)
    b += C(rx + 228, ly2 - 5, 7, ACC, BG, 1.5) + T(rx + 242, ly2, f'subset ({N} tasks)', 14, ACC_T, 700)
    b += T(gx, ly2, 'GAP', 12, FG2, 700, 'end', extra='letter-spacing="1.2"')
    rows = sorted(SC, key=lambda r: (-r['full_score'], r['model']))
    rh2 = 30.2; top = ly2 + 22
    bottom = top + len(rows) * rh2
    for v in range(0, 61, 10):
        b += L(X(v), top - 4, X(v), bottom, GRID)
        b += T(X(v), bottom + 20, f'{v}%', 13, FG2, 500, 'middle')
    for i, r in enumerate(rows):
        cy = top + i * rh2 + rh2 / 2
        g = r['sub_score'] - r['full_score']; far = abs(g) > 5
        b += T(lab_r, cy + 5, short(r), 14, FG if far else FG1, 700 if far else 500, 'end')
        x1, x2 = X(r['full_score']), X(r['sub_score'])
        b += L(min(x1, x2), cy, max(x1, x2), cy, ACC if far else '#d9b8a6', 3)
        b += C(x1, cy, 6.5, BG, DOT, 2.5) + C(x2, cy, 7, ACC, BG, 1.5)
        b += T(gx, cy + 5, f"{g:+.1f}", 14, ACC_T if far else FG2, 700 if far else 500, 'end')
    b += T((ax0 + ax1) / 2, bottom + 46, 'Score (share of trials passed)', 15, FG, 700, 'middle')
    nfar = sum(abs(r['sub_score'] - r['full_score']) > 5 for r in rows)
    b += T(rx, bottom + 80, f"{len(rows) - nfar} of {len(rows)} entries land within 5 points; the gap column is highlighted where they don't.", 14, FG1, 500)

    b += footer(H - 186, 'Good enough to compare harnesses and catch regressions — not a substitute for a full leaderboard run.', [
        f"Source: public Terminal-Bench 4.0 leaderboard, every trial of every entry (5 attempts per task). {A['n_score']} of {A['n_rows']} entries used; Opus 5 / max excluded (trial records don't match its published score).",
        f"The subset is scored from those same trials. Compared only with the other {NF - N} tasks (no shared trials), the average gap is {A['held_out']['mae']:.1f} points and rank agreement {A['held_out']['spearman']:.2f}.",
        f"Against {A['n_random']:,} random {N}-task picks from the {A['n_eligible']} tasks meeting the same constraints, this subset is closer than {100*(1-RB['score_mae_share_better']):.0f}% and cheaper than {100*(1-A['cloud_cheaper']):.0f}% of them.",
        "Caveats: chosen with this leaderboard in view, so expect it to be a little less accurate for new models. Pass rates and costs come from the leaderboard's agents and models.",
        f"Several subset tasks have open verifier-hardening issues in the Terminal-Bench v4.1 milestone; pinned to the v4.0 task digests the leaderboard ran.",
    ])
    gen.page(b, 'subset_sheet')


# =================================================================== sheet 2: cost
def sheet_cost():
    gen.W, gen.H = W, H = 1600, 1214
    k = COST['median_ratio']
    b = T(60, 60, 'TERMINAL-BENCH 4.0 · TASK SELECTION', 15, ACC_T, 800, extra='letter-spacing="1.5"')
    b += T(60, 104, f'What will a full run cost? About {k:.1f}× the subset', 42, FG, 800)
    b += T(60, 142, f"Checked on the {A['n_cost']} leaderboard entries with complete cost records ({A['n_models_cost']} models).", 19, FG1)
    b += kpis(176, [
        (f"{k:.1f}×", ACC, 'full-run cost ÷ subset cost', f"every entry between {COST['min_ratio']:.1f}× and {COST['max_ratio']:.1f}×"),
        (f"±{COST['mae']*100:.0f}%", FG, 'typical error for a new model', f"model left out when fitting · worst {COST['worst']*100:.0f}%"),
        (f"{NAIVE:.1f}× is too low", FG, f'scaling by trial count (330 ÷ {NTR})', f"undershoots every entry, by {abs(COST['naive_mean'])*100:.0f}% on average"),
    ])

    # ---------------- left: estimate vs actual per entry
    y0 = 414; rx0 = 60
    b += section(rx0, y0, 'Estimated vs actual full-run cost', f'Subset cost × {k:.1f}, where the multiplier comes from the other models')
    ly2 = y0 + 62
    b += C(rx0 + 8, ly2 - 5, 7, DOT, BG, 1.5) + T(rx0 + 22, ly2, 'actual (leaderboard)', 14, FG1, 600)
    b += C(rx0 + 200, ly2 - 5, 7, ACC, BG, 1.5) + T(rx0 + 214, ly2, f'estimate (× {k:.1f})', 14, ACC_T, 700)
    b += C(rx0 + 380, ly2 - 5, 5, BG, SOFT, 1.8) + T(rx0 + 392, ly2, f'trial-count scaling (× {NAIVE:.1f})', 14, FG2, 600)
    lab_r, ax0, ax1 = 290, 310, 740; hi = 8500
    X = lambda v: ax0 + (ax1 - ax0) * v / hi
    rows = sorted(CO, key=lambda r: -r['full_cost'])
    rh = 33; top = ly2 + 24; bottom = top + len(rows) * rh
    for v in range(0, 8001, 2000):
        b += L(X(v), top - 4, X(v), bottom, GRID) + T(X(v), bottom + 20, '$0' if v == 0 else f'${v // 1000}k', 13, FG2, 500, 'middle')
    for i, r in enumerate(rows):
        cy = top + i * rh + rh / 2
        b += T(lab_r, cy + 5, short(r), 14, FG1, 500, 'end')
        xa, xe, xn = X(r['full_cost']), X(r['est_cost']), X(r['naive_cost'])
        b += L(xn, cy, xa, cy, PALE, 2) + L(min(xa, xe), cy, max(xa, xe), cy, '#d9b8a6', 3)
        b += C(xn, cy, 5, BG, SOFT, 1.8) + C(xa, cy, 7, DOT, BG, 1.5) + C(xe, cy, 7, ACC, BG, 1.5)
    b += T((ax0 + ax1) / 2, bottom + 46, 'Full-run cost (reported by the leaderboard)', 15, FG, 700, 'middle')

    # ---------------- right: why the multiplier is 6.5x
    rx = 840
    b += section(rx, y0, f'Why {k:.1f}× and not {NAIVE:.1f}×?', 'The subset avoids the expensive tasks')
    tp = {t['task']: t for t in A['tasks']}; elig = set(A['eligible'])
    bars = [('All 66 tasks', 1.0, DOT),
            ('GPU, multi-container and refusal tasks (excluded)', st.mean(t['rel_cost_mean'] for t in A['tasks'] if t['task'] not in elig), SOFT),
            (f"Tasks meeting the constraints ({A['n_eligible']})", st.mean(tp[t]['rel_cost_mean'] for t in elig), '#8c8f97'),
            (f'This subset ({N})', A['f_share'] * NF / N, ACC)]
    bx0, bw, by = rx, 560, y0 + 80
    b += T(rx, by - 8, 'Average cost of a task, relative to an average Terminal-Bench 4.0 task', 14, FG2, 600)
    for i, (lab, v, col) in enumerate(bars):
        y = by + 14 + i * 74
        b += T(bx0, y + 16, lab, 16, FG, 700)
        b += f'<rect x="{bx0}" y="{y + 28}" width="{bw * v / 1.5:.1f}" height="24" rx="6" fill="{col}"/>'
        b += T(bx0 + bw * v / 1.5 + 12, y + 47, f'{v:.2f}×', 17, ACC_T if col == ACC else FG, 800)
    b += L(bx0 + bw / 1.5, by + 30, bx0 + bw / 1.5, by + 14 + 4 * 74, FG2, 1.2, 'stroke-dasharray="4 4"')
    ey = by + 14 + 4 * 74 + 30
    b += f'<rect x="{rx}" y="{ey}" width="{gen.W - 60 - rx}" height="150" rx="14" fill="{TILE}"/>'
    b += T(rx + 24, ey + 38, 'The arithmetic', 18, FG, 800)
    b += T(rx + 24, ey + 72, f'330 ÷ {NTR} trials = {NAIVE:.2f}×', 17, FG1, 600)
    b += T(rx + 24, ey + 102, f'…but each subset trial costs {A["f_share"] * NF / N:.2f}× an average trial', 17, FG1, 600)
    b += T(rx + 24, ey + 132, f'{NAIVE:.2f} ÷ {A["f_share"] * NF / N:.2f} ≈ {NAIVE / (A["f_share"] * NF / N):.1f}× — the multiplier', 17, ACC_T, 800)

    b += footer(H - 150, f'Budget a full run at about {k:.1f}× what the subset cost you; expect ±10%, occasionally ±25%.', [
        f"Source: public Terminal-Bench 4.0 leaderboard trial costs. {A['n_cost']} of {A['n_rows']} entries used: all 330 trial costs present and summing to the published total ({len(A['excluded_cost_null'])} excluded for missing trial costs, {len(A['excluded_cost_mismatch'])} for totals that don't reconcile).",
        "Estimate for each entry uses the median full ÷ subset ratio of the other models (all efforts of the entry's own model left out). Reported leaderboard costs, not invoices; sandbox compute excluded.",
        "The multiplier reflects these agents and models; a harness that spends very differently on cheap vs expensive tasks will shift it.",
    ])
    gen.page(b, 'subset_cost')


# =================================================================== sheet 3: the diagonal
def sheet_scatter():
    gen.W, gen.H = W, H = 1600, 1170
    b = T(60, 60, 'TERMINAL-BENCH 4.0 · TASK SELECTION CHECK', 15, ACC_T, 800, extra='letter-spacing="1.5"')
    b += T(60, 104, f'Does our {N}-task subset track the full benchmark?', 42, FG, 800)
    b += T(60, 142, f"Not a harness result: each dot is an existing public leaderboard entry, re-scored on just our {N} tasks. {SCORE['w5']} of {A['n_score']} land within ±5 points.", 19, FG1)
    x0, y0, pw = 150, 200, 780
    lo, hi = 0, 65
    X = lambda v: x0 + pw * (v - lo) / (hi - lo); Y = lambda v: y0 + pw - pw * (v - lo) / (hi - lo)
    band = (f'M{X(lo):.1f},{Y(lo+5):.1f} L{X(hi-5):.1f},{Y(hi):.1f} L{X(hi):.1f},{Y(hi):.1f} L{X(hi):.1f},{Y(hi-5):.1f} '
            f'L{X(lo+5):.1f},{Y(lo):.1f} L{X(lo):.1f},{Y(lo):.1f} Z')
    b += f'<path d="{band}" fill="{BAND}"/>'
    for v in range(0, 61, 10):
        b += L(X(v), y0, X(v), y0 + pw, GRID) + L(x0, Y(v), x0 + pw, Y(v), GRID)
        b += T(X(v), y0 + pw + 28, f'{v}%', 15, FG2, 500, 'middle') + T(x0 - 14, Y(v) + 5, f'{v}%', 15, FG2, 500, 'end')
    b += L(X(lo), Y(lo), X(hi), Y(hi), '#a9a398', 1.8)
    b += L(x0, y0 + pw, x0 + pw, y0 + pw, '#b3ada2', 1.5) + L(x0, y0, x0, y0 + pw, '#b3ada2', 1.5)
    b += T(x0 + pw / 2, y0 + pw + 66, f'Full leaderboard score ({NF} tasks × 5 attempts)', 18, FG, 700, 'middle')
    b += T(x0 - 70, y0 + pw / 2, f'Subset score ({N} tasks × 5 attempts)', 18, FG, 700, 'middle', f'transform="rotate(-90 {x0-70} {y0+pw/2})"')
    b += T(X(2.5), Y(61), 'above the line: subset scores higher', 15, FG2, 600)
    b += T(X(63), Y(2.5), 'below: subset scores lower', 15, FG2, 600, 'end')
    b += T(X(49), Y(58.6), '±5 points', 14, '#b99a86', 700, 'middle', f'transform="rotate(-45 {X(49):.1f} {Y(58.6):.1f})"')
    groups = {}
    for r in SC:
        groups.setdefault((round(r['full_score'], 1), round(r['sub_score'], 1)), []).append(r)
    far = []
    for (fx, sy), rs in sorted(groups.items()):
        out = abs(sy - fx) > 5
        if out: far += rs
        b += C(X(fx), Y(sy), 10 if len(rs) == 1 else 12, ACC if out else DOT, BG, 2.5)
        if len(rs) > 1: b += T(X(fx), Y(sy) + 4.5, len(rs), 12, '#fff', 800, 'middle')
    # outlier labels: place to the left (above line) or right (below line)
    for r in sorted(far, key=lambda r: r['full_score']):
        g = r['sub_score'] - r['full_score']; px, py = X(r['full_score']), Y(r['sub_score'])
        if r['model'] == 'GPT-5.6 Luna':
            tx, ty, anc = px - 14, py - 40, 'end'
        elif g > 0:
            tx, ty, anc = px - 22, py + 22, 'end'
        else:
            tx, ty, anc = px + 22, py + 12, 'start'
        b += T(tx, ty, short(r), 15, ACC_T, 750, anc) + T(tx, ty + 19, f'{g:+.1f} pts', 14, ACC_T, 600, anc)

    # ---------------- right: the tasks and their leaderboard pass rates
    rx = 1030; ry = 200
    tasks = sorted([t for t in A['tasks'] if t['subset']], key=lambda t: t['pass_rate'])
    b += T(rx, ry + 8, f'The {N} tasks', 22, FG, 800)
    b += T(rx, ry + 34, 'Pass rate across all leaderboard entries, hard → easy', 14, FG2, 500)
    bx0, bw = rx + 318, 140
    rh = (y0 + pw - (ry + 66)) / N
    for i, t in enumerate(tasks):
        y = ry + 66 + i * rh
        if i % 2 == 0:
            b += f'<rect x="{rx - 10}" y="{y:.1f}" width="{W - 60 - rx + 10}" height="{rh:.1f}" rx="6" fill="#f4f2ec"/>'
        cy = y + rh / 2 + 5
        b += T(rx, cy, t['task'], 15, FG, 650)
        b += f'<rect x="{bx0}" y="{cy - 10:.1f}" width="{bw}" height="11" rx="5.5" fill="{PALE}"/>'
        if t['pass_rate'] > 0.004:
            b += f'<rect x="{bx0}" y="{cy - 10:.1f}" width="{max(bw * t["pass_rate"], 5.5):.1f}" height="11" rx="5.5" fill="{ACC}"/>'
        b += T(W - 60, cy, f"{t['pass_rate'] * 100:.0f}%", 14, FG1, 600, 'end')
    b += T(rx, y0 + pw + 30, 'Treat gaps under ~5 points between two harnesses', 15, ACC_T, 700)
    b += T(rx, y0 + pw + 52, f'as noise on {N} tasks.', 15, ACC_T, 700)

    b += L(60, H - 96, W - 60, H - 96)
    b += T(60, H - 64, f"Source: public Terminal-Bench 4.0 leaderboard, all trials of {A['n_score']} entries ({A['n_models_score']} models; Opus 5 / max excluded — trial records don't match its published score). "
           f"The subset is scored from the same trials.", 12.5, FG2)
    b += T(60, H - 42, f"Average gap {SCORE['mae']:.1f} points, rank agreement {SCORE['spearman']:.2f}; against only the other {NF - N} tasks (no shared trials): {A['held_out']['mae']:.1f} points, {A['held_out']['spearman']:.2f}. "
           "The ±5-point band is a visual guide, not a confidence interval. Chosen with this leaderboard in view.", 12.5, FG2)
    gen.page(b, 'subset_scatter')

# =================================================================== sheet 4: per-model deep dive
EFF = ['max', 'xhigh', 'high', 'medium', 'low']


def heat(p):
    """passes out of 5 -> fill colour (pale sand -> accent)."""
    if p == 0:
        return '#f1eee7'
    c0, c1 = (0xf3, 0xdc, 0xcf), (0xb8, 0x4a, 0x22)
    f = (p - 1) / 4
    return '#%02x%02x%02x' % tuple(round(a + (b_ - a) * (0.25 + 0.75 * f)) for a, b_ in zip(c0, c1))


def sheet_models():
    gen.W, gen.H = W, H = 1600, 1560
    S_ = A['subset']
    tasks = sorted([t for t in A['tasks'] if t['subset']], key=lambda t: t['pass_rate'])
    seen = json.load(open(ROOT / 'data' / 'subset_digests_seen.json'))
    other_rev = {rid for rid, n, dg in seen if n in gen.json.load(open(ROOT / 'subset.json')) and dg != json.load(open(ROOT / 'subset.json'))[n]}
    fams = {}
    for r in SC: fams.setdefault(r['model'], []).append(r)
    order = sorted(fams, key=lambda m: -max(r['full_score'] for r in fams[m]))
    for m in order: fams[m].sort(key=lambda r: EFF.index(r['effort']))

    def peers(r):
        return [o for o in SC if o['model'] != r['model'] and abs(o['full_score'] - r['full_score']) <= 8]

    b = T(60, 60, 'TERMINAL-BENCH 4.0 · TASK SELECTION CHECK', 15, ACC_T, 800, extra='letter-spacing="1.5"')
    b += T(60, 104, f'How each leaderboard entry does on the {N} tasks', 42, FG, 800)
    b += T(60, 142, 'Passes out of 5 attempts, for every public entry on every subset task. Existing leaderboard trials, not a new harness run.', 19, FG1)

    lab_r, gx0, cw, ch, gap = 290, 310, 40, 27, 3
    top = 372
    # column headers (rotated task names) + difficulty strip
    for j, t in enumerate(tasks):
        cx = gx0 + j * cw + cw / 2
        b += T(cx + 4, top - 44, t['task'], 13.5, FG1, 600, 'start', f'transform="rotate(-52 {cx + 4:.1f} {top - 44})"')
        b += f'<rect x="{gx0 + j * cw + 3}" y="{top - 30}" width="{cw - 6}" height="6" rx="3" fill="{PALE}"/>'
        b += f'<rect x="{gx0 + j * cw + 3}" y="{top - 30}" width="{max((cw - 6) * t["pass_rate"], 2):.1f}" height="6" rx="3" fill="{SOFT}"/>'
    b += T(gx0 - 12, top - 23, 'leaderboard pass rate', 11.5, FG2, 600, 'end')
    b += T(gx0, top - 8, 'hard →', 12, FG2, 600) + T(gx0 + N * cw, top - 8, '→ easy', 12, FG2, 600, 'end')
    # right-hand columns
    fx, sx, gxx, dx0, dx1 = 1150, 1230, 1300, 1330, 1540
    D = lambda v: dx0 + (dx1 - dx0) * v / 65
    b += T(fx, top - 8, 'FULL', 12, FG2, 700, 'end', 'letter-spacing="1.2"') + T(sx, top - 8, 'SUBSET', 12, FG2, 700, 'end', 'letter-spacing="1.2"')
    b += T(gxx, top - 8, 'GAP', 12, FG2, 700, 'end', 'letter-spacing="1.2"')
    b += C(dx0 + 6, top - 12, 5.5, BG, DOT, 2) + T(dx0 + 16, top - 8, 'full', 12, FG2, 600)
    b += C(dx0 + 62, top - 12, 6, ACC, BG, 1) + T(dx0 + 72, top - 8, 'subset', 12, ACC_T, 700)
    y = top
    for fi, m in enumerate(order):
        if fi: y += 10
        for r in fams[m]:
            P = peers(r)
            cy = y + ch / 2
            g_ = r['sub_score'] - r['full_score']; far = abs(g_) > 5
            dag = ' †' if r['id'] in other_rev else ''
            b += T(lab_r, cy + 5, f"{r['model']} · {r['effort']}{dag}", 14, FG if far else FG1, 700 if far else 500, 'end')
            for j, t in enumerate(tasks):
                p = r['per_task'][t['task']]['p']
                x = gx0 + j * cw
                b += f'<rect x="{x + gap / 2}" y="{y + gap / 2}" width="{cw - gap}" height="{ch - gap}" rx="4" fill="{heat(p)}"/>'
                if p: b += T(x + cw / 2, cy + 4.5, p, 12.5, '#fff' if p >= 3 else ACC_T, 700, 'middle')
                if P:
                    d = p / 5 - st.mean(o['per_task'][t['task']]['p'] / 5 for o in P)
                    if d >= 0.6:
                        b += f'<rect x="{x + gap / 2 + 1}" y="{y + gap / 2 + 1}" width="{cw - gap - 2}" height="{ch - gap - 2}" rx="4" fill="none" stroke="{FG}" stroke-width="2"/>'
                    elif d <= -0.6:
                        b += f'<rect x="{x + gap / 2 + 1}" y="{y + gap / 2 + 1}" width="{cw - gap - 2}" height="{ch - gap - 2}" rx="4" fill="none" stroke="{FG}" stroke-width="1.6" stroke-dasharray="3 2.5"/>'
            b += T(fx, cy + 5, f"{r['full_score']:.0f}%", 14, FG1, 600, 'end') + T(sx, cy + 5, f"{r['sub_score']:.0f}%", 14, ACC_T, 700, 'end')
            b += T(gxx, cy + 5, f"{g_:+.1f}", 14, ACC_T if far else FG2, 700 if far else 500, 'end')
            b += L(dx0, cy, dx1, cy, '#efece5', 1)
            x1, x2 = D(r['full_score']), D(r['sub_score'])
            b += L(min(x1, x2), cy, max(x1, x2), cy, ACC if far else '#d9b8a6', 2.5)
            b += C(x1, cy, 5.5, BG, DOT, 2) + C(x2, cy, 6, ACC, BG, 1)
            y += ch
    for v in (0, 20, 40, 60):
        b += T(D(v), y + 18, f'{v}%', 11.5, FG2, 500, 'middle')

    # legend
    ly = y + 46
    b += T(60, ly, 'Passes of 5:', 13.5, FG1, 700)
    for p in range(6):
        b += f'<rect x="{150 + p * 34}" y="{ly - 16}" width="30" height="22" rx="4" fill="{heat(p)}"/>'
        b += T(165 + p * 34, ly, p, 12, '#fff' if p >= 3 else ACC_T, 700, 'middle')
    b += f'<rect x="380" y="{ly - 16}" width="30" height="22" rx="4" fill="{PALE}" stroke="{FG}" stroke-width="2"/>'
    b += T(418, ly, 'much better than entries with a similar full score (≥3 more passes of 5)', 13, FG1, 500)
    b += f'<rect x="920" y="{ly - 16}" width="30" height="22" rx="4" fill="{PALE}" stroke="{FG}" stroke-width="1.6" stroke-dasharray="3 2.5"/>'
    b += T(958, ly, 'much worse', 13, FG1, 500)
    b += T(1060, ly, '† ran a different revision of the subset tasks', 13, FG1, 500)

    # what to notice
    ny = ly + 34; nh = 196; nw = (W - 120 - 2 * 24) / 3
    def famtask(m, t):
        rs = fams[m]; return sum(r['per_task'][t]['p'] for r in rs), 5 * len(rs)
    a1, n1 = famtask('GPT-6 Astra', 'wal-recovery-ordering'); a2, n2 = famtask('GPT-6 Astra', 'gsea-proteomics')
    a3, n3 = famtask('Fable 5.1', 'gsea-proteomics'); a4, n4 = famtask('Fable 5.1', 'wal-recovery-ordering')
    luna = next(r for r in SC if r['model'] == 'GPT-5.6 Luna')
    fl = next(r for r in SC if r['model'] == 'Fable 5.1' and r['effort'] == 'low')
    fm = next(r for r in SC if r['model'] == 'Fable 5.1' and r['effort'] == 'medium')
    notes = [
        ('Model families have signature tasks', [
            f'GPT-6 Astra passes wal-recovery-ordering {a1}/{n1}',
            f'times but gsea-proteomics {a2}/{n2}; Fable 5.1 is',
            f'the mirror image ({a3}/{n3} and {a4}/{n4}). A mix of',
            'both kinds keeps any one family from being', 'favoured — but see † on Astra.']),
        ('Big gaps come from a few tasks', [
            f"GPT-5.6 Luna's +{luna['sub_score'] - luna['full_score']:.0f} is mostly two tasks it aces",
            f"(atrx-vep-crispr {luna['per_task']['atrx-vep-crispr']['p']}/5, photonic-waveguide-routing",
            f"{luna['per_task']['photonic-waveguide-routing']['p']}/5) where similar entries rarely pass.",
            'With 19 tasks, one or two surprises move', 'a score by 5–10 points.']),
        ('Big effort steps survive, small ones don\u2019t', [
            f"Fable 5.1 low → medium: +{fm['full_score'] - fl['full_score']:.0f} pts on the full benchmark,",
            f"+{fm['sub_score'] - fl['sub_score']:.0f} on the subset. Differences under ~4 points",
            '(e.g. high vs xhigh) can reverse on 19 tasks.', 'Compare harnesses, not near-identical', 'configurations.']),
    ]
    for i, (hd, ls_) in enumerate(notes):
        x = 60 + i * (nw + 24)
        b += f'<rect x="{x:.1f}" y="{ny}" width="{nw:.1f}" height="{nh}" rx="14" fill="{TILE}"/>'
        b += T(x + 22, ny + 36, hd, 17.5, FG, 800)
        for k_, l_ in enumerate(ls_):
            b += T(x + 22, ny + 66 + k_ * 23, l_, 14.5, FG1, 500)

    b += L(60, H - 70, W - 60, H - 70)
    b += T(60, H - 42, f"Source: public Terminal-Bench 4.0 leaderboard trials ({A['n_score']} entries, 5 attempts per task; Opus 5 / max excluded). Similar entries = other models within ±8 points "
           "on the full benchmark. Rows grouped by model, highest effort first.", 12.5, FG2)
    gen.page(b, 'subset_models')


if __name__ == '__main__':
    sheet_subset(); sheet_cost(); sheet_scatter(); sheet_models()
