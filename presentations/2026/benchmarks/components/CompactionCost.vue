<script setup lang="ts">
// Pareto view of one change: move the compaction point earlier and each
// model's run moves up and to the left, towards "better" (higher score,
// lower cost). Grey is before, amber is after (the change is a fast-agent
// setting), teal marks the improvement. The earlier bar-based version of
// this slide is CompactionBars.vue.
//
// 19-task TB 4 subset, 95 trials per run. Figures are copied from the
// fast-agent docs chart (docs/benchmark_data/tb4/compaction.py, 4 Nov 2026):
// score is rewards ÷ 95, cost is fast-agent's recorded cost at list rates.
//
// Axes: cost is zoomed to each pair ($0.15–$0.30, $5–$8) so the points sit
// apart; score shows 30% to 50%. Both are windows, and the ticks say so.
type Run = { point: string; score: number; cost: number };
const TRIALS = 95;
const pairs: { model: string; late: Run; early: Run; ticks: number[] }[] = [
  { model: "GLM-5.3 Flash", late: { point: "1M", score: 35.8, cost: 26.15 }, early: { point: "200k", score: 37.9, cost: 20.06 }, ticks: [0.15, 0.2, 0.25, 0.3] },
  { model: "Grok 4.7", late: { point: "425k", score: 40.0, cost: 721.3 }, early: { point: "190k", score: 42.1, cost: 604.99 }, ticks: [5, 6, 7, 8] },
];

const W = 410;
const H = 262;
const X0 = 62;
const X1 = 396;
const Y0 = 220; // score 30%
const Y1 = 14; // score 50%
const y = (score: number) => Y0 - ((score - 30) / 20) * (Y0 - Y1);

const panels = pairs.map((p) => {
  const before = p.late.cost / TRIALS;
  const after = p.early.cost / TRIALS;
  // Cost axis is zoomed to the run pair (it doesn't start at $0) so the two
  // points sit well apart; the tick labels show the range.
  const xmin = p.ticks[0];
  const xmax = p.ticks[p.ticks.length - 1] + (p.ticks[p.ticks.length - 1] - xmin) * 0.06;
  const x = (c: number) => X0 + ((c - xmin) / (xmax - xmin)) * (X1 - X0);
  const a = { x: x(before), y: y(p.late.score) };
  const b = { x: x(after), y: y(p.early.score) };
  // stop the arrow short of the amber dot
  const len = Math.hypot(b.x - a.x, b.y - a.y);
  const tip = { x: b.x - ((b.x - a.x) / len) * 18, y: b.y - ((b.y - a.y) / len) * 18 };
  return {
    model: p.model,
    from: p.late.point,
    to: p.early.point,
    a,
    b,
    tip,
    ticks: p.ticks.map((t) => ({ x: x(t), label: `$${t < 1 ? t.toFixed(2) : t}` })),
    beforeText: `${Math.floor(p.late.score)}% · $${before.toFixed(2)}`,
    afterText: `${Math.floor(p.early.score)}% · $${after.toFixed(2)}`,
    up: `${Math.floor(p.early.score - p.late.score)} points`,
    down: `${Math.floor(((p.late.cost - p.early.cost) / p.late.cost) * 100)}%`,
  };
});
</script>

<template>
  <div class="fa-par">
    <section v-for="p in panels" :key="p.model">
      <header>
        <h3>{{ p.model }}</h3>
        <p class="fa-par-change">Compaction @ <b>{{ p.to }}</b></p>
      </header>
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="`${p.model}: compaction point ${p.from} to ${p.to}, score up ${p.up}, cost down ${p.down}`">
        <defs>
          <marker :id="`fa-par-head-${p.to}`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" class="fa-par-arrowhead" />
          </marker>
        </defs>
        <g v-for="s in [30, 35, 40, 45, 50]" :key="s">
          <line :x1="X0" :x2="X1" :y1="y(s)" :y2="y(s)" :class="s === 30 ? 'fa-par-axis' : 'fa-par-grid'" />
          <text :x="X0 - 8" :y="y(s) + 4" text-anchor="end" class="fa-par-tick">{{ s }}%</text>
        </g>
        <line :x1="X0" :x2="X0" :y1="Y1" :y2="Y0" class="fa-par-axis" />
        <g v-for="t in p.ticks" :key="t.label">
          <line :x1="t.x" :x2="t.x" :y1="Y0" :y2="Y0 + 5" class="fa-par-axis" />
          <text :x="t.x" :y="Y0 + 20" text-anchor="middle" class="fa-par-tick">{{ t.label }}</text>
        </g>
        <text :x="(X0 + X1) / 2" :y="H - 4" text-anchor="middle" class="fa-par-axis-title">Cost per trial (USD, list price)</text>
        <text :x="14" :y="(Y0 + Y1) / 2" text-anchor="middle" :transform="`rotate(-90 14 ${(Y0 + Y1) / 2})`" class="fa-par-axis-title">Score (share of 95 trials)</text>
        <text :x="X0 + 10" :y="Y1 + 16" class="fa-par-better">↖ better: higher score, lower cost</text>

        <line :x1="p.a.x" :y1="p.a.y" :x2="p.tip.x" :y2="p.tip.y" class="fa-par-arrow" :marker-end="`url(#fa-par-head-${p.to})`" />

        <circle :cx="p.a.x" :cy="p.a.y" r="8" class="fa-par-before" />
        <text :x="p.a.x" :y="p.a.y + 26" text-anchor="middle" class="fa-par-point fa-par-point--before">{{ p.from }}</text>
        <text :x="p.a.x" :y="p.a.y + 42" text-anchor="middle" class="fa-par-values">{{ p.beforeText }}</text>

        <circle :cx="p.b.x" :cy="p.b.y" r="11" class="fa-par-after" />
        <text :x="p.b.x - 18" :y="p.b.y - 4" text-anchor="end" class="fa-par-point">{{ p.to }}</text>
        <text :x="p.b.x - 18" :y="p.b.y + 14" text-anchor="end" class="fa-par-values fa-par-values--after">{{ p.afterText }}</text>
      </svg>
      <footer class="fa-par-deltas" :style="{ paddingLeft: `${(X0 / W) * 100}%`, paddingRight: `${((W - X1) / W) * 100}%` }">
        <div><strong>▲ {{ p.up }}</strong><span>score</span></div>
        <div><strong>▼ {{ p.down }}</strong><span>cost per trial</span></div>
      </footer>
    </section>
  </div>
</template>

<style scoped>
.fa-par { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; font-family: var(--font-read); }
.fa-par header { display: flex; align-items: baseline; justify-content: space-between; padding-bottom: 8px; border-bottom: var(--border); }
.fa-par h3 { margin: 0; font-size: 24px; white-space: nowrap; }
.fa-par-change { margin: 0; white-space: nowrap; display: flex; align-items: baseline; gap: 8px; font: 600 15px/1 var(--font-read); color: var(--text-muted); font-variant-numeric: tabular-nums; }
.fa-par-change s { margin-left: 4px; font-weight: 800; font-size: 17px; color: var(--text-muted); text-decoration-thickness: 2px; }
.fa-par-change i { font-style: normal; font-size: 13px; color: var(--petrol); }
.fa-par-change b { font-weight: 900; font-size: 22px; color: var(--petrol); }
.fa-par svg { display: block; width: 100%; height: auto; margin-top: 6px; }
.fa-par-grid { stroke: var(--line); stroke-width: 2; }
.fa-par-axis { stroke: var(--petrol); stroke-width: 2; }
.fa-par-tick { font-size: 12px; font-weight: 600; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.fa-par-axis-title { font-size: 12px; font-weight: 700; fill: var(--text-muted); }
.fa-par-better { font-size: 12px; font-weight: 800; fill: var(--teal); }
.fa-par-arrow { stroke: var(--petrol); stroke-width: 3; }
.fa-par-arrowhead { fill: var(--petrol); }
.fa-par-before { fill: var(--petrol-tint); stroke: var(--petrol); stroke-width: 2; }
.fa-par-after { fill: var(--amber); stroke: var(--petrol); stroke-width: 2; }
.fa-par-point { font: 900 20px/1 var(--font-read); fill: var(--petrol); font-variant-numeric: tabular-nums; paint-order: stroke; stroke: var(--ivory); stroke-width: 5px; stroke-linejoin: round; }
.fa-par-point--before { font-size: 15px; font-weight: 800; fill: var(--text-muted); }
.fa-par-values { font-size: 12px; font-weight: 600; fill: var(--text-muted); font-variant-numeric: tabular-nums; paint-order: stroke; stroke: var(--ivory); stroke-width: 4px; stroke-linejoin: round; }
.fa-par-values--after { font-size: 13px; font-weight: 800; fill: var(--petrol); }
.fa-par-deltas { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px; padding-top: 10px; border-top: 2px solid var(--line); }
.fa-par-deltas div { display: flex; flex-direction: column; gap: 4px; }
/* numbers are Figtree 900 with tabular figures, per the brand's type rules */
.fa-par-deltas strong { font: 900 36px/1 var(--font-read); color: var(--teal); font-variant-numeric: tabular-nums; white-space: nowrap; }
.fa-par-deltas span { font: 600 14px/1.2 var(--font-read); color: var(--text-muted); white-space: nowrap; }
</style>
