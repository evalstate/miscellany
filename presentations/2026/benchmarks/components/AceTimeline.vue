<script setup lang="ts">
// When each model first wrote an *.ace-research host, August to November
// 2026: one lane per model, a dot per day it happened (bigger for more
// trials), and how much that model had been run. Data: data/ace-research.json
// (analysis/ace-research/extract.py, lab-host-timeline report).
import ace from "../data/ace-research.json";

const W = 868, LEFT = 168, RIGHT = 560, ROW = 34, TOP = 30;
const T0 = Date.UTC(2026, 7, 1), T1 = Date.UTC(2026, 10, 15);
const x = (iso: string) => LEFT + ((Date.parse(iso + "T00:00:00Z") - T0) / (T1 - T0)) * (RIGHT - LEFT);
const fmt = (iso: string) => `${Number(iso.slice(8, 10))} ${["Aug", "Sep", "Oct", "Nov"][Number(iso.slice(5, 7)) - 8]}`;
const months = [["Aug", "2026-08-01"], ["Sep", "2026-09-01"], ["Oct", "2026-10-01"], ["Nov", "2026-11-01"]].map(([label, iso]) => ({ label, x: x(iso) }));
const num = (n: number) => n.toLocaleString("en-GB");
const rows = ace.rows.map((r, i) => {
  const first = [...r.days.map((d) => d.date), ...r.later.map((d) => d.date)].sort()[0];
  return {
    ...r, y: TOP + i * ROW, first: fmt(first),
    dots: r.days.map((d) => ({ x: x(d.date), r: 4 + Math.sqrt(d.n) * 2.2, n: d.n })),
    extra: r.later.map((d) => ({ x: x(d.date) })),
    count: r.trials ? `${r.wrote} of ${num(r.trials)}` : "1, later runs",
    before: r.before === null ? "" : `${num(r.before)} before`,
  };
});
const H = TOP + rows.length * ROW + 6;
</script>

<template>
  <div class="ace">
    <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="First sightings of the ace-research host by model, August to November 2026">
      <g v-for="m in months" :key="m.label">
        <line :x1="m.x" :x2="m.x" y1="18" :y2="H - 4" class="ace-grid" />
        <text :x="m.x + 4" y="12" class="ace-month">{{ m.label }}</text>
      </g>
      <text x="580" y="12" class="ace-month">first</text>
      <text x="650" y="12" class="ace-month">wrote it</text>
      <text x="766" y="12" class="ace-month">trials run</text>
      <g v-for="r in rows" :key="r.family">
        <text :x="LEFT - 12" :y="r.y + 17" text-anchor="end" class="ace-name">{{ r.family }}</text>
        <line :x1="LEFT" :x2="RIGHT" :y1="r.y + 12" :y2="r.y + 12" class="ace-lane" />
        <circle v-for="d in r.dots" :key="d.x" :cx="d.x" :cy="r.y + 12" :r="d.r" :class="`ace-dot is-${r.lab}`" />
        <circle v-for="d in r.extra" :key="d.x" :cx="d.x" :cy="r.y + 12" r="5" :class="`ace-dot ace-open is-${r.lab}`" />
        <text x="580" :y="r.y + 17" class="ace-val ace-strong">{{ r.first }}</text>
        <text x="650" :y="r.y + 17" class="ace-val">{{ r.count }}</text>
        <text x="766" :y="r.y + 17" class="ace-val ace-dim">{{ r.before }}</text>
      </g>
    </svg>
    <p class="ace-key"><i class="is-DeepSeek" />DeepSeek <i class="is-OpenAI" />OpenAI <i class="is-other" />other <i class="ace-ring" />later harness-leaderboard runs <span>Bigger dot: more trials that day. “Before”: trials of that model run before its first sighting.</span></p>
    <p class="ace-never"><b>Never:</b> <span v-for="n in ace.never" :key="n.family">{{ n.family }} <i>0 of {{ num(n.trials) }}</i></span></p>
  </div>
</template>

<style scoped>
.ace { font-family: var(--font-read); }
.ace svg { display: block; width: 100%; height: auto; }
.ace-grid { stroke: var(--line); stroke-width: 2; }
.ace-lane { stroke: var(--line); stroke-width: 2; }
.ace-month { font-size: 12px; font-weight: 700; fill: var(--text-muted); }
.ace-name { font-size: 14px; font-weight: 800; fill: var(--petrol); }
.ace-val { font-size: 13px; font-weight: 600; fill: var(--petrol); font-variant-numeric: tabular-nums; }
.ace-strong { font-weight: 900; }
.ace-dim { fill: var(--text-muted); font-weight: 500; }
.ace-dot.is-DeepSeek { fill: var(--orange); }
.ace-dot.is-OpenAI { fill: var(--teal); }
.ace-dot.is-other { fill: var(--petrol); }
.ace-dot { stroke: var(--ivory); stroke-width: 1.5; }
.ace-open { fill: var(--ivory) !important; stroke-width: 3; }
.ace-open.is-DeepSeek { stroke: var(--orange); }
.ace-open.is-other { stroke: var(--petrol); }
.ace .ace-key { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; margin: 8px 0 0; font-size: 12px; color: var(--text-muted); }
.ace-key i { width: 11px; height: 11px; margin-left: 8px; border-radius: 50%; }
.ace-key i:first-child { margin-left: 0; }
.ace-key i.is-DeepSeek { background: var(--orange); }
.ace-key i.is-OpenAI { background: var(--teal); }
.ace-key i.is-other { background: var(--petrol); }
.ace-key i.ace-ring { box-sizing: border-box; border: 3px solid var(--petrol); }
.ace-key span { margin-left: 12px; }
.ace .ace-never { margin: 8px 0 0; padding-top: 8px; border-top: 2px solid var(--line); font-size: 14px; line-height: 1.5; color: var(--petrol); }
.ace-never b { font-weight: 800; }
.ace-never span { display: inline-block; margin-right: 14px; white-space: nowrap; }
.ace-never span i { font-style: normal; color: var(--text-muted); font-variant-numeric: tabular-nums; }
</style>
