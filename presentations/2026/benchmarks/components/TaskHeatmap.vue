<script setup lang="ts">
// 89 tasks × 5 reasoning levels; each cell is the task's passes out of 5.
// Tasks are ordered by the lowest level that solves them reliably (3+ of 5),
// then by reasoning tokens, so the dark area grows to the right as the level
// rises. The orange tick on each row marks that level's reach: the last task
// it, or a lower level, solves reliably. The right-hand panel carries the
// run figures for the level: score, reasoning tokens per trial, run cost.
import data from "../data/luna-effort.json";

const props = withDefaults(
  defineProps<{
    runs?: { effort: string; score: number; cost: number }[];
    pick?: string;
    reach?: boolean;
  }>(),
  { runs: () => [], pick: "", reach: true },
);

const levels = data.levels.map((l) => l.effort);
type Task = (typeof data.tasks)[number];
const pass = (t: Task, e: string) => t.passes[e as keyof Task["passes"]];
const firstSolve = (t: Task) => {
  const i = levels.findIndex((e) => pass(t, e) >= 3);
  return i < 0 ? levels.length : i;
};
const total = (t: Task) => levels.reduce((a, e) => a + pass(t, e), 0);
// within each group, fewest reasoning tokens first (median over the task's passing trials); never-passed tasks last
const need = (t: Task) => t.need ?? Infinity;
const tasks = [...data.tasks].sort((a, b) => firstSolve(a) - firstSolve(b) || need(a) - need(b) || total(b) - total(a));

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.floor((s.length - 1) / 2)] : 0;
};
const groupDefs = [
  { label: "solved at low", from: 0, to: 0 },
  { label: "medium", from: 1, to: 1 },
  { label: "higher", from: 2, to: 4 },
  { label: "unsolved", from: 5, to: 5 },
];

const W = 1000;
const LABEL = 84;
const PANEL = 236;
const GRID = W - LABEL - PANEL - 12;
const cell = GRID / tasks.length;
const rowH = 57;
const gridH = levels.length * rowH;
const TOP = 22;
const H = TOP + gridH + 58;
const groups = groupDefs.map((g) => {
  const idx = tasks.map((t, i) => [t, i] as const).filter(([t]) => firstSolve(t) >= g.from && firstSolve(t) <= g.to);
  const start = idx.length ? idx[0][1] : 0;
  const needs = idx.map(([t]) => t.need).filter((n): n is number => n !== null);
  return { ...g, n: idx.length, x0: LABEL + start * cell, x1: LABEL + (start + idx.length) * cell, need: median(needs) };
});
// reach: how many tasks, from the left, are solved reliably at this level or below
const reachX = levels.map((_, r) => LABEL + tasks.filter((t) => firstSolve(t) <= r).length * cell);
const meanTokens = Object.fromEntries(data.levels.map((l) => [l.effort, l.mean]));
const run = (e: string) => props.runs.find((x) => x.effort === e);
const k = (v: number) => (v >= 1000 ? `${(Math.floor(v / 100) / 10).toFixed(1)}k` : `${v}`);
const PX = W - PANEL + 8;
</script>

<template>
  <svg class="th" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Pass count by task and reasoning level, tasks ordered easiest to hardest">
    <text :x="PX" :y="14" class="th-head">Score</text>
    <text :x="PX + 84" :y="14" class="th-head">Cost of the run</text>
    <g :transform="`translate(0 ${TOP})`">
    <g v-for="(lv, r) in levels" :key="lv">
      <rect v-if="lv === pick" :x="2" :y="r * rowH" :width="W - 4" :height="rowH" rx="10" class="th-pick" />
      <text :x="LABEL - 10" :y="r * rowH + rowH / 2 + 5" text-anchor="end" class="th-lv">{{ lv }}</text>
      <rect
        v-for="(t, i) in tasks"
        :key="t.task"
        :x="LABEL + i * cell + 0.5"
        :y="r * rowH + 3"
        :width="cell - 1"
        :height="rowH - 6"
        rx="2"
        class="th-cell"
        :style="{ opacity: 0.08 + 0.92 * (pass(t, lv) / 5) }"
      />
      <line v-if="reach" :x1="reachX[r]" :x2="reachX[r]" :y1="r * rowH" :y2="r * rowH + rowH" class="th-reach" />
      <template v-if="run(lv)">
        <text :x="PX" :y="r * rowH + rowH / 2 + 4" class="th-score">{{ Math.floor(run(lv)!.score) }}%</text>
        <text :x="PX + 84" :y="r * rowH + rowH / 2 + 4" class="th-score">${{ run(lv)!.cost.toFixed(2) }}</text>
        <text :x="PX" :y="r * rowH + rowH / 2 + 21" class="th-meta">{{ k(meanTokens[lv]) }} reasoning / trial · {{ (Math.floor((run(lv)!.cost / tasks.length) * 1000) / 10).toFixed(1) }}¢ a task</text>
      </template>
    </g>
    <g v-for="g in groups" :key="g.label">
      <line :x1="g.x0 + 1" :x2="g.x1 - 1" :y1="gridH + 8" :y2="gridH + 8" class="th-axis" />
      <text :x="g.x0 + 2" :y="gridH + 27" class="th-group"><tspan class="th-n">{{ g.n }}</tspan> {{ g.label }}</text>
      <text v-if="g.need && g.to < 5" :x="g.x0 + 2" :y="gridH + 44" class="th-need">≈{{ k(g.need) }} reasoning tokens</text>
    </g>
    </g>
  </svg>
</template>

<style scoped>
.th { display: block; width: 100%; height: auto; font-family: var(--font-read); }
.th-cell { fill: var(--petrol); }
.th-pick { fill: none; stroke: var(--petrol); stroke-width: 2; }
.th-lv { font: 800 16px var(--font-read); fill: var(--petrol); }
.th-reach { stroke: var(--orange); stroke-width: 4; }
.th-axis { stroke: var(--petrol); stroke-width: 3; }
.th-group { font: 700 13px var(--font-read); fill: var(--petrol); }
.th-n { font-weight: 900; }
.th-need { font: 600 12px var(--font-read); fill: var(--text-muted); }
.th-head { font: 700 12px var(--font-read); fill: var(--text-muted); }
.th-score { font: 900 30px var(--font-read); fill: var(--petrol); font-variant-numeric: tabular-nums; }
.th-meta { font: 600 12px var(--font-read); fill: var(--text-muted); font-variant-numeric: tabular-nums; }
</style>
