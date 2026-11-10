<script setup lang="ts">
// Cost of one full TB 2.1 run (445 trials) with fast-agent, month by month,
// for the Sol models at high effort and the Luna models at max effort.
// `shown` is how many months are revealed (drive it from $clicks + 1).
//
// Amber bars are fast-agent runs. Their cost is costUsdStandardList from
// data/run-costs.json: each run's tokens priced at one list-rate table, so
// billing tier and route don't move the bars. A list-price equivalent, not
// money spent.
// Petrol bars are public leaderboard runs on other harnesses (publicRuns,
// analysis/run-costs/public_runs.py): GPT-5.5, the model before the Sol line,
// and the first GPT-5.6 Luna run, made before that model's price cut. Their
// cost is what each run recorded, so it is on a different basis.
import { computed } from "vue";
import raw from "../data/run-costs.json";

const props = withDefaults(defineProps<{ shown?: number }>(), { shown: 5 });

type Run = { designation: string; month: string; model: string; effort: string; score: number; costUsdStandardList: number; pick?: boolean };
type PublicRun = { designation: string; month: string; date: string; model: string; harness: string; score: number; costUsd: number };
// one public run opens each panel
const opener: Record<string, string> = { Sol: "terminus-2", Luna: "codex" };
const earlier = (raw.publicRuns as PublicRun[]).filter((r) => r.harness === opener[r.designation]);
const picks = (raw.runs as Run[]).filter((r) => r.pick);
const months = [...new Set(picks.map((r) => r.month))].sort();
const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const label = (m: string) => MONTH[Number(m.slice(5)) - 1];
const pretty = (id: string) => id.replace(/^gpt-/, "GPT-").replace(/-(sol|luna)$/, "");

const W = 868;
const H = 384;
const PANEL = 410;
const TOP = 48;
const BASE = 298;
const BAR = 46;
const BREAK = 16; // extra space after the first month: May and June have no run

const panels = computed(() =>
  [
    { name: "Sol", effort: "high effort", max: 500, ticks: [0, 100, 200, 300, 400, 500] },
    { name: "Luna", effort: "max effort", max: 250, ticks: [0, 50, 100, 150, 200, 250] },
  ].map((p, i) => {
    const x0 = i * (PANEL + 48);
    const slot = (PANEL - 44 - BREAK) / (months.length + 1);
    const pub = earlier.find((q) => q.designation === p.name);
    const entries = [
      ...(pub ? [{ key: pub.date, cost: pub.costUsd, score: pub.score, month: `${Number(pub.date.slice(8))} ${label(pub.month)}`, model: pretty(pub.model), other: pub.harness, kind: "public" }] : []),
      ...months.map((m) => {
        const r = picks.find((q) => q.designation === p.name && q.month === m)!;
        const off = r.effort !== p.effort.split(" ")[0];
        return { key: m, cost: r.costUsdStandardList, score: r.score, month: label(m), model: pretty(r.model), other: off ? `at ${r.effort}` : "", kind: off ? "other" : "" };
      }),
    ];
    const bars = entries.map((e, k) => {
      const h = (e.cost / p.max) * (BASE - TOP);
      const left = x0 + 44 + k * slot + (k > 0 ? BREAK : 0);
      return { ...e, x: left + (slot - BAR) / 2, cx: left + slot / 2, y: BASE - h, h, cost: `$${Math.floor(e.cost)}`, score: `${Math.floor(e.score * 100)}%`, on: k < props.shown };
    });
    return { ...p, x0, bars, y: (v: number) => BASE - (v / p.max) * (BASE - TOP) };
  }),
);
</script>

<template>
  <svg class="fa-cost" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="List-price cost of a full Terminal-Bench 2.1 run by month, Sol and Luna models">
    <g v-for="p in panels" :key="p.name">
      <text :x="p.x0 + 44" y="16" class="fa-cost-title">{{ p.name }} <tspan class="fa-cost-sub" dx="6">{{ p.effort }}</tspan></text>
      <g v-for="t in p.ticks" :key="t">
        <line :x1="p.x0 + 44" :x2="p.x0 + PANEL" :y1="p.y(t)" :y2="p.y(t)" :class="t ? 'fa-cost-grid' : 'fa-cost-base'" />
        <text :x="p.x0 + 36" :y="p.y(t) + 4" text-anchor="end" class="fa-cost-tick">${{ t }}</text>
      </g>
      <template v-for="b in p.bars" :key="b.key">
        <g class="fa-cost-bar" :class="[b.kind, { on: b.on }]">
          <rect :x="b.x" :y="b.y" :width="BAR" :height="b.h" rx="4" />
          <text :x="b.cx" :y="b.y - 8" text-anchor="middle" class="fa-cost-value">{{ b.cost }}</text>
          <text :x="b.cx" :y="BASE + 20" text-anchor="middle" class="fa-cost-month">{{ b.month }}</text>
          <text :x="b.cx" :y="BASE + 36" text-anchor="middle" class="fa-cost-sub">{{ b.model }}</text>
          <text :x="b.cx" :y="BASE + 62" text-anchor="middle" class="fa-cost-score">{{ b.score }}</text>
          <text v-if="b.other" :x="b.cx" :y="BASE + 80" text-anchor="middle" class="fa-cost-sub">{{ b.other }}</text>
        </g>
      </template>
      <text :x="p.x0 + 36" :y="BASE + 60" text-anchor="end" class="fa-cost-tick">score</text>
    </g>
  </svg>
</template>

<style scoped>
.fa-cost {
  display: block;
  width: 100%;
  height: auto;
  font-family: var(--font-read);
}
.fa-cost-title { font: 900 20px/1 var(--font-voice); letter-spacing: var(--track-voice); fill: var(--petrol); }
.fa-cost-sub { font: 500 12px/1 var(--font-read); letter-spacing: 0; fill: var(--text-muted); }
.fa-cost-tick { font-size: 12px; font-weight: 600; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.fa-cost-grid { stroke: var(--line); stroke-width: 2; }
.fa-cost-base { stroke: var(--petrol); stroke-width: 2; }
.fa-cost-bar { opacity: 0; transition: opacity var(--dur-ui) var(--ease-ui); }
.fa-cost-bar.on { opacity: 1; }
.fa-cost-bar rect { fill: var(--amber); stroke: var(--petrol); stroke-width: 2; }
/* a run at a different effort level is not part of the series: outline only */
.fa-cost-bar.other rect { fill: var(--ivory); stroke-dasharray: 5 4; }
/* a public leaderboard run on another harness: not fast-agent, so not amber */
.fa-cost-bar.public rect { fill: var(--petrol); }
.fa-cost-value { font-size: 16px; font-weight: 900; fill: var(--petrol); font-variant-numeric: tabular-nums; }
.fa-cost-month { font-size: 14px; font-weight: 700; fill: var(--petrol); }
.fa-cost-score { font-size: 22px; font-weight: 900; fill: var(--petrol); font-variant-numeric: tabular-nums; }
</style>
