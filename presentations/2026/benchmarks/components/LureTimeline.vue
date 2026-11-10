<script setup lang="ts">
// The install-lure campaign on one time axis, February to November 2026:
// what was online when (five lanes), and how many runs met it each month.
// Dates are from the lure report's follow-up (git history, archive captures,
// package registries, trace timestamps). Counts: data/lure.json.
import lure from "../data/lure.json";

const W = 868, LEFT = 150, RIGHT = 862;
const T0 = Date.UTC(2026, 1, 1), T1 = Date.UTC(2026, 11, 1);
const x = (iso: string) => LEFT + ((Date.parse(iso + "T00:00:00Z") - T0) / (T1 - T0)) * (RIGHT - LEFT);
const months = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"].map((label, i) => {
  const iso = `2026-${String(i + 2).padStart(2, "0")}-01`;
  const next = i === 9 ? "2026-12-01" : `2026-${String(i + 3).padStart(2, "0")}-01`;
  return { label, x: x(iso), mid: (x(iso) + x(next)) / 2 };
});

const ROW = 31, TOP = 26, H = 18;
const lanes = [
  { name: "PyPI package", from: "2026-02-25", to: "2026-03-26", fade: "2026-05-08", cls: "pkg",
    note: "0.1.0 on 25 Feb, 0.2.0 on 19 Mar · gone from PyPI by 8 May" },
  { name: "Decoy model card", from: "2026-03-19", to: "2026-12-01", cls: "card",
    inside: "created 46 minutes after package 0.2.0 · still live on 7 Nov" },
  { name: "Notes on a lab site", from: "2026-05-15", to: "2026-08-17", cls: "site",
    inside: "last live reply 17 Aug", note: "404 from 28 Aug" },
  { name: "Two mirrors", from: "2026-06-04", to: "2026-08-27", cls: "site",
    inside: "17 tasks, 8–17 Jun", before: "27 Aug: “Take controlled research pages offline”" },
  { name: "Search-index copies", from: "2026-08-28", to: "2026-11-05", cls: "index",
    inside: "snippets still served", note: "to 5 Nov" },
].map((l, i) => ({ ...l, y: TOP + i * ROW, x0: x(l.from), x1: Math.min(x(l.to), RIGHT) }));

const BASE = TOP + 5 * ROW + 96, BARMAX = 62;
const peak = Math.max(...lure.months.map((m) => m.met));
const bars = lure.months.map((m) => {
  const i = Number(m.month.slice(5)) - 2;
  return { ...m, mid: months[i].mid, h: (m.met / peak) * BARMAX };
});
</script>

<template>
  <svg class="ltl" :viewBox="`0 0 ${W} ${BASE + 22}`" role="img" aria-label="Timeline of the install-lure campaign, February to November 2026">
    <g v-for="m in months" :key="m.label">
      <line :x1="m.x" :x2="m.x" y1="18" :y2="BASE" class="ltl-grid" />
      <text :x="m.mid" y="12" text-anchor="middle" class="ltl-month">{{ m.label }}</text>
    </g>

    <g v-for="l in lanes" :key="l.name">
      <text :x="LEFT - 10" :y="l.y + 13" text-anchor="end" class="ltl-name">{{ l.name }}</text>
      <rect v-if="l.fade" :x="l.x1" :y="l.y" :width="x(l.fade) - l.x1" :height="H" rx="4" class="ltl-fade" />
      <rect :x="l.x0" :y="l.y" :width="l.x1 - l.x0" :height="H" rx="4" :class="`ltl-bar is-${l.cls}`" />
      <text v-if="l.inside" :x="l.x0 + 8" :y="l.y + 13" :class="`ltl-inside is-${l.cls}`">{{ l.inside }}</text>
      <text v-if="l.before" :x="l.x0 - 8" :y="l.y + 13" text-anchor="end" class="ltl-note ltl-before">{{ l.before }}</text>
      <text v-if="l.note" :x="(l.fade ? x(l.fade) : l.x1) + 8" :y="l.y + 13" class="ltl-note">{{ l.note }}</text>
    </g>

    <text :x="LEFT - 10" :y="BASE - 26" text-anchor="end" class="ltl-name">Runs that met it</text>
    <text :x="LEFT - 10" :y="BASE - 10" text-anchor="end" class="ltl-sub">{{ lure.encounters }} in all</text>
    <text :x="LEFT - 10" :y="BASE + 15" text-anchor="end" class="ltl-sub">runs that searched the web</text>
    <line :x1="LEFT" :x2="RIGHT" :y1="BASE" :y2="BASE" class="ltl-axis" />
    <g v-for="b in bars" :key="b.month">
      <template v-if="b.met > 0">
        <rect :x="b.mid - 22" :y="BASE - b.h" width="44" :height="b.h" rx="3" class="ltl-col" />
        <text :x="b.mid" :y="BASE - b.h - 6" text-anchor="middle" class="ltl-count">{{ b.met }}</text>
      </template>
      <text v-else :x="b.mid" :y="BASE - 6" text-anchor="middle" class="ltl-count">0</text>
      <text :x="b.mid" :y="BASE + 15" text-anchor="middle" class="ltl-sub">{{ b.searched }}</text>
    </g>
    <text :x="x('2026-07-09')" :y="BASE - BARMAX - 24" class="ltl-note">first sighting 9 Jul</text>
  </svg>
</template>

<style scoped>
.ltl { display: block; width: 100%; height: auto; font-family: var(--font-read); }
.ltl-grid { stroke: var(--line); stroke-width: 2; }
.ltl-axis { stroke: var(--petrol); stroke-width: 2; }
.ltl-month { font-size: 12px; font-weight: 700; fill: var(--text-muted); }
.ltl-name { font-size: 13px; font-weight: 800; fill: var(--petrol); }
.ltl-sub { font-size: 11px; font-weight: 500; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.ltl-bar.is-pkg { fill: var(--petrol-tint); }
.ltl-bar.is-card { fill: var(--teal); }
.ltl-bar.is-site { fill: var(--petrol); }
.ltl-bar.is-index { fill: var(--paper); stroke: var(--petrol); stroke-width: 2; stroke-dasharray: 5 4; }
.ltl-fade { fill: none; stroke: var(--petrol-tint); stroke-width: 2; stroke-dasharray: 4 4; }
.ltl-inside { font-size: 11.5px; font-weight: 700; fill: var(--ivory); }
.ltl-inside.is-index { fill: var(--petrol); }
.ltl-note { font-size: 11.5px; font-weight: 600; fill: var(--petrol); paint-order: stroke; stroke: var(--ivory); stroke-width: 4px; stroke-linejoin: round; }
.ltl-col { fill: var(--orange); }
.ltl-count { font-size: 15px; font-weight: 900; fill: var(--petrol); font-variant-numeric: tabular-nums; }
.ltl-before { font-size: 10.5px; }
</style>
