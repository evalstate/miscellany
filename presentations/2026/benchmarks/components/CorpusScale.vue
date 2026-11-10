<script setup lang="ts">
// The size of the corpus atif-scan read, as nine large figures in three
// colour bands: how much (petrol), how long and how costly (teal), how
// varied (paper).
// Sources: corpus scan of 8 Nov 2026 (181,732 trajectories); the census of
// the same date (164,870 unique trials) for tokens, calls, steps, models,
// harnesses and spend; data/corpus-hours.json for agent time
// (analysis/corpus-hours/hours.py).
import hours from "../data/corpus-hours.json";

const years = (Math.floor(hours.years * 10) / 10).toFixed(1);
const thousandHours = Math.floor(hours.hours / 1000);
const figures = [
  { big: "181,732", label: "agent trajectories read", tone: "ink" },
  { big: "637 billion", label: "input tokens in them", tone: "ink" },
  { big: "7.1 million", label: "tool calls", tone: "ink" },
  { big: `${years} years`, label: `of agent working time: ${thousandHours},000 hours`, tone: "teal" },
  { big: "6.6 million", label: "agent steps", tone: "teal" },
  { big: "$226,000", label: "of recorded model spend", tone: "teal" },
  { big: "91", label: "models, from 14 families", tone: "paper" },
  { big: "35", label: "harnesses", tone: "paper" },
  { big: "6 months", label: "30 Apr to 7 Nov 2026", tone: "paper" },
];
</script>

<template>
  <div class="fa-scale">
    <div v-for="f in figures" :key="f.label" :class="`is-${f.tone}`">
      <b>{{ f.big }}</b>
      <span>{{ f.label }}</span>
    </div>
  </div>
</template>

<style scoped>
.fa-scale { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px 12px; font-family: var(--font-read); }
.fa-scale div { padding: 12px 18px 14px; border: var(--border); border-radius: var(--radius-lg); }
.fa-scale b { display: block; font: 900 46px/1.05 var(--font-read); font-variant-numeric: tabular-nums; white-space: nowrap; }
.fa-scale span { display: block; margin-top: 3px; font-size: 16px; font-weight: 600; line-height: 1.25; }
.is-ink { background: var(--petrol); color: var(--ivory); }
.is-ink span { color: rgba(255, 247, 232, 0.78); }
.is-teal { background: var(--teal); border-color: var(--teal) !important; color: var(--ivory); }
.is-teal span { color: rgba(255, 247, 232, 0.86); }
.is-paper { background: var(--paper); color: var(--petrol); }
.is-paper span { color: var(--text-muted); }
</style>
