<script setup lang="ts">
// When a provider safety stop lands in a trial, and what is left afterwards.
// Data: data/safety-stops.json (analysis/safety-stops/extract.py).
import data from "../data/safety-stops.json";

const pct = (a: number, b: number) => Math.floor((a / b) * 100);
const labs = (["Claude", "OpenAI"] as const).map((lab) => {
  const v = data.labs[lab];
  return { lab: lab === "Claude" ? "Claude 5.5 models" : "OpenAI models", stops: v.stops, before: v.beforeAnyTool,
    share: pct(v.beforeAnyTool, v.stops), width: `${(v.beforeAnyTool / v.stops) * 100}%` };
});
const opus = data.rows.filter((r) => r.model === "Opus 5.5");
const opusStops = opus.reduce((s, r) => s + r.stops, 0), opusTrials = opus.reduce((s, r) => s + r.trials, 0);
const openai = data.rows.filter((r) => r.lab === "OpenAI" && r.stops > 1).map((r) => r.stops);
// models with no stop in any of their runs
const stopped = new Set(data.rows.map((r) => r.model));
const names = (lab: "Claude" | "OpenAI") => [...new Set(data.noStops[lab].map((r) => r.model.replace("Claude ", "")))].filter((m) => !stopped.has(m));
</script>

<template>
  <div class="sfx">
    <section>
      <h3>When it lands</h3>
      <div v-for="l in labs" :key="l.lab" class="sfx-when">
        <strong>{{ l.lab }}</strong>
        <div class="sfx-track"><span :style="{ width: l.width }" /><i /></div>
        <p><b>{{ l.before }} of {{ l.stops }}</b> before any tool call. The rest come after the agent has started work.</p>
      </div>
    </section>
    <section>
      <h3>What is left</h3>
      <p class="sfx-big"><b>{{ data.stops }}</b> trials stopped</p>
      <p>The stop is returned as an error and the trial ends there. None carries on.</p>
      <p class="sfx-big"><b>{{ data.rewarded }}</b> still rewarded</p>
      <p>In those, the answer was already on disk when the stop came.</p>
    </section>
    <section>
      <h3>What it costs</h3>
      <p>Opus 5.5: <b>{{ opusStops }} of {{ opusTrials }}</b> trials, {{ pct(opusStops, opusTrials) }}% of the benchmark.</p>
      <p>The OpenAI runs: <b>{{ Math.min(...openai) }} to {{ Math.max(...openai) }}</b> of 445 each.</p>
      <p class="sfx-none"><b>No stops at all:</b> {{ names("OpenAI").join(", ") }}; {{ names("Claude").join(", ") }}.</p>
    </section>
  </div>
</template>

<style scoped>
.sfx { display: grid; grid-template-columns: 1.15fr 1fr 1fr; gap: 16px; align-items: stretch; font-family: var(--font-read); }
.sfx section { min-width: 0; padding: 14px 18px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.sfx h3 { margin: 0 0 10px; font-size: 20px; }
.sfx section p { margin: 0 0 9px; font-size: 14.5px; line-height: 1.35; color: var(--petrol); }
.sfx section p b { font-weight: 800; }
.sfx-when { margin-bottom: 14px; }
.sfx-when strong { display: block; font-size: 15px; font-weight: 800; color: var(--petrol); }
.sfx-track { display: flex; gap: 2px; height: 20px; margin: 5px 0 6px; }
.sfx-track span { border-radius: 3px; background: var(--orange); }
.sfx-track i { flex: 1; border-radius: 3px; background: var(--petrol-tint); }
.sfx section p.sfx-big { margin: 0 0 2px; font-size: 16px; }
.sfx section p.sfx-big b { font: 900 32px/1 var(--font-voice); margin-right: 6px; font-variant-numeric: tabular-nums; }
.sfx section p.sfx-none { margin-top: 12px; padding-top: 9px; border-top: 2px solid var(--line); font-size: 13px; color: var(--text-muted); }
.sfx section p.sfx-none b { color: var(--petrol); }
</style>
