<script setup lang="ts">
// Answer-lookup flags by run (one harness, one model, one effort, 445 trials):
// the runs with 15 or more, then every model by whether benchmark material ever reached it. The replay row with no model calls is left out.
// Data: data/answer-lookup.json (analysis/answer-lookup/extract.py).
import data from "../data/answer-lookup.json";

const order = data.kinds.map((k) => ({ key: k.key, label: k.label }));
const live = data.rows.filter((r) => r.agent !== "Slai Agent Runtime");
const top = live.filter((r) => r.lookup >= 15);
const max = top[0].lookup;
const rows = top.map((r) => ({
  ...r,
  harness: r.agent.replace(/ \d+\.[\d.]+$/, ""),
  segments: order.map((k) => ({ key: k.key, n: (r.kinds as Record<string, number>)[k.key] })).filter((s) => s.n > 0),
}));
const reached = data.models.filter((m) => m.received > 0);
const flagged = data.models.filter((m) => m.received === 0 && m.lookup > 0);
const clean = data.models.filter((m) => m.lookup === 0);
</script>

<template>
  <div class="lkr">
    <div class="lk-legend">
      <span v-for="k in order" :key="k.key"><i :class="`lk-${k.key}`" />{{ k.label }}</span>
    </div>
    <div v-for="r in rows" :key="r.agent + r.model + r.effort" class="lkr-row">
      <strong>{{ r.model }} <small>{{ r.effort }}</small></strong>
      <span class="lkr-harness">{{ r.harness }}</span>
      <div class="lkr-bar">
        <span v-for="s in r.segments" :key="s.key" :class="`lk-${s.key}`" :style="{ width: `${(s.n / max) * 100}%` }" />
        <b>{{ r.lookup }}</b>
      </div>
      <em>{{ r.lookupRewarded }} rewarded</em>
    </div>
    <div class="lkr-names">
      <p><b>Benchmark material reached {{ reached.length }} of {{ data.models.length }} models at least once</b> (trials, all runs):
        <span v-for="m in reached" :key="m.model">{{ m.model }} <i>{{ m.received }}</i></span></p>
      <p><b>Flagged, nothing arrived:</b> <span v-for="m in flagged" :key="m.model">{{ m.model }}</span>
        <b class="gap">No flags:</b> <span v-for="m in clean" :key="m.model">{{ m.model }}</span></p>
    </div>
  </div>
</template>

<style scoped>
.lkr { font-family: var(--font-read); }
.lkr-row { display: grid; grid-template-columns: 250px 96px 1fr 92px; column-gap: 12px; align-items: center; margin-bottom: 4px; }
.lkr-row strong { font-size: 15px; font-weight: 700; color: var(--petrol); text-align: right; white-space: nowrap; }
.lkr-row small { font-size: 12px; font-weight: 600; color: var(--text-muted); }
.lkr-harness { font-size: 13px; font-weight: 600; color: var(--text-muted); white-space: nowrap; }
.lkr-bar { display: flex; align-items: center; gap: 2px; height: 20px; }
.lkr-bar span { height: 100%; border-radius: 3px; }
.lkr-bar b { margin-left: 6px; font-size: 15px; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.lkr-row em { font-style: normal; font-size: 13px; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.lkr-names { margin-top: 10px; padding-top: 8px; border-top: 2px solid var(--line); }
.lkr-names p { margin: 0 0 5px; font-size: 13px; line-height: 1.5; color: var(--petrol); }
.lkr-names b { font-weight: 800; }
.lkr-names b.gap { margin-left: 10px; }
.lkr-names span { display: inline-block; white-space: nowrap; margin-right: 10px; }
.lkr-names span i { font-style: normal; font-weight: 800; color: var(--text-muted); font-variant-numeric: tabular-nums; }
</style>
