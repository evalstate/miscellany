<script setup lang="ts">
// Answer-lookup flags by task: the ten tasks with the most, each bar split by
// kind of lookup, and whether the task itself needs the network.
// Data: data/answer-lookup.json (analysis/answer-lookup/extract.py).
import data from "../data/answer-lookup.json";

const order = data.kinds.map((k) => ({ key: k.key, label: k.label }));
const top = data.tasks.slice(0, 10);
const max = top[0].lookup;
const rows = top.map((t) => ({
  ...t,
  segments: order.map((k) => ({ key: k.key, n: (t.kinds as Record<string, number>)[k.key] })).filter((s) => s.n > 0),
  web: t.network === "required" || t.network === "expected",
}));
const held = data.tasks.slice(0, 8).reduce((s, t) => s + t.lookup, 0);
const none = data.tasks.filter((t) => t.lookup === 0).length;
</script>

<template>
  <div class="lkt">
    <div class="lk-legend">
      <span v-for="k in order" :key="k.key"><i :class="`lk-${k.key}`" />{{ k.label }}</span>
    </div>
    <div v-for="r in rows" :key="r.task" class="lkt-row">
      <strong>{{ r.task }}</strong>
      <div class="lkt-bar">
        <span v-for="s in r.segments" :key="s.key" :class="`lk-${s.key}`" :style="{ width: `${(s.n / max) * 100}%` }" />
        <b>{{ r.lookup }}</b>
      </div>
      <em>{{ r.lookupRewarded }} rewarded</em>
      <em class="lkt-web">{{ r.web ? "needs the web" : "" }}</em>
    </div>
    <p class="lk-note"><b>Eight of {{ data.tasks.length }} tasks hold {{ held }} of the {{ data.total.lookup }} flags.</b> {{ none }} tasks have none. Each task has 325 trials here.</p>
  </div>
</template>

<style scoped>
.lkt { font-family: var(--font-read); }
.lkt-row { display: grid; grid-template-columns: 236px 1fr 92px 104px; column-gap: 12px; align-items: center; margin-bottom: 4px; }
.lkt-row strong { font: 500 14px var(--font-code); color: var(--petrol); text-align: right; white-space: nowrap; }
.lkt-bar { display: flex; align-items: center; gap: 2px; height: 20px; }
.lkt-bar span { height: 100%; border-radius: 3px; }
.lkt-bar b { margin-left: 6px; font-size: 15px; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.lkt-row em { font-style: normal; font-size: 13px; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.lkt-row em.lkt-web { color: var(--teal); font-weight: 800; }
</style>
