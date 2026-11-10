<script setup lang="ts">
// What the "answer lookup" flags are made of: every flagged trial sorted into
// one of five kinds, drawn as one bar, with how often each kind was rewarded.
// Data: data/answer-lookup.json (analysis/answer-lookup/extract.py).
import data from "../data/answer-lookup.json";

const gloss: Record<string, string> = {
  own: "Solution, tests or a finished run for this very task",
  material: "The benchmark's repo, a mirror or a task page came back",
  upstream: "The real project's fix",
  git: "Outside the repo's history",
  search: "Nothing came back",
};
const total = data.total;
const pct = (a: number, b: number) => Math.floor((a / b) * 100);
const kinds = data.kinds.map((k) => ({ ...k, gloss: gloss[k.key], rate: pct(k.rewarded, k.trials) }));
const columns = kinds.map((k) => `${k.trials}fr`).join(" ");
const share = (Math.floor((total.lookup / total.trials) * 1000) / 10).toFixed(1);
</script>

<template>
  <div class="lkk">
    <p class="lkk-lead">
      <b>{{ total.lookup }}</b>
      <span>of {{ total.trials.toLocaleString("en-GB") }} trials in {{ total.rows }} public Terminal-Bench 2.1 runs carry an answer-lookup flag: {{ share }}%.<br />Each trial is sorted into one of five kinds.</span>
    </p>
    <div class="lkk-bar" :style="{ gridTemplateColumns: columns }">
      <span v-for="k in kinds" :key="k.key" :class="`lk-${k.key}`"><b>{{ k.trials }}</b></span>
    </div>
    <div class="lkk-cols" :style="{ gridTemplateColumns: columns }">
      <div v-for="k in kinds" :key="k.key">
        <strong>{{ k.label }}</strong>
        <em><b>{{ k.rate }}%</b> rewarded</em>
        <p>{{ k.gloss }}</p>
      </div>
    </div>
    <p class="lk-note">Across all {{ total.trials.toLocaleString("en-GB") }} trials, <b>{{ pct(total.rewarded, total.trials) }}%</b> are rewarded.</p>
  </div>
</template>

<style scoped>
.lkk { font-family: var(--font-read); }
.lkk-lead { display: flex; align-items: center; gap: 18px; margin: 0 0 12px; }
.lkk-lead b { font: 900 64px/0.9 var(--font-voice); color: var(--petrol); font-variant-numeric: tabular-nums; }
.lkk-lead span { font-size: 19px; line-height: 1.3; font-weight: 500; color: var(--petrol); }
.lkk-bar, .lkk-cols { display: grid; column-gap: 4px; }
.lkk-bar span { display: flex; align-items: center; height: 52px; border-radius: var(--radius-sm); }
.lkk-bar b { padding-left: 12px; font-size: 24px; font-weight: 900; color: var(--ivory); font-variant-numeric: tabular-nums; }
.lkk-bar .lk-own b, .lkk-bar .lk-git b, .lkk-bar .lk-search b { color: var(--petrol); }
.lkk-cols { margin-top: 10px; }
.lkk-cols > div { min-width: 0; padding-right: 8px; }
.lkk-cols strong { display: block; font-size: 16px; line-height: 1.2; font-weight: 800; color: var(--petrol); }
.lkk-cols p { margin: 4px 0 0; font-size: 13px; line-height: 1.3; color: var(--text-muted); }
.lkk-cols em { display: block; margin-top: 2px; font-style: normal; font-size: 14px; color: var(--text-muted); }
.lkk-cols em b { font-size: 20px; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
</style>
