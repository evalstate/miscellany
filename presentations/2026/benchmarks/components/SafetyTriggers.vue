<script setup lang="ts">
// Which tasks end on a provider safety stop, for the two groups of models
// where it happens. Bars are stops out of that task's trials in the affected
// runs. Data: data/safety-stops.json (analysis/safety-stops/extract.py).
import data from "../data/safety-stops.json";

const KIND: Record<string, string> = {
  "break-filter-js-from-html": "security", "crack-7z-hash": "security", "vulnerable-secret": "security",
  "password-recovery": "security", "fix-code-vulnerability": "security", "feal-differential-cryptanalysis": "security",
  "model-extraction-relu-logits": "model", "protein-assembly": "biology", "dna-assembly": "biology", "dna-insert": "biology",
  "sparql-university": "other",
};
const num = (n: number) => n.toLocaleString("en-GB");
const panel = (lab: "Claude" | "OpenAI", title: string, sub: string) => {
  const v = data.labs[lab];
  return { title, sub, stops: v.stops, trials: v.trialsInThoseRuns, runs: v.runs,
    tasks: v.tasks.map((t) => ({ ...t, kind: KIND[t.task] ?? "other", width: `${(t.stops / t.trials) * 100}%` })) };
};
const panels = [
  panel("Claude", "Claude Opus, Sonnet and Haiku 5.5", "on fast-agent"),
  panel("OpenAI", "GPT-6 Astra, GPT-5.6 Sol, GPT-6.1 Sol and Terra", "on four harnesses"),
];
const kinds = [["security", "Security and cryptography"], ["biology", "Biology"], ["model", "Model extraction"], ["other", "Other"]];
</script>

<template>
  <div class="sft">
    <div class="lk-legend"><span v-for="[k, label] in kinds" :key="k"><i :class="`sf-${k}`" />{{ label }}</span></div>
    <div class="sft-grid">
      <section v-for="p in panels" :key="p.title">
        <h3>{{ p.title }}</h3>
        <p class="sft-sub">{{ p.sub }} · <b>{{ p.stops }} stops</b> in {{ num(p.trials) }} trials, {{ p.runs }} runs</p>
        <div v-for="t in p.tasks" :key="t.task" class="sft-row">
          <strong>{{ t.task }}</strong>
          <div class="sft-track"><span :class="`sf-${t.kind}`" :style="{ width: t.width }" /></div>
          <em>{{ t.stops }} of {{ t.trials }}</em>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.sft { font-family: var(--font-read); }
.sft-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; align-items: start; }
.sft h3 { margin: 0; font-size: 18px; white-space: nowrap; }
.sft .sft-sub { margin: 2px 0 10px; font-size: 13px; color: var(--text-muted); }
.sft-sub b { color: var(--petrol); font-weight: 800; }
.sft-row { display: grid; grid-template-columns: 226px 1fr 62px; column-gap: 10px; align-items: center; margin-bottom: 5px; }
.sft-row strong { font: 500 12px var(--font-code); color: var(--petrol); text-align: right; white-space: nowrap; }
.sft-track { height: 18px; border-radius: 3px; background: var(--ivory-deep); }
.sft-track span { display: block; height: 100%; min-width: 3px; border-radius: 3px; }
.sft-row em { font-style: normal; font-size: 13px; font-weight: 700; color: var(--petrol); font-variant-numeric: tabular-nums; white-space: nowrap; }
</style>
