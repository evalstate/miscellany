<script setup lang="ts">
// Which models have ever ended a trial on a provider safety stop, across the
// whole corpus, and which never have. Two labs' models and nobody else's.
// Data: data/safety-corpus.json (analysis/safety-stops/corpus.py). Counts only.
import data from "../data/safety-corpus.json";

const num = (n: number) => n.toLocaleString("en-GB");
const RARE = 5; // fewer stops than this: listed in a line, not given a bar
const max = Math.max(...data.models.map((m) => m.stops));
const lab = (name: "Anthropic" | "OpenAI", family: string) => {
  const mine = data.models.filter((m) => m.family === family);
  return {
    name, stops: data.labs[name], trials: data.labTrials[name],
    bars: mine.filter((m) => m.stops >= RARE).map((m) => ({ ...m, width: `${(m.stops / max) * 100}%` })),
    rare: mine.filter((m) => m.stops < RARE).map((m) => `${m.model} ${m.stops}`).join(", "),
    clear: data.clearModels.filter((m) => m.family === family).map((m) => m.model).join(", "),
  };
};
const labs = [lab("Anthropic", "Claude"), lab("OpenAI", "GPT")];
const others = data.clearFamilies.map((f) => ({ ...f, label: f.family === "other" ? "Other models" : f.family }))
  .sort((a, b) => Number(a.family === "other") - Number(b.family === "other"));
const otherTrials = others.reduce((s, f) => s + f.trials, 0);
const otherModels = others.reduce((s, f) => s + f.models, 0);
const deepswe = data.benchmarks.find((b) => b.bench === "DeepSWE")!;
</script>

<template>
  <div class="sfl">
    <section v-for="l in labs" :key="l.name">
      <h3>{{ l.name }}</h3>
      <p class="sfl-big"><b>{{ l.stops }}</b> trials stopped</p>
      <div v-for="m in l.bars" :key="m.model" class="sfl-row">
        <strong>{{ m.model }}</strong>
        <div class="sfl-bar"><span :style="{ width: m.width }" /><b>{{ m.stops }}</b></div>
      </div>
      <p class="sfl-foot"><b>Fewer than {{ RARE }}:</b> {{ l.rare }}.</p>
      <p class="sfl-foot"><b>None:</b> {{ l.clear }}.</p>
    </section>
    <section class="sfl-zero">
      <h3>Everyone else</h3>
      <p class="sfl-big"><b>0</b> trials stopped</p>
      <div v-for="f in others" :key="f.family" class="sfl-fam">
        <strong>{{ f.label }}</strong><span>{{ num(f.trials) }} trials</span>
      </div>
      <p class="sfl-foot"><b>{{ otherModels }} models, {{ num(otherTrials) }} trials.</b></p>
    </section>
  </div>
  <p class="lk-note sfl-note"><b>{{ data.stops }} of {{ num(data.trials) }} trials ended on a provider safety stop.</b> All are Terminal-Bench tasks: DeepSWE has {{ deepswe.stops }} in {{ num(deepswe.trials) }}.</p>
</template>

<style scoped>
.sfl { display: grid; grid-template-columns: 1fr 1fr 0.8fr; gap: 16px; align-items: stretch; font-family: var(--font-read); }
.sfl section { min-width: 0; padding: 10px 18px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.sfl section.sfl-zero { background: var(--ivory-deep); }
.sfl h3 { margin: 0; font-size: 20px; }
.sfl section p.sfl-big { margin: 0 0 8px; font-size: 15px; color: var(--petrol); }
.sfl section p.sfl-big b { font: 900 34px/1 var(--font-voice); margin-right: 6px; font-variant-numeric: tabular-nums; }
.sfl-row { display: grid; grid-template-columns: 104px 1fr; column-gap: 10px; align-items: center; margin-bottom: 6px; }
.sfl-row strong { font-size: 14px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.sfl-bar { display: flex; align-items: center; height: 16px; }
.sfl-bar span { height: 100%; min-width: 3px; border-radius: 3px; background: var(--orange); }
.sfl-bar b { margin-left: 6px; font-size: 14px; line-height: 1; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.sfl section p.sfl-foot { margin: 8px 0 0; padding-top: 7px; border-top: 2px solid var(--line); font-size: 12.5px; line-height: 1.35; color: var(--text-muted); }
.sfl section p.sfl-foot b { color: var(--petrol); font-weight: 800; }
.sfl-fam { display: flex; justify-content: space-between; align-items: baseline; padding: 1px 0; border-bottom: 2px solid var(--line); font-size: 13px; line-height: 1.3; color: var(--petrol); }
.sfl-fam strong { font-weight: 700; }
.sfl-fam span { font-size: 12.5px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.sfl-zero p.sfl-foot { border-top: 0; padding-top: 2px; }
.sfl-note { margin-top: 8px; font-size: 15px; }
</style>
