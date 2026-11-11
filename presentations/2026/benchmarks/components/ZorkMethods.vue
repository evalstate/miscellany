<script setup lang="ts">
// extract-moves-from-video by model: how often each is rewarded, and how often
// its code uses approximate text matching to line up the scrolling screens.
// Data: data/zork-methods.json (analysis/zork-methods/extract.py). Method
// labels are pattern matches on the agent's own tool calls.
import data from "../data/zork-methods.json";

const pct = (a: number, b: number) => Math.floor((a / b) * 100);
const rows = data.models.map((m) => ({ ...m, pass: pct(m.rewarded, m.n), fz: pct(m.fuzzy, m.n) }));
const by = (name: string) => data.models.find((m) => m.model === name)!;
const sol = by("GPT-6.1 Sol"), terra = by("GPT-6.1 Terra"), opus = by("Opus 5.5"), sonnet = by("Sonnet 5.5");
const luna = data.luna.fuzzy;
</script>

<template>
  <div class="zm">
    <div class="zm-chart">
      <div class="zm-row zm-head"><span /><b><i class="zm-a" />Rewarded</b><b><i class="zm-t" />Fuzzy matching in its code</b></div>
      <div v-for="r in rows" :key="r.model" class="zm-row">
        <strong>{{ r.model }} <small>{{ r.n }}</small></strong>
        <div class="zm-bar"><span class="zm-a" :style="{ width: `${Math.max(1, r.pass) * 0.78}%` }" /><em>{{ r.pass }}%</em></div>
        <div class="zm-bar"><span class="zm-t" :style="{ width: `${Math.max(1, r.fz) * 0.78}%` }" /><em>{{ r.fz }}%</em></div>
      </div>
    </div>
    <div class="zm-facts">
      <section>
        <h3>GPT-6.1 Sol: nearly always</h3>
        <p>Fuzzy matching in <b>{{ sol.fuzzy }} of {{ sol.n }}</b> trials, cropped re-reads in {{ sol.crop }}, a table of OCR fixes in {{ sol.fixes }}.</p>
        <p class="zm-quote"><q>My plan is to do line fuzzy matching against the previous screen by selecting the longest contiguous approximate run.</q></p>
      </section>
      <section>
        <h3>It isn't required</h3>
        <p>GPT-6.1 Terra uses it in <b>{{ terra.fuzzy }} of {{ terra.n }}</b> and is rewarded {{ terra.rewarded }} times, in a median {{ terra.steps }} steps. Opus 5.5 and Sonnet 5.5 pass all {{ opus.n + sonnet.n }}.</p>
      </section>
      <section>
        <h3>It helps GPT-6 Luna</h3>
        <p>With fuzzy matching <b>{{ luna.with.rewarded }} of {{ luna.with.n }}</b> rewarded. Without it, <b>{{ luna.without.rewarded }} of {{ luna.without.n }}</b>.</p>
      </section>
    </div>
  </div>
</template>

<style scoped>
.zm { display: grid; grid-template-columns: 1.25fr 1fr; gap: 22px; align-items: start; font-family: var(--font-read); }
.zm-row { display: grid; grid-template-columns: 158px 1fr 1fr; column-gap: 12px; align-items: center; height: 30px; border-bottom: 2px solid var(--line); }
.zm-head { height: 26px; border-bottom: var(--border); }
.zm-head b { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.zm-head i { width: 13px; height: 13px; border-radius: 3px; }
.zm-row strong { font-size: 15px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.zm-row strong small { margin-left: 4px; font-size: 12px; font-weight: 500; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.zm-bar { display: flex; align-items: center; height: 16px; }
.zm-bar span { height: 100%; border-radius: 3px; }
.zm-a { background: var(--amber); box-shadow: inset 0 0 0 2px var(--petrol); }
.zm-t { background: var(--teal); }
.zm-bar em { margin-left: 6px; font-style: normal; font-size: 14px; line-height: 1; font-weight: 800; color: var(--petrol); font-variant-numeric: tabular-nums; }
.zm-facts { display: grid; gap: 9px; }
.zm-facts section { padding: 9px 16px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.zm-facts h3 { margin: 0 0 3px; font-size: 18px; }
.zm-facts p { margin: 0; font-size: 14px; line-height: 1.35; color: var(--petrol); }
.zm-facts p b { font-weight: 800; }
.zm-facts p.zm-quote { margin-top: 5px; padding-top: 5px; border-top: 2px solid var(--line); font-size: 13px; }
.zm-facts q { quotes: "“" "”"; font-style: italic; }
</style>
