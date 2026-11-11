<script setup lang="ts">
// Benchmark material reaching the agent, as a rate per model over the whole
// corpus, not a count: the corpus is mostly the models we chose to run most.
// Terminal-Bench 2.x, models with 1,000 or more trials.
// Data: data/lookup-rates.json (analysis/answer-lookup/corpus_rates.py).
import data from "../data/lookup-rates.json";

const num = (n: number) => n.toLocaleString("en-GB");
const rate = (got: number, n: number) => (Math.floor((got / n) * 10000) / 10).toFixed(1);
const flagged = data.models.filter((m) => m.obtained > 0);
const max = Math.max(...flagged.map((m) => m.obtained / m.trials));
const rows = flagged.map((m) => ({ ...m, rate: rate(m.obtained, m.trials), width: `${(m.obtained / m.trials / max) * 100}%` }));
const clear = data.models.filter((m) => m.obtained === 0);
const clearTrials = clear.reduce((s, m) => s + m.trials, 0);
const benches = data.benchmarks.map((b) => ({ ...b, rate: rate(b.obtained, b.trials) }));
const tb = data.benchmarks[0];
const top = data.models.reduce((a, b) => (b.trials > a.trials ? b : a));
const small = data.small.slice(0, 2);
</script>

<template>
  <div class="lkr">
    <div class="lkr-chart">
      <div class="lkr-row lkr-head"><span>Terminal-Bench 2.x, models with {{ num(data.minTrials) }}+ trials</span><b>Trials where benchmark material arrived, per 1,000</b></div>
      <div v-for="r in rows" :key="r.model" class="lkr-row">
        <strong>{{ r.model }} <small>{{ num(r.trials) }}</small></strong>
        <div class="lkr-bar"><span :style="{ width: r.width }" /><em>{{ r.rate }}</em><i>{{ r.obtained }}</i></div>
      </div>
    </div>
    <div class="lkr-side">
      <section>
        <h3>By benchmark</h3>
        <div v-for="b in benches" :key="b.bench" class="lkr-bench"><strong>{{ b.bench }}</strong><span><b>{{ b.rate }}</b> per 1,000 <small>{{ b.obtained }} of {{ num(b.trials) }}</small></span></div>
      </section>
      <section>
        <h3>None seen by this scan</h3>
        <p>{{ clear.map((m) => m.model).join(", ") }}: <b>0 in {{ num(clearTrials) }}</b> trials.</p>
      </section>
      <section>
        <h3>Why a rate</h3>
        <p>{{ top.model }} is {{ num(top.trials) }} of the {{ num(tb.trials) }} trials and has {{ top.obtained }} of the {{ tb.obtained }} flags. Smaller samples run higher: {{ small.map((m) => `${m.model} ${rate(m.obtained, m.trials)} (${m.trials} trials)`).join(", ") }}.</p>
      </section>
    </div>
  </div>
</template>

<style scoped>
.lkr { display: grid; grid-template-columns: 1.35fr 1fr; gap: 22px; align-items: start; font-family: var(--font-read); }
.lkr-row { display: grid; grid-template-columns: 196px 1fr; column-gap: 12px; align-items: center; height: 21px; border-bottom: 2px solid var(--line); }
.lkr-head { height: 24px; border-bottom: var(--border); }
.lkr-head span, .lkr-head b { font-size: 12px; font-weight: 700; color: var(--text-muted); white-space: nowrap; }
.lkr-head span { font-weight: 500; overflow: visible; }
.lkr-head { grid-template-columns: 1fr; }
.lkr-head span { display: none; }
.lkr-row strong { font-size: 13.5px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.lkr-row strong small { margin-left: 4px; font-size: 11.5px; font-weight: 500; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.lkr-bar { display: flex; align-items: center; height: 13px; }
.lkr-bar span { height: 100%; min-width: 2px; max-width: 78%; border-radius: 3px; background: var(--orange); }
.lkr-bar em { margin-left: 6px; font-style: normal; font-size: 13px; line-height: 1; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.lkr-bar i { margin-left: 6px; font-style: normal; font-size: 11.5px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.lkr-side { display: grid; gap: 9px; }
.lkr-side section { padding: 9px 16px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.lkr-side h3 { margin: 0 0 3px; font-size: 18px; }
.lkr-side p { margin: 0; font-size: 13.5px; line-height: 1.35; color: var(--petrol); }
.lkr-side p b { font-weight: 800; }
.lkr-bench { display: flex; justify-content: space-between; align-items: baseline; padding: 2px 0; border-bottom: 2px solid var(--line); font-size: 13.5px; color: var(--petrol); }
.lkr-bench strong { font-weight: 700; }
.lkr-bench b { font-weight: 900; font-variant-numeric: tabular-nums; }
.lkr-bench small { margin-left: 4px; font-size: 11.5px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
</style>
