<script setup lang="ts">
// DeepSWE has no network. This grid shows, per model and effort setting, the
// share of trials that tried anyway: asked for the task's upstream repository
// or dug through git objects outside the branch history. The scan records the
// attempt, not what came back. Data: data/deepswe-attempts.json
// (analysis/answer-lookup/extract_deepswe.py, public deepswe-annotated Space).
import data from "../data/deepswe-attempts.json";

type Cell = { effort: string; trials: number; attempt: number; rewarded: number; attemptRewarded: number } | null;
const pct = (a: number, b: number) => Math.floor((a / b) * 100);
const shade = (p: number) => (p >= 40 ? 4 : p >= 20 ? 3 : p >= 10 ? 2 : p >= 4 ? 1 : 0);
const grid = data.grid.map((g) => ({
  model: g.model,
  cells: (g.cells as Cell[]).map((c) => (c ? { p: pct(c.attempt, c.trials), tip: `${c.attempt} of ${c.trials}` } : null)),
}));
const t = data.total;
const pairs = ["DeepSeek V4 Pro", "GPT-5.6 Terra", "GPT-5.6 Sol", "GPT-5.6 Luna"].map((model) => {
  const r = data.rows.find((x) => x.model === model && x.effort === "max")!;
  return { model, tried: pct(r.attemptRewarded, r.attempt), not: pct(r.rewarded - r.attemptRewarded, r.trials - r.attempt) };
});
</script>

<template>
  <div class="dsa">
    <div class="dsa-grid">
      <div class="dsa-head"><span /><em v-for="e in data.efforts" :key="e">{{ e }}</em></div>
      <div v-for="g in grid" :key="g.model" class="dsa-row">
        <strong>{{ g.model }}</strong>
        <template v-for="(c, i) in g.cells" :key="i">
          <span v-if="c" :class="`s${shade(c.p)}`" :title="c.tip">{{ c.p }}%</span>
          <span v-else class="none" />
        </template>
      </div>
    </div>
    <div class="dsa-side">
      <p class="dsa-lead"><b>{{ t.attempt.toLocaleString("en-GB") }}</b> of {{ t.trials.toLocaleString("en-GB") }} trials try: {{ pct(t.attempt, t.trials) }}%.</p>
      <p>{{ t.upstream.toLocaleString("en-GB") }} ask for the task's upstream repository. {{ t.git.toLocaleString("en-GB") }} dig through git objects outside the history.</p>
      <p>One run checked by hand, 27 flags: <b>19</b> real requests, <b>17</b> refused by the proxy, 2 unclear, <b>0</b> brought anything back. 8 were not requests.</p>
      <table>
        <thead><tr><th>Pass rate at max effort</th><th>tried</th><th>didn't</th></tr></thead>
        <tbody><tr v-for="p in pairs" :key="p.model"><td>{{ p.model }}</td><td>{{ p.tried }}%</td><td>{{ p.not }}%</td></tr></tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.dsa { display: grid; grid-template-columns: 1.15fr 1fr; gap: 30px; align-items: start; font-family: var(--font-read); }
.dsa-head, .dsa-row { display: grid; grid-template-columns: 132px repeat(5, 1fr); gap: 4px; align-items: center; margin-bottom: 4px; }
.dsa-head em { font-style: normal; font-size: 12px; font-weight: 700; color: var(--text-muted); text-align: center; }
.dsa-row strong { font-size: 14px; font-weight: 700; color: var(--petrol); text-align: right; padding-right: 6px; white-space: nowrap; }
.dsa-row span { display: grid; place-items: center; height: 26px; border-radius: 4px; font-size: 14px; font-weight: 800; font-variant-numeric: tabular-nums; }
.dsa-row .s0 { background: var(--ivory-deep); color: var(--text-muted); }
.dsa-row .s1 { background: color-mix(in srgb, var(--orange) 22%, var(--paper)); color: var(--petrol); }
.dsa-row .s2 { background: color-mix(in srgb, var(--orange) 48%, var(--paper)); color: var(--petrol); }
.dsa-row .s3 { background: color-mix(in srgb, var(--orange) 74%, var(--paper)); color: var(--petrol); }
.dsa-row .s4 { background: var(--orange); color: var(--petrol); }
.dsa-row .none { background: transparent; }
.dsa-side p { margin: 0 0 9px; font-size: 16px; line-height: 1.35; color: var(--petrol); }
.dsa-side p.dsa-lead { font-size: 19px; }
.dsa-side p b { font-weight: 800; }
.dsa-lead b { font: 900 34px/1 var(--font-voice); margin-right: 4px; font-variant-numeric: tabular-nums; }
.dsa-side table { width: 100%; margin-top: 4px; border-collapse: collapse; font-size: 14px; }
.dsa-side th, .dsa-side td { padding: 3px 8px; border-bottom: 2px solid var(--line); text-align: right; font-variant-numeric: tabular-nums; background: transparent; }
.dsa-side th:first-child, .dsa-side td:first-child { text-align: left; }
.dsa-side th { font-size: 12px; font-weight: 700; color: var(--text-muted); text-transform: none; letter-spacing: 0; }
</style>
