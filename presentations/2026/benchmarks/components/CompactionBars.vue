<script setup lang="ts">
// "Change, then better": one setting moves (the compaction point: how full
// the context gets before fast-agent summarises it), and the two results
// follow. Grey is before, amber is after (amber because the change is a
// fast-agent setting), teal marks the improvement.
//
// 19-task TB 4 subset, 95 trials per run. Figures are copied from the
// fast-agent docs chart (docs/benchmark_data/tb4/compaction.py, 4 Nov 2026):
// score is rewards ÷ 95, cost is fast-agent's recorded cost at list rates.
type Run = { point: string; tokens: number; score: number; cost: number };
const TRIALS = 95;
const pairs: { model: string; late: Run; early: Run }[] = [
  {
    model: "GLM-5.3 Flash",
    late: { point: "1M", tokens: 1_000_000, score: 35.8, cost: 26.15 },
    early: { point: "200k", tokens: 200_000, score: 37.9, cost: 20.06 },
  },
  {
    model: "Grok 4.7",
    late: { point: "425k", tokens: 425_000, score: 40.0, cost: 721.3 },
    early: { point: "190k", tokens: 190_000, score: 42.1, cost: 604.99 },
  },
];
const pct = (v: number) => `${v}%`;
const rows = pairs.map((p) => ({
  model: p.model,
  from: p.late.point,
  to: p.early.point,
  cut: pct((p.early.tokens / p.late.tokens) * 100),
  score: {
    before: `${Math.floor(p.late.score)}%`,
    after: `${Math.floor(p.early.score)}%`,
    delta: `${Math.floor(p.early.score - p.late.score)} points`,
    wBefore: pct((p.late.score / 60) * 100),
    wAfter: pct((p.early.score / 60) * 100),
  },
  cost: {
    before: `$${(p.late.cost / TRIALS).toFixed(2)}`,
    after: `$${(p.early.cost / TRIALS).toFixed(2)}`,
    delta: `${Math.floor(((p.late.cost - p.early.cost) / p.late.cost) * 100)}%`,
    wBefore: "100%",
    wAfter: pct((p.early.cost / p.late.cost) * 100),
  },
}));
</script>

<template>
  <div class="fa-cmp">
    <div class="fa-cmp-head fa-cmp-head--change">
      <strong>Compaction point</strong>
      <span>how full the context gets before it is summarised</span>
    </div>
    <div class="fa-cmp-head"><strong>Score</strong></div>
    <div class="fa-cmp-head"><strong>Cost per trial</strong></div>

    <template v-for="r in rows" :key="r.model">
      <div class="fa-cmp-change">
        <h3>{{ r.model }}</h3>
        <p class="fa-cmp-points"><s>{{ r.from }}</s><i aria-hidden="true">❯</i><b>{{ r.to }}</b></p>
        <div class="fa-cmp-meter" :aria-label="`Compaction point moved from ${r.from} to ${r.to}`">
          <span :style="{ width: r.cut }" />
        </div>
      </div>
      <div v-for="(m, k) in [r.score, r.cost]" :key="k" class="fa-cmp-result">
        <p class="fa-cmp-delta">{{ k === 0 ? "▲" : "▼" }} {{ m.delta }}</p>
        <div class="fa-cmp-bars">
          <div><u><span :style="{ width: m.wBefore }" /></u><em>{{ m.before }}</em></div>
          <div class="is-after"><u><span :style="{ width: m.wAfter }" /></u><em>{{ m.after }}</em></div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.fa-cmp {
  display: grid;
  grid-template-columns: 1.35fr 1fr 1fr;
  column-gap: 36px;
  font-family: var(--font-read);
}
.fa-cmp-head { padding-bottom: 8px; border-bottom: var(--border); }
.fa-cmp-head strong { font: 900 26px/1.1 var(--font-voice); letter-spacing: var(--track-voice); color: var(--petrol); }
.fa-cmp-head span { display: block; margin-top: 2px; font-size: 13px; font-weight: 500; color: var(--text-muted); }
.fa-cmp-head:not(.fa-cmp-head--change) { display: flex; align-items: flex-end; }

.fa-cmp-change, .fa-cmp-result { padding: 14px 0 16px; border-bottom: 2px solid var(--line); }
.fa-cmp-change h3 { margin: 0 0 2px; font: 800 17px/1.2 var(--font-read); letter-spacing: 0; color: var(--text-muted); }
.fa-cmp-points { margin: 0 0 8px; display: flex; align-items: baseline; gap: 12px; font-variant-numeric: tabular-nums; }
.fa-cmp-points s { font: 700 28px/1 var(--font-read); color: var(--text-muted); text-decoration-thickness: 2px; }
.fa-cmp-points i { font-style: normal; font-size: 20px; color: var(--petrol); }
.fa-cmp-points b { font: 900 52px/1 var(--font-voice); letter-spacing: var(--track-voice); color: var(--petrol); }
/* the whole track is the old compaction point; the amber part is the new one */
.fa-cmp-meter { height: 22px; border-radius: var(--radius-sm); background: var(--petrol-tint); }
.fa-cmp-meter span { display: block; height: 100%; border: var(--border); border-radius: var(--radius-sm); background: var(--amber); }

.fa-cmp-result { display: flex; flex-direction: column; justify-content: flex-end; }
.fa-cmp-delta { margin: 0 0 10px; font: 900 34px/1 var(--font-voice); letter-spacing: var(--track-voice); color: var(--teal); white-space: nowrap; }
.fa-cmp-bars { display: grid; gap: 5px; }
/* each row: a track the bar is measured against, then a fixed column for the value */
.fa-cmp-bars div { display: grid; grid-template-columns: 1fr 62px; align-items: center; gap: 10px; height: 22px; }
.fa-cmp-bars u { display: block; height: 100%; text-decoration: none; }
.fa-cmp-bars span { display: block; height: 100%; border-radius: var(--radius-sm); background: var(--petrol-tint); }
.fa-cmp-bars em { font: 700 16px/1 var(--font-read); font-style: normal; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.fa-cmp-bars .is-after span { border: var(--border); background: var(--amber); }
.fa-cmp-bars .is-after em { font-weight: 900; font-size: 19px; color: var(--petrol); }
</style>
