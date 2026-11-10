<script setup lang="ts">
// Two Terminal-Bench 2.1 leaderboard rows: the published score, the score
// once atif-scan has read the row's own trial files, and what can't be seen
// in them. Data: data/leaderboard-adjust.json, extracted from the public
// Space evalstate/tb21-annotated (analysis/leaderboard-adjust/extract.py).
//
// "Adjusted" leaves out rewarded trials where another model answered some of
// the steps, and rewarded trials flagged for a reward hack.
import raw from "../data/leaderboard-adjust.json";

type Row = (typeof raw.rows)[number];
const floor1 = (v: number) => (Math.floor(v * 10) / 10).toFixed(1);
const pct = (v: number) => `${Math.floor(v * 100)}%`;

function card(r: Row) {
  const dropped = r.adjusted.fallback + r.adjusted.leads;
  const lookups = r.categories.answer_lookup;
  const facts: { big: string; text: string }[] = [];
  if (r.otherModel) facts.push({ big: `${r.otherModel.trials}`, text: `trials had steps answered by ${r.otherModel.name.replace("claude-opus-4-8", "Opus 4.8")}, not ${r.model}` });
  if (lookups && lookups.trials > 5) facts.push({ big: `${lookups.trials}`, text: `trials looked the answer up. ${lookups.rewarded} of them were rewarded` });
  facts.push({ big: pct(r.hiddenOutputShare), text: "of the output tokens are reasoning nobody can read" });
  if (r.webTrials) facts.push({ big: `${r.webTrials}`, text: "trials searched the web. The results were never recorded" });
  else if (r.cost.listOutput)
    facts.push({ big: `$${Math.floor(r.cost.fitted.output)}`, text: `per million output tokens is what the trial files charge. ${r.model} lists at $${r.cost.listOutput}` });
  return {
    id: r.id,
    title: `${r.model} · ${r.agent}`,
    published: `${floor1(r.leaderboard.accuracy)}%`,
    adjusted: `${floor1(r.adjusted.accuracy)}%`,
    segments: [
      { cls: "kept", n: r.adjusted.rewarded, label: "rewarded and clean" },
      { cls: "dropped", n: dropped, label: r.adjusted.fallback ? "another model answered" : "flagged as a reward hack" },
      { cls: "failed", n: r.trials - r.rewarded, label: "not rewarded" },
    ].map((s) => ({ ...s, width: `${(s.n / r.trials) * 100}%` })),
    facts: facts.slice(0, 3),
  };
}
const cards = raw.rows.map(card);
</script>

<template>
  <div class="fa-behind">
    <section v-for="c in cards" :key="c.id">
      <h3>{{ c.title }}</h3>
      <p class="fa-behind-score"><s>{{ c.published }}</s><i aria-hidden="true">❯</i><b>{{ c.adjusted }}</b></p>
      <div class="fa-behind-bar">
        <span v-for="s in c.segments" :key="s.cls" :class="`is-${s.cls}`" :style="{ width: s.width }"><em>{{ s.n }}</em></span>
      </div>
      <p class="fa-behind-key">
        <span v-for="s in c.segments" :key="s.cls"><i :class="`is-${s.cls}`" />{{ s.label }}</span>
      </p>
      <ul>
        <li v-for="f in c.facts" :key="f.text"><b>{{ f.big }}</b><span>{{ f.text }}</span></li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.fa-behind { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; font-family: var(--font-read); }
.fa-behind section { padding: 16px 20px 12px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.fa-behind h3 { margin: 0; font-size: 21px; }
.fa-behind-score { margin: 4px 0 10px; display: flex; align-items: baseline; gap: 12px; font-variant-numeric: tabular-nums; }
.fa-behind-score s { font: 800 26px/1 var(--font-read); color: var(--text-muted); text-decoration-thickness: 2px; }
.fa-behind-score i { font-style: normal; font-size: 18px; color: var(--petrol); }
.fa-behind-score b { font: 900 54px/1 var(--font-read); color: var(--petrol); }
.fa-behind-bar { display: flex; gap: 2px; height: 26px; }
.fa-behind-bar span { display: flex; align-items: center; min-width: 0; border-radius: 4px; overflow: hidden; }
.fa-behind-bar em { padding-left: 7px; font: 800 13px/1 var(--font-read); font-style: normal; color: var(--ivory); font-variant-numeric: tabular-nums; }
.is-kept { background: var(--teal); }
.is-dropped { background: var(--orange); }
.is-failed { background: var(--petrol-tint); }
.fa-behind-bar .is-dropped em, .fa-behind-bar .is-failed em { color: var(--petrol); }
.fa-behind-key { margin: 6px 0 0; display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; font-weight: 600; color: var(--text-muted); }
.fa-behind-key span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.fa-behind-key i { width: 11px; height: 11px; border-radius: 3px; }
.fa-behind ul { list-style: none; margin: 10px 0 0; padding: 0; }
.fa-behind li { display: grid; grid-template-columns: 74px 1fr; gap: 12px; align-items: center; margin: 0; padding: 7px 0; border-top: 2px solid var(--line); }
.fa-behind li::before { content: none; }
.fa-behind li b { font: 900 26px/1 var(--font-read); color: var(--petrol); text-align: right; font-variant-numeric: tabular-nums; }
.fa-behind li span { font-size: 15px; line-height: 1.25; color: var(--petrol); }
</style>
