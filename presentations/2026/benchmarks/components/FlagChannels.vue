<script setup lang="ts">
// Where a trial is flagged: in its reasoning text, or in its tool calls. Split
// by whether the trace records any reasoning text at all, to show what a run
// with hidden reasoning can and can't be flagged for.
// Data: data/awareness-channels.json (analysis/awareness-channels/extract.py).
import data from "../data/awareness-channels.json";

const num = (n: number) => n.toLocaleString("en-GB");
const pct = (a: number, b: number) => Math.floor((a / b) * 1000) / 10;
const whole = (a: number, b: number) => Math.floor((a / b) * 100);
const MAX = 40; // per cent, the full width of a bar
const MIN = 400; // families with fewer trials are folded into "Others"
const row = (f: { family: string; trials: number; tool: number; reasoning: number }) => ({
  name: f.family, trials: f.trials, tool: pct(f.tool, f.trials), reasoning: pct(f.reasoning, f.trials),
  toolW: `${(f.tool / f.trials / MAX) * 10000}%`, reasoningW: `${(f.reasoning / f.trials / MAX) * 10000}%`,
});
const fold = (recorded: boolean) => {
  const mine = data.families.filter((f) => f.recorded === recorded);
  const big = mine.filter((f) => f.trials >= MIN && f.family !== "other");
  const rest = mine.filter((f) => !big.includes(f));
  const sum = (k: "trials" | "tool" | "reasoning") => rest.reduce((s, f) => s + f[k], 0);
  const rows = big.map(row);
  if (rest.length) rows.push(row({ family: "Others", trials: sum("trials"), tool: sum("tool"), reasoning: sum("reasoning") }));
  return rows;
};
const withText = fold(true).sort((a, b) => b.reasoning - a.reasoning);
const without = fold(false);
const rec = data.recorded, hid = data.notRecorded, yes = data.toolGivenReasoning, no = data.toolGivenNoReasoning;
</script>

<template>
  <div class="fch">
    <div class="fch-chart">
      <div class="fch-row fch-head"><span /><b><i class="fch-r" />Flagged in reasoning</b><b><i class="fch-t" />Flagged in a tool call</b></div>
      <h3>Reasoning text recorded <small>{{ num(rec.trials) }} trials</small></h3>
      <div v-for="f in withText" :key="f.name" class="fch-row">
        <strong>{{ f.name }} <small>{{ num(f.trials) }}</small></strong>
        <div class="fch-bar"><span class="fch-r" :style="{ width: f.reasoningW }" /><em>{{ f.reasoning.toFixed(1) }}%</em></div>
        <div class="fch-bar"><span class="fch-t" :style="{ width: f.toolW }" /><em>{{ f.tool.toFixed(1) }}%</em></div>
      </div>
      <h3>No reasoning text <small>{{ num(hid.trials) }} trials</small></h3>
      <div v-for="f in without" :key="f.name" class="fch-row">
        <strong>{{ f.name }} <small>{{ num(f.trials) }}</small></strong>
        <div class="fch-bar fch-blank">nothing to read</div>
        <div class="fch-bar"><span class="fch-t" :style="{ width: f.toolW }" /><em>{{ f.tool.toFixed(1) }}%</em></div>
      </div>
    </div>
    <div class="fch-facts">
      <section>
        <p class="fch-big"><b>{{ whole(yes.tool, yes.trials) }}%</b> against <b>{{ whole(no.tool, no.trials) }}%</b></p>
        <p>A trial flagged in its reasoning has a flagged tool call {{ whole(yes.tool, yes.trials) }}% of the time. Without the reasoning flag, {{ whole(no.tool, no.trials) }}%.</p>
      </section>
      <section>
        <p class="fch-big"><b>{{ num(rec.reasoning - rec.both) }}</b> trials</p>
        <p>are flagged in reasoning and nowhere in their tool calls. Hide the text and they read as clean.</p>
      </section>
      <section>
        <p class="fch-big"><b>{{ whole(hid.trials, data.trials) }}%</b> of trials</p>
        <p>record no reasoning text. {{ num(hid.tool) }} of them have a flagged tool call, with nothing beside it to say why.</p>
      </section>
    </div>
  </div>
</template>

<style scoped>
.fch { display: grid; grid-template-columns: 1.55fr 1fr; gap: 22px; align-items: start; font-family: var(--font-read); }
.fch-row { display: grid; grid-template-columns: 128px 1fr 1fr; column-gap: 14px; align-items: center; height: 17px; }
.fch-head { height: 20px; }
.fch-head b { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.fch-head i { width: 13px; height: 13px; border-radius: 3px; }
.fch h3 { margin: 7px 0 3px; padding-bottom: 2px; border-bottom: var(--border); font-size: 15px; }
.fch h3 small { margin-left: 6px; font: 500 12px var(--font-read); letter-spacing: 0; color: var(--text-muted); }
.fch-row strong { font-size: 13px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.fch-row strong small { margin-left: 4px; font-size: 11.5px; font-weight: 500; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.fch-bar { display: flex; align-items: center; height: 12px; }
.fch-bar span { height: 100%; min-width: 2px; border-radius: 3px; }
.fch-r { background: var(--teal); }
.fch-t { background: var(--petrol); }
.fch-bar em { margin-left: 6px; font-style: normal; font-size: 12px; line-height: 1; font-weight: 700; color: var(--petrol); font-variant-numeric: tabular-nums; }
.fch-blank { padding-left: 8px; border-radius: 3px; background: var(--ivory-deep); font-size: 11.5px; line-height: 12px; color: var(--text-muted); }
.fch-facts { display: grid; gap: 10px; }
.fch-facts section { padding: 10px 16px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.fch-facts p { margin: 0; font-size: 14px; line-height: 1.35; color: var(--petrol); }
.fch-facts p.fch-big { margin-bottom: 3px; font-size: 15px; }
.fch-facts p.fch-big b { font: 900 30px/1 var(--font-voice); font-variant-numeric: tabular-nums; }
</style>
