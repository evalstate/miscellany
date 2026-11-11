<script setup lang="ts">
// Which tasks end on a provider safety stop, model by model. Each cell is the
// share of that model's attempts at the task that were stopped, on fast-agent
// only, so the columns compare; the attempts per task are under each name.
// Tasks are grouped by subject (our reading of the task names and
// instructions). Data: data/safety-corpus.json (analysis/safety-stops/corpus.py).
import data from "../data/safety-corpus.json";

const GROUPS: [string, string[]][] = [
  ["Exploits", ["vulnerable-secret", "break-filter-js-from-html", "fix-code-vulnerability"]],
  ["Passwords and ciphers", ["crack-7z-hash", "password-recovery", "feal-differential-cryptanalysis"]],
  ["Biology", ["protein-assembly", "dna-assembly", "dna-insert"]],
  ["Model extraction", ["model-extraction-relu-logits"]],
];
const OTHER = "No obvious subject";
const num = (n: number) => n.toLocaleString("en-GB");
const m = data.matrix;
const group = (task: string) => GROUPS.find(([, tasks]) => tasks.includes(task))?.[0] ?? OTHER;
const order = [...GROUPS.map(([g]) => g), OTHER];
const rows = m.rows
  .map((r) => ({
    task: r.task, group: group(r.task),
    cells: r.cells.map((c) => ({ ...c, pct: Math.floor((c.stops / c.trials) * 100), width: `${(c.stops / c.trials) * 100}%`, all: c.stops / c.trials >= 0.9 })),
  }))
  // keep a task when some model is stopped on it nearly every time, or two models are stopped on it often
  .filter((r) => r.cells.some((c) => c.all) || r.cells.filter((c) => c.pct >= 20).length >= 2)
  .sort((a, b) => order.indexOf(a.group) - order.indexOf(b.group))
  .map((r, i, all) => ({ ...r, first: i === 0 || all[i - 1].group !== r.group }));
// attempts per task, per model: a single number or a range
const attempts = m.models.map((_, i) => {
  const ns = m.rows.map((r) => r.cells[i].trials);
  const lo = Math.min(...ns), hi = Math.max(...ns);
  return lo === hi ? `${lo}` : `${lo}–${hi}`;
});
const others = data.clearFamilies.reduce((s, f) => s + f.trials, 0);
const fable = rows.filter((r) => r.cells[0].all);
</script>

<template>
  <p class="sfm-lead"><b>{{ data.stops }}</b> of {{ num(data.trials) }} trials ended on a provider safety stop: <b>OpenAI {{ data.labs.OpenAI }}</b>, <b>Anthropic {{ data.labs.Anthropic }}</b>, every other lab <b>0</b> in {{ num(others) }}.</p>
  <div class="sfm">
    <div class="sfm-row sfm-groups"><span /><span /><b style="grid-column: span 4">Anthropic</b><b style="grid-column: span 4">OpenAI GPT</b></div>
    <div class="sfm-row sfm-head"><span /><span>share of attempts stopped</span><b v-for="name in m.models" :key="name">{{ name.replace("GPT-", "") }}</b></div>
    <div class="sfm-row sfm-n"><span /><span>attempts per task</span><em v-for="(a, i) in attempts" :key="i">{{ a }}</em></div>
    <div v-for="r in rows" :key="r.task" class="sfm-row" :class="{ first: r.first }">
      <i>{{ r.first ? r.group : "" }}</i>
      <strong>{{ r.task }}</strong>
      <div v-for="(c, i) in r.cells" :key="i" class="sfm-cell" :class="{ all: c.all, none: !c.stops }">
        <span><u :style="{ width: c.width }" /></span><em>{{ c.stops ? `${c.pct}%` : "" }}</em>
      </div>
    </div>
  </div>
  <p class="lk-note sfm-note"><b>Fable 5 is stopped every time on {{ fable.length }} tasks,</b> {{ fable.filter((r) => r.group === OTHER).length }} of them with no obvious subject.</p>
</template>

<style scoped>
.sfm-lead { margin: -4px 0 6px; font: 500 16px/1.3 var(--font-read); color: var(--petrol); }
.sfm-lead b { font-weight: 900; font-variant-numeric: tabular-nums; }
.sfm { font-family: var(--font-read); }
.sfm-row { display: grid; grid-template-columns: 124px 212px repeat(8, 1fr); column-gap: 10px; align-items: center; height: 16px; }
.sfm-row > :nth-child(7) { margin-left: 10px; }
.sfm-row.first { margin-top: 4px; }
.sfm-groups { height: 18px; }
.sfm-groups b { font-size: 13px; font-weight: 800; color: var(--petrol); border-bottom: var(--border); }
.sfm-groups > :nth-child(7) { margin-left: 0; }
.sfm-groups > :nth-child(4) { margin-left: 10px; }
.sfm-head { height: 18px; }
.sfm-head span, .sfm-n span { font-size: 11px; color: var(--text-muted); text-align: right; white-space: nowrap; }
.sfm-head b { font-size: 12px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.sfm-n { height: 14px; }
.sfm-n em { font-style: normal; font-size: 11px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.sfm-row > i { font-style: normal; font-size: 12px; font-weight: 800; color: var(--petrol); white-space: nowrap; }
.sfm-row strong { font: 500 11px var(--font-code); color: var(--petrol); text-align: right; white-space: nowrap; }
.sfm-cell { display: grid; grid-template-columns: 1fr 34px; column-gap: 5px; align-items: center; }
.sfm-cell span { height: 11px; border-radius: 3px; background: var(--ivory-deep); overflow: hidden; }
.sfm-cell u { display: block; height: 100%; background: var(--orange); text-decoration: none; }
.sfm-cell em { font-style: normal; font-size: 12px; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.sfm-cell.all em { font-weight: 900; color: var(--petrol); }
.sfm-note { margin-top: 6px; font-size: 14px; }
</style>
