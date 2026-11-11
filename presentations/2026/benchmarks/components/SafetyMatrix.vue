<script setup lang="ts">
// Which tasks end on a provider safety stop, model by model. Each cell is the
// stops out of that model's attempts at the task, on fast-agent only, so the
// columns compare. Data: data/safety-corpus.json (analysis/safety-stops/corpus.py).
import data from "../data/safety-corpus.json";

const m = data.matrix;
const groups = [{ name: "Anthropic", span: 4 }, { name: "OpenAI", span: 4 }];
const rows = m.rows.map((r) => ({
  task: r.task,
  cells: r.cells.map((c) => ({ ...c, width: `${(c.stops / c.trials) * 100}%`, all: c.stops > 0 && c.stops / c.trials >= 0.9 })),
}));
const f = data.fable;
</script>

<template>
  <div class="sfm">
    <div class="sfm-row sfm-groups"><span /><b v-for="g in groups" :key="g.name" :style="{ gridColumn: `span ${g.span}` }">{{ g.name }}</b></div>
    <div class="sfm-row sfm-head"><span>stops, of attempts</span><b v-for="name in m.models" :key="name">{{ name }}</b></div>
    <div v-for="r in rows" :key="r.task" class="sfm-row">
      <strong>{{ r.task }}</strong>
      <div v-for="(c, i) in r.cells" :key="i" class="sfm-cell" :class="{ none: !c.stops, all: c.all }">
        <template v-if="c.stops"><i><span :style="{ width: c.width }" /></i><em>{{ c.stops }}/{{ c.trials }}</em></template>
        <template v-else>·</template>
      </div>
    </div>
  </div>
  <p class="lk-note sfm-note"><b>Fable 5 never stops on Claude Code: Opus 4.8 answers instead,</b> {{ f.swappedTrials }} trials on {{ f.swappedTasks }} tasks. {{ f.overlap }} of its {{ f.stopTasks }} fast-agent stop tasks are among them.</p>
</template>

<style scoped>
.sfm { font-family: var(--font-read); }
.sfm-row { display: grid; grid-template-columns: 218px repeat(8, 1fr); column-gap: 8px; align-items: center; height: 15.5px; }
.sfm-row > div:nth-of-type(5), .sfm-row > b:nth-of-type(5) { margin-left: 8px; }
.sfm-groups { height: 18px; }
.sfm-groups b { font-size: 13px; font-weight: 800; color: var(--petrol); border-bottom: var(--border); }
.sfm-groups b:nth-of-type(2) { margin-left: 8px; }
.sfm-head { height: 20px; margin-bottom: 2px; }
.sfm-head span { font-size: 11.5px; color: var(--text-muted); text-align: right; }
.sfm-head b { font-size: 12px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.sfm-row strong { font: 500 11.5px var(--font-code); color: var(--petrol); text-align: right; white-space: nowrap; }
.sfm-cell { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--text-muted); }
.sfm-cell.none { padding-left: 12px; }
.sfm-cell i { flex: 0 0 30px; height: 10px; border-radius: 3px; background: var(--ivory-deep); overflow: hidden; }
.sfm-cell i span { display: block; height: 100%; min-width: 2px; background: var(--orange); }
.sfm-cell em { font-style: normal; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.sfm-cell.all em { font-weight: 900; color: var(--petrol); }
.sfm-note { margin-top: 6px; font-size: 14px; white-space: nowrap; }
</style>
