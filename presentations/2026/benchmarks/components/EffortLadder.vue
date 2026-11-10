<script setup lang="ts">
// One model, five reasoning levels, the same 445 trials. Columns show score,
// reasoning tokens per trial and run cost; the strip below shows the lowest
// level at which each of the 89 tasks is solved (3 or more of 5 attempts).
type Level = { effort: string; score: number; reasoning: number; cost: number };
type Bucket = { label: string; tasks: number; tone: string };
const props = withDefaults(defineProps<{ levels?: Level[]; buckets?: Bucket[]; pick?: string; part?: "levels" | "strip" }>(), { levels: () => [], buckets: () => [], pick: "", part: "levels" });
const maxR = Math.max(1, ...props.levels.map((l) => l.reasoning));
const totalTasks = props.buckets.reduce((a, b) => a + b.tasks, 0);
const fmt = (n: number) => n.toLocaleString("en-GB");
</script>

<template>
  <div class="el">
    <div v-if="part === 'levels'" class="el-levels">
      <div v-for="l in levels" :key="l.effort" class="el-col" :class="{ 'is-pick': l.effort === pick }">
        <strong>{{ Math.floor(l.score) }}%</strong>
        <div class="el-barwrap"><i :style="{ height: `${(l.reasoning / maxR) * 100}%` }" /></div>
        <span class="el-tok">{{ fmt(l.reasoning) }}</span>
        <b>{{ l.effort }}</b>
        <small>${{ l.cost.toFixed(2) }} a run</small>
      </div>
    </div>
    <p v-if="part === 'levels'" class="el-key">Score, and mean reasoning tokens per trial</p>
    <div v-if="part === 'strip'" class="el-strip">
      <p>Lowest level that solves each task (3+ of 5 attempts), 89 tasks</p>
      <div class="el-bar">
        <span v-for="b in buckets" :key="b.label" :class="`el-b--${b.tone}`" :style="{ flex: b.tasks }">
          <b>{{ b.tasks }}</b> {{ b.label }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.el { display: flex; flex-direction: column; gap: 12px; font-family: var(--font-read); }
.el-levels { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; align-items: end; }
.el-col { display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 10px 6px 8px; border-radius: var(--radius-lg); border: 2px solid transparent; }
.el-col.is-pick { border-color: var(--petrol); background: var(--paper); }
.el-col strong { font: 900 34px/1 var(--font-read); color: var(--petrol); font-variant-numeric: tabular-nums; }
.el-barwrap { position: relative; width: 56px; height: 140px; display: flex; align-items: flex-end; border-bottom: 2px solid var(--petrol); }
.el-barwrap i { display: block; width: 100%; background: var(--teal); border-radius: var(--radius-sm) var(--radius-sm) 0 0; }
.el-tok { font: 700 14px/1 var(--font-read); color: var(--text-muted); font-variant-numeric: tabular-nums; }
.el-col b { font: 900 20px/1 var(--font-voice); letter-spacing: var(--track-voice); color: var(--petrol); margin-top: 4px; }
.el-col small { font-size: 13px; color: var(--text-muted); }
.el-key { margin: -8px 0 0; font-size: 13px; color: var(--text-muted); }
.el-strip { margin-top: 14px; }
.el-strip p { margin: 0 0 6px; font-size: 14px; color: var(--text-muted); }
.el-bar { display: flex; gap: 3px; height: 40px; }
.el-bar span { display: flex; align-items: center; justify-content: center; gap: 5px; min-width: 0; border-radius: var(--radius-sm); font-size: 13px; white-space: nowrap; overflow: hidden; }
.el-bar b { font-weight: 900; font-size: 16px; }
.el-b--low { background: var(--teal); color: var(--ivory); }
.el-b--medium { background: var(--petrol); color: var(--ivory); }
.el-b--higher { background: var(--ivory-deep); color: var(--petrol); border: 2px solid var(--petrol); }
.el-b--never { background: var(--ivory-deep); color: var(--text-muted); }
</style>
