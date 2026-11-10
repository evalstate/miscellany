<script setup lang="ts">
// Reward rate with and without Stockfish, per model family, for the same
// fast-agent chess-best-move trials as ChessSankey. Shows whether the engine
// effect is the engine or the mix of models that choose to use it.
import flows from "../data/chess-flows.json";

const W = 868;
const ROW = 26;
const X0 = 170;
const X1 = 640;
const top = 30;
const H = top + flows.models.length * ROW + 6;
const x = (rate: number) => X0 + (X1 - X0) * rate;
const rate = ([n, won]: number[]) => won / n;
const ticks = [0, 0.25, 0.5, 0.75, 1];
</script>

<template>
  <svg class="fa-dumb" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Reward rate with and without Stockfish by model">
    <g v-for="t in ticks" :key="t">
      <line :x1="x(t)" :x2="x(t)" :y1="top - 6" :y2="H - 4" class="fa-dumb-grid" />
      <text :x="x(t)" y="14" text-anchor="middle" class="fa-dumb-tick">{{ t * 100 }}%</text>
    </g>
    <circle :cx="X1 + 30" cy="9" r="6" class="fa-dumb-with" />
    <text :x="X1 + 42" y="14" class="fa-dumb-tick">Stockfish</text>
    <circle :cx="X1 + 140" cy="9" r="5" class="fa-dumb-without" />
    <text :x="X1 + 151" y="14" class="fa-dumb-tick">without</text>

    <g v-for="(m, i) in flows.models" :key="m.model" :transform="`translate(0 ${top + i * ROW + ROW / 2})`">
      <text :x="X0 - 14" y="5" text-anchor="end" class="fa-dumb-label">{{ m.model }}</text>
      <line :x1="x(rate(m.without))" :x2="x(rate(m.withEngine))" class="fa-dumb-bar" />
      <circle :cx="x(rate(m.without))" r="6" class="fa-dumb-without"><title>{{ m.model }} without Stockfish: {{ m.without[1] }} of {{ m.without[0] }}</title></circle>
      <circle :cx="x(rate(m.withEngine))" r="7" class="fa-dumb-with"><title>{{ m.model }} with Stockfish: {{ m.withEngine[1] }} of {{ m.withEngine[0] }}</title></circle>
      <text :x="X1 + 24" y="5" class="fa-dumb-num">{{ m.withEngine[1] }} of {{ m.withEngine[0] }}</text>
      <text :x="X1 + 134" y="5" class="fa-dumb-num">{{ m.without[1] }} of {{ m.without[0] }}</text>
    </g>
  </svg>
</template>

<style scoped>
.fa-dumb {
  display: block;
  width: 100%;
  height: auto;
  font-family: var(--font-read);
}
.fa-dumb-grid { stroke: var(--line); stroke-width: 2; }
.fa-dumb-tick { font-size: 13px; font-weight: 600; fill: var(--text-muted); }
.fa-dumb-label { font-size: 15px; font-weight: 700; fill: var(--petrol); }
.fa-dumb-num { font-size: 14px; font-weight: 500; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.fa-dumb-bar { stroke: var(--petrol); stroke-width: 2; }
.fa-dumb-with { fill: var(--teal); stroke: var(--ivory); stroke-width: 2; }
.fa-dumb-without { fill: var(--ivory); stroke: var(--petrol); stroke-width: 2; }
</style>
