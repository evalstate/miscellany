<script setup lang="ts">
// Headline price vs what a full run actually cost. Opens on the list prices
// (per million tokens); one click reveals the recorded run cost and the
// tokens behind it. Prices are the card-packs pricing catalog rates in effect
// at each run date, and they reproduce Harbor's recorded totals.
import { computed } from "vue";

type Side = {
  name: string;
  effort: string;
  tone: "teal" | "petrol";
  price: { input: number; cached: number; output: number };
  run: { cost: number; input: number; cached: number; output: number; score: number };
};
const props = defineProps<{ models: [Side, Side]; reveal: boolean; benchmark: string }>();

const money = (v: number) => `$${v < 0.01 ? v.toFixed(3) : v.toFixed(2)}`;
const tokensM = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)}B` : `${Math.floor(v)}M`);
const maxCost = computed(() => Math.max(...props.models.map((m) => m.run.cost)));
const maxIn = computed(() => Math.max(...props.models.map((m) => m.run.input)));
const maxOut = computed(() => Math.max(...props.models.map((m) => m.run.output)));
</script>

<template>
  <div class="lvr" :class="{ 'is-revealed': reveal }">
    <div class="lvr-row lvr-row--head">
      <span />
      <h3 v-for="m in models" :key="m.name" :class="`t-${m.tone}`">{{ m.name }} <small>{{ m.effort }}</small></h3>
    </div>

    <div class="lvr-row">
      <p class="lvr-label">List price<br /><small>per million tokens</small></p>
      <div v-for="m in models" :key="m.name" class="lvr-cell">
        <div class="lvr-price">
          <span><b>{{ money(m.price.input) }}</b> input</span>
          <span><b>{{ money(m.price.cached) }}</b> cached</span>
          <span><b>{{ money(m.price.output) }}</b> output</span>
        </div>
      </div>
    </div>

    <div class="lvr-reveal">
      <div class="lvr-row">
        <p class="lvr-label">Tokens used<br /><small>{{ benchmark }}</small></p>
        <div v-for="m in models" :key="m.name" class="lvr-cell lvr-tokens">
          <span><b>{{ tokensM(m.run.input) }}</b> input ({{ Math.floor((m.run.cached / m.run.input) * 100) }}% cached)</span>
          <i :class="`f-${m.tone}`" :style="{ width: `${(m.run.input / maxIn) * 100}%` }" />
          <span><b>{{ m.run.output.toFixed(1) }}M</b> output</span>
          <i :class="`f-${m.tone}`" :style="{ width: `${(m.run.output / maxOut) * 100}%` }" />
        </div>
      </div>
      <div class="lvr-row lvr-row--cost">
        <p class="lvr-label">Run cost<br /><small>{{ benchmark }}</small></p>
        <div v-for="m in models" :key="m.name" class="lvr-cell">
          <strong :class="`t-${m.tone}`">${{ Math.floor(m.run.cost) }}</strong>
          <i class="lvr-costbar" :class="`f-${m.tone}`" :style="{ width: `${(m.run.cost / maxCost) * 100}%` }" />
          <small>score {{ Math.floor(m.run.score) }}%</small>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.lvr { display: flex; flex-direction: column; gap: 10px; font-family: var(--font-read); }
.lvr-row { display: grid; grid-template-columns: 170px 1fr 1fr; gap: 36px; align-items: start; }
.lvr-row + .lvr-row, .lvr-reveal { border-top: 2px solid var(--line); padding-top: 10px; }
.lvr-reveal { display: flex; flex-direction: column; gap: 10px; opacity: 0; transform: translateY(8px); transition: opacity 260ms var(--ease-ui), transform 260ms var(--ease-ui); }
.is-revealed .lvr-reveal { opacity: 1; transform: none; }
.lvr-row--head h3 { margin: 0; font-size: 26px; white-space: nowrap; }
.lvr-row--head small { font: 600 16px/1 var(--font-read); color: var(--text-muted); letter-spacing: 0; }
.lvr-label { margin: 0; font: 700 16px/1.25 var(--font-read); color: var(--petrol); }
.lvr-label small { font-weight: 500; font-size: 13px; color: var(--text-muted); }
.lvr-cell { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.lvr-price { display: flex; gap: 22px; font-size: 14px; color: var(--text-muted); }
.lvr-price b { display: block; font: 900 28px/1.05 var(--font-read); color: var(--petrol); font-variant-numeric: tabular-nums; }
.lvr-bars { display: flex; flex-direction: column; gap: 4px; }
.lvr-tokens i { display: block; height: 8px; border-radius: var(--radius-sm); margin-bottom: 2px; }
.lvr-tokens span { font-size: 14px; color: var(--text-muted); }
.lvr-tokens b { font: 900 20px/1 var(--font-read); color: var(--petrol); font-variant-numeric: tabular-nums; margin-right: 4px; }
.lvr-row--cost strong { font: 900 46px/1 var(--font-read); font-variant-numeric: tabular-nums; }
.lvr-costbar { display: block; height: 14px; border-radius: var(--radius-sm); }
.lvr-row--cost small { font-size: 13px; color: var(--text-muted); }
.t-teal { color: var(--teal); }
.t-petrol { color: var(--petrol); }
.f-teal { background: var(--teal); }
.f-petrol { background: var(--petrol); }
</style>
