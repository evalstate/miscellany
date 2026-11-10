<script setup lang="ts">
// A short list of reward rates: label, a bar as long as the rate, and the
// count behind it ("45 of 45"). Bars are amber because these are fast-agent
// trials; a row marked `base` is the comparison line and is drawn in the
// neutral tint.
defineProps<{ rows: { label: string; won: number; n: number; base?: boolean }[] }>();
const pct = (won: number, n: number) => Math.floor((won / n) * 100);
</script>

<template>
  <div class="fa-rates">
    <div v-for="r in rows" :key="r.label" class="fa-rates-row" :class="{ base: r.base }">
      <strong>{{ r.label }}<em>{{ r.won }} of {{ r.n }} trials</em></strong>
      <div class="fa-rates-track"><span :style="{ width: `${Math.max(1, (r.won / r.n) * 100)}%` }" /></div>
      <b>{{ pct(r.won, r.n) }}%</b>
    </div>
  </div>
</template>

<style scoped>
.fa-rates { display: grid; gap: 9px; font-family: var(--font-read); }
.fa-rates-row { display: grid; grid-template-columns: var(--rates-label, 176px) 1fr 56px; column-gap: 12px; align-items: center; }
.fa-rates-row strong { font-size: 16px; font-weight: 700; line-height: 1.15; color: var(--petrol); }
.fa-rates-track { height: 26px; border-radius: var(--radius-sm); background: var(--ivory-deep); }
.fa-rates-track span { display: block; height: 100%; border: var(--border); border-radius: var(--radius-sm); background: var(--amber); }
.fa-rates-row.base .fa-rates-track span { border-color: var(--petrol-tint); background: var(--petrol-tint); }
.fa-rates-row b { font-size: 22px; font-weight: 900; color: var(--petrol); text-align: right; font-variant-numeric: tabular-nums; }
.fa-rates-row em { display: block; font-style: normal; font-size: 12px; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
</style>
