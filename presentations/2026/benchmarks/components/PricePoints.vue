<script setup lang="ts">
// One run's tokens, priced at each tariff the provider offers. The recorded
// run cost is the filled dot; other dots reprice the same recorded tokens.
// `shown` steps through the price points (clicks).
type Point = { label: string; cost: number; note: string; rate?: string; recorded?: boolean; muted?: boolean; step: number };
type Row = { model: string; detail: string; harness: string; score: number; tone: "teal" | "petrol"; points: Point[] };
type Rate = { model: string; tier: string; input: string; cached: string; output: string };
const props = defineProps<{ rows: Row[]; shown: number; max: number; ticks: number[]; rates?: Rate[] }>();

const X0 = 0;
const pct = (c: number) => `${(c / props.max) * 100}%`;
</script>

<template>
  <div class="pp">
    <div class="pp-axis">
      <span class="pp-legend">Current rates per million tokens:<br />input / cached / output</span>
      <div class="pp-scale">
        <span v-for="t in ticks" :key="t" :style="{ left: pct(t) }">${{ t }}</span>
      </div>
    </div>
    <div v-for="r in rows" :key="r.model" class="pp-row">
      <div class="pp-name">
        <h3 :class="`t-${r.tone}`">{{ r.model }}</h3>
        <p>{{ r.detail }} · score {{ Math.floor(r.score) }}%</p>
        <p class="pp-harness">{{ r.harness }}</p>
      </div>
      <div class="pp-track">
        <i v-for="t in ticks" :key="t" class="pp-grid" :style="{ left: pct(t) }" />
        <template v-for="p in r.points" :key="p.label">
          <div
            class="pp-pt"
            :class="[p.muted ? 't-muted' : `t-${r.tone}`, { 'is-recorded': p.recorded, 'is-on': shown >= p.step }]"
            :style="{ left: pct(p.cost) }"
          >
            <b :class="`f-${r.tone}`" />
            <strong>${{ p.cost.toFixed(2) }}</strong>
            <span>{{ p.label }}</span>
            <small>{{ p.note }}</small>
            <em v-if="p.rate">{{ p.rate }}</em>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.pp { display: flex; flex-direction: column; gap: 6px; font-family: var(--font-read); }
.pp-axis, .pp-row { display: grid; grid-template-columns: 230px 1fr; gap: 28px; }
.pp-scale { position: relative; height: 20px; border-bottom: 2px solid var(--petrol); }
.pp-scale span { position: absolute; bottom: 4px; transform: translateX(-50%); font: 600 13px/1 var(--font-read); color: var(--text-muted); font-variant-numeric: tabular-nums; }
.pp-row { align-items: center; padding: 6px 0; border-bottom: 2px solid var(--line); }
.pp-name h3 { margin: 0; font-size: 24px; white-space: nowrap; }
.pp-name p { margin: 4px 0 0; font-size: 14px; color: var(--text-muted); }
.pp-track { position: relative; height: 150px; }
.pp-grid { position: absolute; top: 0; bottom: 0; border-left: 2px solid var(--line); }
.pp-pt {
  position: absolute;
  top: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  transform: translateX(-50%);
  opacity: 0;
  transition: opacity 260ms var(--ease-ui);
  text-align: center;
  width: 150px;
  z-index: 1;
}
.pp-pt strong, .pp-pt span, .pp-pt small, .pp-pt em { background: var(--surface-page); padding: 0 4px; }
.pp-pt em { margin-top: 3px; font: 700 13px/1.2 var(--font-read); font-variant-numeric: tabular-nums; font-style: normal; color: var(--petrol); white-space: nowrap; }
.pp-legend { align-self: end; font: 600 12px/1.3 var(--font-read); color: var(--text-muted); }
.pp-pt.is-on { opacity: 1; }
.pp-pt b { width: 22px; height: 22px; border-radius: 50%; border: 3px solid currentColor; background: var(--ivory) !important; }
.pp-pt.is-recorded b { background: currentColor !important; }
.pp-pt strong { margin-top: 8px; font: 900 26px/1 var(--font-read); font-variant-numeric: tabular-nums; }
.pp-pt span { margin-top: 4px; font: 700 14px/1.2 var(--font-read); color: var(--petrol); }
.pp-pt small { font: 500 12px/1.25 var(--font-read); color: var(--text-muted); }
.t-teal { color: var(--teal); }
.t-muted { color: rgba(8, 44, 52, 0.45); }
.t-muted strong { text-decoration: line-through; text-decoration-thickness: 2px; }
.pp-pt.t-muted span { color: var(--text-muted); }
.pp-name .pp-harness { margin-top: 2px; font-size: 13px; }
.slidev-layout .pp-rates { margin: 6px 0 0; font-size: 13px; width: 100%; }
.slidev-layout .pp-rates th { padding: 5px 12px; font-size: 11px; }
.slidev-layout .pp-rates td { padding: 3px 12px; font-variant-numeric: tabular-nums; }
.pp-rates td b { font-weight: 800; margin-right: 4px; }
.t-petrol { color: var(--petrol); }
</style>
