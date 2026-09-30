<script setup lang="ts">
// The receipt that travels with every number: n, date, model and harness
// version (design rule). Missing fields render as "TBC" so gaps stay visible.
// A source-only receipt (e.g. configuration data, not a run) omits the run fields.
import { computed } from "vue";

const props = defineProps<{ n?: string | number; date?: string; model?: string; harness?: string; source?: string; mark?: string }>();

const isRun = computed(() => [props.n, props.date, props.model, props.harness].some((v) => v !== undefined) || !props.source);
const parts = computed(() => !isRun.value ? [] : [
  ["n", props.n],
  ["date", props.date],
  ["model", props.model],
  ["harness", props.harness],
]);
</script>

<template>
  <p class="fa-receipt">
    <span v-if="mark" class="fa-receipt-mark">{{ mark }}</span>
    <template v-for="[label, value] in parts" :key="label">
      <span><b>{{ label }}</b> {{ value ?? "TBC" }}</span>
    </template>
    <span v-if="source"><b>source</b> {{ source }}</span>
    <span v-if="$slots.default" class="fa-receipt-note"><slot /></span>
  </p>
</template>

<style scoped>
.fa-receipt {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 18px;
  margin: 0;
  padding-top: 10px;
  border-top: 2px solid var(--line);
  font: 500 13px/1.4 var(--font-read);
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}
.fa-receipt b {
  font-weight: 700;
  letter-spacing: var(--track-label);
  text-transform: uppercase;
  font-size: 12px;
  margin-right: 4px;
}
.fa-receipt-mark {
  font-weight: 800;
  color: var(--amber-ink);
}
.fa-receipt-note { flex-basis: 100%; }
</style>
