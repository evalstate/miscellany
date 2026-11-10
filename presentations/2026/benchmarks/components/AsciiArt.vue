<script setup lang="ts">
// A verbatim ASCII render from a trace, shown as a petrol "terminal" tile.
// `art` is a key from data/deepseek-ascii.ts. The heading is the label the
// agent itself printed (e.g. "h1 N"), so slides quote the model, not us.
import { computed } from "vue";
import * as ascii from "../data/deepseek-ascii";

const props = withDefaults(
  defineProps<{ art: keyof typeof ascii; label?: string; size?: number; tone?: "petrol" | "paper"; cross?: boolean }>(),
  { size: 9, tone: "petrol", cross: false },
);
const text = computed(() => ascii[props.art]);
</script>

<template>
  <figure class="fa-ascii" :class="[`fa-ascii--${tone}`, { 'is-crossed': cross }]">
    <figcaption v-if="label" class="fa-ascii-label">{{ label }}</figcaption>
    <pre :style="{ fontSize: `${size}px` }">{{ text }}</pre>
  </figure>
</template>

<style scoped>
.fa-ascii {
  margin: 0;
  display: flex;
  flex-direction: column;
  border: var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  min-width: 0;
}
.fa-ascii--petrol { background: var(--petrol); color: var(--ivory); }
.fa-ascii--paper { background: var(--paper); color: var(--petrol); }
.fa-ascii-label {
  padding: 8px 14px;
  font: 500 14px/1 var(--font-code);
  border-bottom: 2px solid var(--petrol-2);
  color: rgba(255, 247, 232, 0.72);
}
.fa-ascii--paper .fa-ascii-label { border-bottom-color: var(--line); color: var(--text-muted); }
.fa-ascii pre {
  flex: 1;
  display: grid;
  place-items: center;
  margin: 0;
  padding: 14px;
  border: 0 !important;
  border-radius: 0;
  background: transparent !important;
  font-family: var(--font-code);
  line-height: 1.02;
  letter-spacing: 0.04em;
  white-space: pre;
  overflow: hidden;
}
.fa-ascii.is-crossed .fa-ascii-label { text-decoration: line-through; text-decoration-color: var(--orange); text-decoration-thickness: 2px; }
</style>
