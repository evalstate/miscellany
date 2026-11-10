<script setup lang="ts">
// A speech balloon for a masked quote from a trace. `who` is the model;
// `where` is the short locator (task, step, date). The quote is the slot.
defineProps<{ who: string; where?: string }>();
</script>

<template>
  <figure class="fa-balloon">
    <blockquote><slot /></blockquote>
    <figcaption><b>{{ who }}</b><span v-if="where"> · {{ where }}</span></figcaption>
  </figure>
</template>

<style scoped>
.fa-balloon { margin: 0 0 14px; }
.fa-balloon blockquote {
  position: relative;
  margin: 0;
  padding: 12px 16px;
  border: var(--border);
  border-radius: var(--radius-lg);
  background: var(--paper);
  font: italic 500 18px/1.35 var(--font-read);
  color: var(--petrol);
}
.fa-balloon blockquote::before { content: "“"; }
.fa-balloon blockquote::after { content: "”"; }
.fa-balloon blockquote :deep(p) { display: inline; margin: 0; font: inherit; }
/* the tail: a small square turned 45°, two sides keylined */
.fa-balloon figcaption {
  position: relative;
  margin: 0;
  padding: 10px 0 0 30px;
  font: 500 13px/1.3 var(--font-read);
  color: var(--text-muted);
}
.fa-balloon figcaption::before {
  content: "";
  position: absolute;
  left: 22px;
  top: -8px;
  width: 12px;
  height: 12px;
  background: var(--paper);
  border-right: var(--border);
  border-bottom: var(--border);
  transform: rotate(45deg);
}
.fa-balloon figcaption b { color: var(--petrol); font-weight: 700; }
</style>
