<script setup lang="ts">
// Tilted label with a solid, print-style offset shadow (never a blur).
withDefaults(defineProps<{ tone?: "paper" | "amber" | "petrol"; tilt?: number }>(), {
  tone: "paper",
  tilt: -6,
});
</script>

<template>
  <span class="fa-sticker" :class="`fa-sticker--${tone}`" :style="{ '--tilt': `${tilt}deg` }"><slot /></span>
</template>

<style scoped>
.fa-sticker {
  display: inline-block;
  padding: 8px 14px;
  border: var(--border);
  border-radius: var(--radius-sm);
  font: 18px/1 var(--font-shout);
  white-space: nowrap;
  transform: rotate(var(--tilt));
  animation: fa-pop var(--dur-pop) var(--ease-pop) 150ms backwards;
}
.fa-sticker--paper { background: var(--paper); color: var(--petrol); box-shadow: var(--shadow-offset); }
.fa-sticker--amber { background: var(--amber); color: var(--petrol); box-shadow: var(--shadow-offset-sm); }
.fa-sticker--petrol { background: var(--petrol); color: var(--ivory); box-shadow: var(--shadow-offset); }
</style>
