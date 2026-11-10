<script setup lang="ts">
// Stacked two-line mark on a burst, for tools that aren't fast-agent.
// Amber is reserved for fast-agent, so this uses orange and tilts the
// other way from the fast-agent wordmark.
withDefaults(
  defineProps<{ top: string; bottom: string; label?: string; size?: number; tilt?: number; spin?: number }>(),
  { size: 300, tilt: 6, spin: -14 },
);
</script>

<template>
  <div
    class="fa-toolmark"
    role="img"
    :aria-label="label ?? `${top}${bottom}`"
    :style="{ '--tm-size': `${size}px`, '--tm-tilt': `${tilt}deg`, '--tm-spin': `${spin}deg` }"
  >
    <span aria-hidden="true">{{ top }}<br />{{ bottom }}</span>
  </div>
</template>

<style scoped>
.fa-toolmark {
  position: relative;
  isolation: isolate;
  display: grid;
  place-items: center;
  width: var(--tm-size);
  aspect-ratio: 1;
  font-family: var(--font-voice);
  font-variation-settings: var(--voice-settings);
  font-weight: 900;
  letter-spacing: var(--track-voice);
  font-size: calc(var(--tm-size) * 0.27);
  line-height: 0.86;
  text-align: center;
  color: var(--petrol);
  text-shadow: 3px 4px 0 var(--ivory);
}
.fa-toolmark > span {
  display: block;
  transform: rotate(var(--tm-tilt)) scale(1.4);
}
.fa-toolmark::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  background: var(--orange);
  mask: url("/fa/burst-score.svg") center / contain no-repeat;
  transform: rotate(var(--tm-spin));
  animation: fa-toolmark-swing 1.2s ease backwards;
}
@keyframes fa-toolmark-swing {
  0% { opacity: 0; transform: scale(0.4) rotate(20deg); }
  50% { opacity: 1; transform: scale(1.05) rotate(20deg); }
  85% { transform: rotate(calc(var(--tm-spin) - 6deg)); }
  100% { transform: rotate(var(--tm-spin)); }
}
@media (prefers-reduced-motion: reduce) {
  .fa-toolmark::before { animation: none; }
}
</style>
