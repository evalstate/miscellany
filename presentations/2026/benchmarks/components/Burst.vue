<script setup lang="ts">
// The one "salesman" per slide: a burst with Ultra shout lettering.
// shape: capsule (primary burst) | star (stickers) | score (headline numbers)
// tone: orange | amber | petrol | teal. Rests at the brand tilt; pops in once.
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    shape?: "capsule" | "star" | "score";
    tone?: "orange" | "amber" | "petrol" | "teal";
    size?: number;
    tilt?: number;
    shadow?: boolean;
    pop?: boolean;
  }>(),
  { shape: "capsule", tone: "orange", size: 132, tilt: -8, shadow: true, pop: true },
);

const masks = { capsule: "/fa/mono/burst.svg", star: "/fa/mono/star.svg", score: "/fa/mono/score.svg" };
const fills = { orange: "var(--orange)", amber: "var(--amber)", petrol: "var(--petrol)", teal: "var(--teal)" };
const inks = { orange: "var(--ivory)", amber: "var(--petrol)", petrol: "var(--ivory)", teal: "var(--ivory)" };
const shadows = { orange: "var(--amber)", amber: "var(--petrol)", petrol: "var(--amber)", teal: "var(--amber)" };

const style = computed(() => ({
  "--burst-size": `${props.size}px`,
  "--tilt": `${props.tilt}deg`,
  "--burst-mask": `url("${masks[props.shape]}")`,
  "--burst-fill": fills[props.tone],
  "--burst-ink": inks[props.tone],
  "--burst-shadow": shadows[props.tone],
}));
</script>

<template>
  <div class="fa-burst" :class="{ 'has-shadow': shadow, 'is-pop': pop }" :style="style">
    <span class="fa-burst-text"><slot /></span>
  </div>
</template>

<style scoped>
.fa-burst {
  position: relative;
  isolation: isolate;
  display: inline-grid;
  place-items: center;
  width: var(--burst-size);
  height: var(--burst-size);
  color: var(--burst-ink);
  transform: rotate(var(--tilt));
}
.fa-burst.is-pop {
  animation: fa-pop var(--dur-pop) var(--ease-pop) 150ms backwards;
}
.fa-burst::before,
.fa-burst::after {
  content: "";
  position: absolute;
  inset: 0;
  mask: var(--burst-mask) center / contain no-repeat;
  z-index: -1;
}
.fa-burst::after {
  background: var(--burst-fill);
}
.fa-burst.has-shadow::before {
  background: var(--burst-shadow);
  transform: translate(4px, 5px);
}
.fa-burst-text {
  display: block;
  max-width: 70%;
  text-align: center;
  font-family: var(--font-shout);
  font-size: calc(var(--burst-size) * 0.18);
  line-height: 1.02;
}
.fa-burst-text :deep(small) {
  display: block;
  font: 700 calc(var(--burst-size) * 0.085) / 1.2 var(--font-read);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.fa-burst-text :deep(p) {
  margin: 0;
}
</style>
