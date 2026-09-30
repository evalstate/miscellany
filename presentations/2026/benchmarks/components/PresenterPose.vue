<script setup lang="ts">
// <PresenterPose pose="B" :height="420" />. The recurring presenter (approved pose sheet, 27 Sep 2026). Poses were cut
// from presenter-approved-poses.png without redrawing:
//  A board (right)  B open palm (to left)  C pointing (to right)
//  D folder         E leaning on a sign    F palm up (to right)
// Keep her outside chart/data areas; one mascot per slide.
import { computed } from "vue";

const props = withDefaults(defineProps<{ pose?: "A" | "B" | "C" | "D" | "E" | "F"; height?: number }>(), {
  pose: "B",
  height: 420,
});

const alts: Record<string, string> = {
  A: "Presenter in a teal suit holding up a blank board",
  B: "Presenter in a teal suit gesturing with an open palm",
  C: "Presenter in a teal suit pointing",
  D: "Presenter in a teal suit holding a folder",
  E: "Presenter in a teal suit leaning over a sign",
  F: "Presenter in a teal suit smiling with her palm up",
};
const src = computed(() => `/fa/illustration/poses/presenter-${props.pose}.png`);
</script>

<template>
  <img class="fa-presenter" :src="src" :alt="alts[pose]" :style="{ height: `${height}px` }" />
</template>

<style scoped>
.fa-presenter {
  display: block;
  width: auto;
  max-width: none;
  object-fit: contain;
}
</style>
