<script setup lang="ts">
// Content + the presenter. She stands on the side her gesture requires:
// open palm (B) presents content to her left, so she sits right; pointing (C)
// and palm-up (F) face right, so she sits left. Override with `side`.
import { computed } from "vue";

const props = withDefaults(
  defineProps<{ pose?: "A" | "B" | "C" | "D" | "E" | "F"; side?: "left" | "right"; height?: number }>(),
  { pose: "B", height: 440 },
);
const resolvedSide = computed(() => props.side ?? (["C", "F"].includes(props.pose) ? "left" : "right"));
</script>

<template>
  <section class="slidev-layout fa-presenter-layout" :class="`is-${resolvedSide}`">
    <div class="fa-presenter-copy">
      <slot />
    </div>
    <figure class="fa-presenter-figure">
      <PresenterPose :pose="pose" :height="height" />
    </figure>
  </section>
</template>

<style scoped>
.fa-presenter-layout {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 24px;
  align-items: center;
  padding-bottom: 0;
}
.fa-presenter-layout.is-left { grid-template-columns: auto 1fr; }
.fa-presenter-layout.is-left .fa-presenter-figure { order: -1; }
.fa-presenter-copy { padding-bottom: 44px; min-width: 0; }
.fa-presenter-figure {
  margin: 0;
  align-self: end;
  display: flex;
  justify-content: center;
}
</style>
