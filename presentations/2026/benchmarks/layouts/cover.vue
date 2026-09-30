<script setup lang="ts">
// Title slide: headline + meta on the left, the homepage splash on the right.
// Named slot `::meta::` for speaker/event details.
withDefaults(defineProps<{ tagline?: string }>(), {
  tagline: "The harness your model deserves.",
});
</script>

<template>
  <section class="slidev-layout fa-cover">
    <div class="fa-cover-copy">
      <slot />
      <div v-if="$slots.meta" class="fa-cover-meta"><slot name="meta" /></div>
    </div>
    <div class="fa-cover-splash">
      <Wordmark :tagline="tagline" :size="300" />
    </div>
  </section>
</template>

<style scoped>
.fa-cover {
  display: grid;
  grid-template-columns: 1.15fr 1fr;
  gap: 32px;
  align-items: center;
}
.fa-cover-copy :deep(h1) {
  font-size: 64px;
  line-height: 0.98;
  margin-bottom: 20px;
}
.fa-cover-copy :deep(h1 + p) {
  font-size: 24px;
  margin-top: 0;
}
.fa-cover-meta {
  margin-top: 36px;
  padding-top: 14px;
  border-top: var(--border);
  font-size: 17px;
}
.fa-cover-meta :deep(p) { margin: 0 0 4px; }
.fa-cover-splash {
  display: grid;
  place-items: center;
}
</style>
