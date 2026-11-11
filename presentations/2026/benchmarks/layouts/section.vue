<script setup lang="ts">
// Section divider on the petrol ground. `number` sets a score burst with
// Ultra lettering (this is the slide's single burst). Five atomic sparkles sit
// round the edges as decoration; each drifts and dims very slowly, out of step.
defineProps<{ number?: string | number }>();
</script>

<template>
  <section class="slidev-layout fa-section">
    <Burst v-if="number !== undefined" shape="score" tone="amber" :size="210" :shadow="false" class="fa-section-burst">
      {{ number }}
    </Burst>
    <div class="fa-section-copy">
      <slot />
    </div>
    <span v-for="n in 5" :key="n" class="fa-section-spark" :class="`is-${n}`" aria-hidden="true"></span>
  </section>
</template>

<style scoped>
.fa-section {
  position: relative;
  display: flex;
  align-items: center;
  gap: 56px;
  background: var(--petrol);
  color: var(--ivory);
  padding: 56px 80px;
}
.fa-section-burst { flex: none; }
.fa-section-burst :deep(.fa-burst-text) { font-size: 80px; max-width: none; }
.fa-section-copy :deep(h1) { font-size: 64px; line-height: 1; margin: 0 0 16px; }
.fa-section-copy :deep(p) { font-size: 22px; color: rgba(255, 247, 232, 0.8); margin: 0; }
</style>
