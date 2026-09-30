<script setup lang="ts">
// Editorial card: flat fill, 2px petrol keyline, radius 14. No shadows, no
// left-border accents. Amber tone is reserved for fast-agent itself.
withDefaults(
  defineProps<{ tone?: "paper" | "ivory" | "deep" | "amber" | "orange" | "petrol" }>(),
  { tone: "paper" },
);
</script>

<template>
  <section class="fa-card" :class="`fa-card--${tone}`">
    <slot />
  </section>
</template>

<style scoped>
.fa-card {
  display: flex;
  flex-direction: column;
  padding: 24px 26px;
  border: var(--border);
  border-radius: var(--radius-lg);
  min-width: 0;
}
.fa-card :deep(> :last-child) { margin-bottom: 0; }
.fa-card :deep(h2), .fa-card :deep(h3) { margin-bottom: 12px; }
.fa-card--paper { background: var(--paper); }
.fa-card--ivory { background: var(--ivory); }
.fa-card--deep { background: var(--ivory-deep); }
.fa-card--amber { background: var(--amber); }
.fa-card--orange { background: var(--orange); color: var(--ivory); }
.fa-card--petrol { background: var(--petrol); color: var(--ivory); }
.fa-card--amber :deep(li::before), .fa-card--orange :deep(li::before) { color: inherit; }
.fa-card--petrol :deep(li::before) { color: var(--amber); }
</style>
