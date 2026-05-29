<script setup lang="ts">
defineProps<{
  label: string
  value: string
  note?: string
}>()
</script>

<template>
  <article class="metric-card deck-panel">
    <div class="metric-label">{{ label }}</div>
    <div class="metric-value">{{ value }}</div>
    <p v-if="note" class="metric-note">{{ note }}</p>
  </article>
</template>

<style scoped>
.metric-card {
  padding: 1.15rem;
  min-height: 150px;
}

.metric-label {
  color: var(--deck-dim);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.metric-value {
  color: var(--deck-accent-hi);
  font-size: 2rem;
  font-weight: 750;
  letter-spacing: -0.06em;
  line-height: 1.1;
  margin-top: 0.8rem;
}

.metric-note {
  color: var(--deck-muted);
  font-size: 0.76rem;
  line-height: 1.35;
  margin-top: 0.7rem;
}
</style>
