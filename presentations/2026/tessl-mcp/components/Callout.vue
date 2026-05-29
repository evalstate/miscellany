<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    title?: string
    tone?: 'accent' | 'info' | 'ok' | 'warn'
  }>(),
  { tone: 'accent' },
)
</script>

<template>
  <aside class="callout" :class="`callout--${props.tone}`">
    <div v-if="title" class="callout-title">{{ title }}</div>
    <div class="callout-body"><slot /></div>
  </aside>
</template>

<style scoped>
.callout {
  margin: 1rem 0;
  padding: 0.95rem 1rem;
  border-radius: var(--deck-radius-sm);
  border: 1px solid var(--deck-accent-line);
  background: var(--deck-accent-bg);
}

.callout--info {
  border-color: var(--deck-info-line);
  background: var(--deck-info-bg);
}

.callout--ok {
  border-color: rgba(106, 209, 156, 0.34);
  background: rgba(106, 209, 156, 0.08);
}

.callout-title {
  color: var(--deck-text);
  font-weight: 700;
  font-size: 0.82rem;
  margin-bottom: 0.35rem;
}

.callout-body :deep(p) {
  margin: 0;
  color: var(--deck-muted);
  font-size: 0.88rem;
  line-height: 1.45;
}
</style>
