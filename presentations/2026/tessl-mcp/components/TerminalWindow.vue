<script setup lang="ts">
defineProps<{
  title?: string
}>()
</script>

<template>
  <section class="terminal deck-panel">
    <header>
      <span class="dot dot-red" />
      <span class="dot dot-amber" />
      <span class="dot dot-green" />
      <span class="terminal-title">{{ title || 'terminal' }}</span>
    </header>
    <div class="terminal-body">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.terminal {
  overflow: hidden;
  margin-top: 1rem;
}

.terminal header {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.58rem 0.8rem;
  border-bottom: 1px solid var(--deck-border);
  background: rgba(0, 0, 0, 0.18);
}

.dot {
  width: 0.64rem;
  height: 0.64rem;
  border-radius: 999px;
  opacity: 0.85;
}

.dot-red { background: var(--deck-no); }
.dot-amber { background: var(--deck-accent); }
.dot-green { background: var(--deck-ok); }

.terminal-title {
  margin-left: 0.5rem;
  color: var(--deck-dim);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.terminal-body {
  padding: 1rem;
}

.terminal-body :deep(pre) {
  margin: 0;
  border: 0;
  box-shadow: none;
  background: transparent !important;
}
</style>
