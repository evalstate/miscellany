<script setup lang="ts">
defineProps<{
  kicker?: string
}>()
</script>

<template>
  <section class="slidev-layout two-column-layout">
    <div v-if="kicker" class="kicker">{{ kicker }}</div>
    <div class="title-region">
      <slot />
    </div>
    <div class="columns-region">
      <div class="column left-column">
        <slot name="left" />
      </div>
      <div class="column right-column">
        <slot name="right" />
      </div>
    </div>
  </section>
</template>

<style scoped>
.title-region :deep(h1) {
  max-width: 980px;
  margin-bottom: 1rem;
}

.columns-region {
  display: grid;
  grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr);
  gap: 1.8rem;
  align-items: start;
}

.column {
  min-width: 0;
}

.left-column :deep(ul) {
  margin-top: 0;
}
</style>
