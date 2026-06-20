<script setup lang="ts">
defineProps<{
  kicker?: string
}>()
</script>

<template>
  <section class="slidev-layout default-layout">
    <div v-if="kicker" class="kicker">{{ kicker }}</div>
    <slot />
  </section>
</template>
