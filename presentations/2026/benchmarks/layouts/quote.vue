<script setup lang="ts">
// Full-screen statement on the petrol ground, in the same family as the
// section dividers. Each paragraph of the slide is one line of the statement.
</script>

<template>
  <section class="slidev-layout fa-quote">
    <div class="fa-quote-copy"><slot /></div>
    <span v-for="n in 5" :key="n" class="fa-section-spark" :class="`is-${n}`" aria-hidden="true"></span>
  </section>
</template>

<style scoped>
.fa-quote { position: relative; overflow: hidden; display: flex; align-items: center; background: var(--petrol); color: var(--ivory); padding: 56px 96px; }
.fa-quote-copy :deep(p) { margin: 0 0 34px; font: 900 56px/1.1 var(--font-voice); color: var(--ivory); }
.fa-quote-copy :deep(p:last-child) { margin-bottom: 0; }
.fa-quote-copy :deep(strong) { color: var(--amber); font-weight: inherit; }
</style>
