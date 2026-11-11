<script setup lang="ts">
// Full-screen statement on the petrol ground, in the same family as the
// section dividers. Each paragraph of the slide is one line of the statement.
</script>

<template>
  <section class="slidev-layout fa-quote">
    <div class="fa-quote-copy"><slot /></div>
    <span class="fa-quote-spark" aria-hidden="true"></span>
  </section>
</template>

<style scoped>
.fa-quote { position: relative; display: flex; align-items: center; background: var(--petrol); color: var(--ivory); padding: 56px 96px; }
.fa-quote-copy :deep(p) { margin: 0 0 34px; font: 400 52px/1.12 var(--font-voice); color: var(--ivory); }
.fa-quote-copy :deep(p:last-child) { margin-bottom: 0; }
.fa-quote-copy :deep(strong) { color: var(--amber); font-weight: inherit; }
.fa-quote-spark { position: absolute; right: 72px; top: 64px; width: 34px; height: 34px; background: var(--amber); mask: url("/fa/sparkle-atomic.svg") center / contain no-repeat; opacity: 0.7; }
</style>
