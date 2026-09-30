<script setup lang="ts">
// Ivory page with a flat brand footer. No eyebrows (deck rule).
withDefaults(defineProps<{ footer?: boolean }>(), { footer: true });
</script>

<template>
  <section class="slidev-layout fa-default">
    <div class="fa-default-body"><slot /></div>
    <footer v-if="footer" class="fa-footer">
      <span class="fa-footer-brand"><img src="/fa/icon-tile.svg" alt="" />fast-agent</span>
      <span class="fa-footer-page">{{ $slidev.nav.currentPage }}</span>
    </footer>
  </section>
</template>

<style scoped>
.fa-default {
  display: flex;
  flex-direction: column;
  padding-bottom: 20px;
}
.fa-default-body {
  flex: 1;
  min-height: 0;
}
.fa-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 10px;
  border-top: 2px solid var(--line);
  font: 700 13px/1 var(--font-read);
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}
.fa-footer-brand {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-voice);
  font-weight: 900;
  font-size: 15px;
  letter-spacing: var(--track-voice);
  color: var(--petrol);
}
.fa-footer-brand img { width: 20px; height: 20px; }
</style>
