<script setup lang="ts">
// Section divider on the petrol ground. `number` sets a score burst with
// Ultra lettering (this is the slide's single burst). Five atomic sparkles sit
// round the edges as decoration; each turns and dims very slowly, out of step.
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
.fa-section-spark {
  position: absolute;
  width: var(--spark, 34px);
  aspect-ratio: 1;
  background: var(--amber);
  mask: url("/fa/sparkle-atomic.svg") center / contain no-repeat;
  opacity: 0.7;
  /* one turn takes minutes; the dim is a few points of opacity */
  animation:
    fa-spark-turn var(--turn, 180s) linear infinite var(--dir, normal),
    fa-spark-dim var(--dim, 9s) ease-in-out var(--wait, 0s) infinite alternate;
}
.fa-section-spark.is-1 { right: 72px; top: 64px; }
.fa-section-spark.is-2 { right: 176px; top: 136px; --spark: 28px; --turn: 240s; --dir: reverse; --dim: 11s; --wait: -4s; }
.fa-section-spark.is-3 { right: 104px; bottom: 76px; --spark: 38px; --turn: 210s; --dim: 13s; --wait: -7s; }
.fa-section-spark.is-4 { left: 64px; top: 56px; --spark: 30px; --turn: 200s; --dir: reverse; --dim: 10s; --wait: -2s; }
.fa-section-spark.is-5 { left: 336px; bottom: 60px; --spark: 32px; --turn: 260s; --dim: 12s; --wait: -9s; }
@keyframes fa-spark-turn {
  to { transform: rotate(360deg); }
}
@keyframes fa-spark-dim {
  from { opacity: 0.74; }
  to { opacity: 0.46; }
}
</style>
