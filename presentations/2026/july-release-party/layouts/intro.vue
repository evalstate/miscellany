<template>
  <section class="slidev-layout intro-layout">
    <div class="intro-mark">JULY</div>
    <main>
      <slot />
      <div class="intro-meta">
        <slot name="meta" />
      </div>
    </main>
  </section>
</template>

<style scoped>
.intro-layout {
  display: grid;
  place-items: center;
  overflow: hidden;
  text-align: center;
}

.intro-layout main {
  width: min(980px, 88%);
}

.intro-layout :deep(h1) {
  margin: 0;
  font-size: 5.2rem;
  letter-spacing: -0.065em;
}

.intro-layout :deep(h2) {
  max-width: 780px;
  margin: 1.15rem auto 0;
  color: var(--deck-muted);
  font-size: 1.55rem;
}

.intro-meta {
  margin-top: 2.4rem;
  color: var(--deck-dim);
  font-family: var(--deck-font-mono);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.intro-meta :deep(p) {
  color: inherit;
  font-size: inherit;
  letter-spacing: inherit;
}

.intro-mark {
  position: absolute;
  right: -0.08em;
  bottom: -0.24em;
  color: rgba(255, 157, 0, 0.075);
  font-size: 17rem;
  font-weight: 800;
  letter-spacing: -0.12em;
  line-height: 1;
  pointer-events: none;
}
</style>
