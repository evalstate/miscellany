<template>
  <section class="slidev-layout intro-layout">
    <div class="intro-mark">MCP</div>
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
  text-align: center;
  overflow: hidden;
}

.intro-layout main {
  width: min(980px, 88%);
}

.intro-layout :deep(h1) {
  font-size: 5.4rem;
  margin: 0;
  letter-spacing: -0.07em;
}

.intro-layout :deep(h2) {
  margin: 1.15rem auto 0;
  max-width: 780px;
  font-size: 1.55rem;
  color: var(--deck-muted);
}

.intro-meta {
  margin-top: 2.4rem;
  color: var(--deck-dim);
  font-size: 0.85rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.intro-meta :deep(p) {
  color: inherit;
  font-size: inherit;
  letter-spacing: inherit;
  text-transform: inherit;
}

.intro-mark {
  position: absolute;
  right: -0.2em;
  bottom: -0.28em;
  color: rgba(245, 164, 0, 0.055);
  font-size: 23rem;
  font-weight: 800;
  letter-spacing: -0.12em;
  line-height: 1;
  pointer-events: none;
}
</style>
