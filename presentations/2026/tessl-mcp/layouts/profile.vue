<script setup lang="ts">
defineProps<{
  handle?: string
  role?: string
}>()
</script>

<template>
  <section class="slidev-layout profile-layout">
    <div class="profile-card deck-panel">
      <div class="avatar">SS</div>
      <div class="profile-main">
        <slot />
        <div class="profile-links">
          <slot name="links" />
        </div>
      </div>
      <div class="profile-meta">
        <span>{{ handle }}</span>
        <span>{{ role }}</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.profile-layout {
  display: grid;
  place-items: center;
}

.profile-card {
  width: min(980px, 100%);
  min-height: 520px;
  padding: 2.6rem;
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 2.2rem;
  align-items: start;
}

.avatar {
  width: 150px;
  height: 150px;
  display: grid;
  place-items: center;
  border: 1px solid var(--deck-accent-line);
  background: var(--deck-accent-bg);
  color: var(--deck-accent-hi);
  border-radius: var(--deck-radius);
  font-size: 3rem;
  font-weight: 750;
  letter-spacing: -0.08em;
}

.profile-main :deep(h1) {
  margin-top: 0;
  margin-bottom: 1.3rem;
}

.profile-main :deep(ul) {
  margin: 0;
  padding-left: 1.15rem;
}

.profile-links {
  margin-top: 1.8rem;
  padding: 0.9rem 1rem;
  border: 1px solid var(--deck-border);
  background: rgba(11, 12, 15, 0.38);
  border-radius: var(--deck-radius-sm);
}

.profile-links :deep(ul) {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem 1.1rem;
  list-style: none;
  padding: 0;
}

.profile-links :deep(li) {
  font-size: 0.85rem;
  color: var(--deck-dim);
}

.profile-meta {
  grid-column: 1 / -1;
  align-self: end;
  display: flex;
  justify-content: space-between;
  color: var(--deck-dim);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  border-top: 1px solid var(--deck-border);
  padding-top: 1rem;
}
</style>
