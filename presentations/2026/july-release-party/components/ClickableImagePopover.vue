<script setup lang="ts">
import { onBeforeUnmount, watch } from "vue";
import { ref } from "vue";

const props = withDefaults(
  defineProps<{
    src: string;
    alt: string;
    prompt?: string;
  }>(),
  {
    prompt: "click to enlarge",
  },
);

const open = ref(false);

function close() {
  open.value = false;
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") close();
}

watch(open, (isOpen) => {
  if (isOpen) window.addEventListener("keydown", onKeydown);
  else window.removeEventListener("keydown", onKeydown);
});

onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <figure class="clickable-image-popover">
    <button class="clickable-image-popover__trigger" type="button" @click="open = true">
      <img :src="props.src" :alt="props.alt" />
      <span>{{ props.prompt }}</span>
    </button>

    <Teleport to="body">
      <div
        v-if="open"
        class="clickable-image-popover__overlay"
        role="dialog"
        aria-modal="true"
        :aria-label="props.alt"
        @click="close"
      >
        <button class="clickable-image-popover__close" type="button" aria-label="Close enlarged image" @click="close">
          ×
        </button>
        <div class="clickable-image-popover__frame" @click.stop>
          <img :src="props.src" :alt="props.alt" />
        </div>
      </div>
    </Teleport>
  </figure>
</template>

<style scoped>
.clickable-image-popover {
  position: relative;
}

.clickable-image-popover__trigger {
  position: relative;
  display: block;
  width: 100%;
  height: 100%;
  padding: 0;
  color: inherit;
  border: 0;
  background: transparent;
  cursor: zoom-in;
}

.clickable-image-popover__trigger img {
  display: block;
  width: 100%;
  height: 100%;
}

.clickable-image-popover__trigger span {
  position: absolute;
  right: 0.55rem;
  bottom: 0.5rem;
  padding: 0.28rem 0.48rem;
  color: var(--deck-accent-hi);
  border: 1px solid rgba(255, 198, 73, 0.36);
  border-radius: 999px;
  background: rgba(243, 244, 246, 0.78);
  font: 850 0.52rem / 1 var(--deck-font-mono);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  opacity: 0.82;
  pointer-events: none;
}

.clickable-image-popover__trigger:hover span,
.clickable-image-popover__trigger:focus-visible span {
  opacity: 1;
}

.clickable-image-popover__trigger:focus-visible {
  outline: 2px solid var(--deck-accent-hi);
  outline-offset: 3px;
}

.clickable-image-popover__overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: grid;
  place-items: center;
  padding: 4.5vh 4vw;
  background:
    radial-gradient(circle at 50% 42%, rgba(255, 198, 73, 0.08), transparent 38%),
    rgba(3, 4, 6, 0.86);
  cursor: zoom-out;
}

.clickable-image-popover__frame {
  max-width: min(94vw, 1540px);
  max-height: 86vh;
  padding: clamp(0.5rem, 1.4vw, 0.95rem);
  border: 1px solid rgba(255, 198, 73, 0.46);
  border-radius: calc(var(--deck-radius) + 12px);
  background: rgba(243, 244, 246, 0.92);
  box-shadow:
    0 26px 70px rgba(0, 0, 0, 0.54),
    0 0 30px rgba(255, 198, 73, 0.08);
  cursor: default;
}

.clickable-image-popover__frame img {
  display: block;
  max-width: 100%;
  max-height: calc(86vh - 2rem);
  object-fit: contain;
  border-radius: var(--deck-radius);
}

.clickable-image-popover__close {
  position: fixed;
  top: 2.2vh;
  right: 2.6vw;
  z-index: 1;
  width: 2.4rem;
  height: 2.4rem;
  display: grid;
  place-items: center;
  color: var(--deck-text);
  border: 1px solid rgba(255, 198, 73, 0.38);
  border-radius: 999px;
  background: rgba(243, 244, 246, 0.84);
  font: 700 1.6rem / 1 var(--deck-font-mono);
  cursor: pointer;
}

.clickable-image-popover__close:hover,
.clickable-image-popover__close:focus-visible {
  color: var(--deck-accent-hi);
  border-color: rgba(255, 198, 73, 0.72);
}
</style>
