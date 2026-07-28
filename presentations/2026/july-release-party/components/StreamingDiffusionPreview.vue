<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { Check, Play } from "@lucide/vue";
import { useTimedStoryboard } from "../composables/useTimedStoryboard";

type Phase = "partial" | "complete";

type Frame = {
  id: string;
  phase: Phase;
  sequence: number;
  elapsed: string;
  image: string;
  label: string;
  duration: number;
};

const frames = computed<readonly Frame[]>(() => [
  {
    id: "partial-1",
    phase: "partial",
    sequence: 1,
    elapsed: "2.6s",
    image: "streaming/flux-step-1.webp",
    label: "Denoising step 1 of 4",
    duration: 760,
  },
  {
    id: "partial-2",
    phase: "partial",
    sequence: 2,
    elapsed: "3.4s",
    image: "streaming/flux-step-2.webp",
    label: "Denoising step 2 of 4",
    duration: 760,
  },
  {
    id: "partial-3",
    phase: "partial",
    sequence: 3,
    elapsed: "4.1s",
    image: "streaming/flux-step-3.webp",
    label: "Denoising step 3 of 4",
    duration: 760,
  },
  {
    id: "partial-4",
    phase: "partial",
    sequence: 4,
    elapsed: "4.7s",
    image: "streaming/flux-step-4.webp",
    label: "Denoising step 4 of 4",
    duration: 820,
  },
  {
    id: "terminal",
    phase: "complete",
    sequence: 4,
    elapsed: "5.3s",
    image: "streaming/flux-step-4.webp",
    label: "Final generated image",
    duration: 1800,
  },
]);

const activated = ref(false);
const { active, animationKey, isRunning, play } = useTimedStoryboard<Frame>(
  frames,
  { defaultDuration: 760, endDelay: 520 },
);

const visible = computed(() => active.value ?? frames.value[0]);

function run() {
  activated.value = true;
  play(true);
}

onMounted(() => {
  frames.value.forEach(({ image }) => {
    const preload = new Image();
    preload.src = image;
  });
});
</script>

<template>
  <section
    class="stream-preview"
    :class="[
      `is-${visible.phase}`,
      {
        'is-activated': activated,
        'is-running': isRunning,
      },
    ]"
    role="button"
    tabindex="0"
    aria-label="Animated sequence of partial image results from a diffusion model"
    title="Click to stream the diffusion result"
    @click.stop="run"
    @keydown.enter.prevent="run"
    @keydown.space.prevent="run"
  >
    <header class="stream-preview__header">
      <div>
        <span>tool result stream</span>
        <strong>
          {{ visible.phase === "complete" ? "Complete" : `Partial result ${visible.sequence}/4` }}
        </strong>
      </div>
      <time>{{ visible.elapsed }}</time>
    </header>

    <div class="stream-preview__image">
      <Transition name="stream-frame">
        <img
          :key="`${animationKey}-${visible.id}`"
          :src="visible.image"
          :alt="visible.label"
        />
      </Transition>

      <div v-if="!activated" class="stream-preview__start">
        <Play :size="24" fill="currentColor" />
        <strong>Stream result</strong>
        <span>click to begin</span>
      </div>
    </div>

    <footer class="stream-preview__footer">
      <div class="stream-preview__steps" aria-hidden="true">
        <i
          v-for="step in 4"
          :key="step"
          :class="{ 'is-reached': visible.sequence >= step }"
        />
      </div>
      <span v-if="visible.phase === 'complete'" class="stream-preview__terminal">
        <Check :size="15" :stroke-width="2.8" />
        terminal result
      </span>
      <span v-else>ordered content chunk</span>
    </footer>
  </section>
</template>

<style scoped>
.stream-preview {
  display: grid;
  height: 100%;
  grid-template-rows: auto minmax(0, 1fr) auto;
  overflow: hidden;
  color: #f9fafb;
  background: #101623;
  border: 1px solid #263044;
  border-radius: 14px;
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.18);
  cursor: pointer;
  outline: none;
}

.stream-preview:focus-visible {
  box-shadow:
    0 0 0 3px rgba(255, 210, 30, 0.65),
    0 18px 44px rgba(15, 23, 42, 0.18);
}

.stream-preview__header,
.stream-preview__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.72rem 0.9rem;
}

.stream-preview__header {
  border-bottom: 1px solid #263044;
}

.stream-preview__header > div {
  display: flex;
  align-items: baseline;
  gap: 0.65rem;
}

.stream-preview__header span,
.stream-preview__header time,
.stream-preview__footer {
  color: #9ca3af;
  font: 600 0.58rem/1 var(--deck-font-mono);
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.stream-preview__header strong {
  color: #fbbf24;
  font-size: 0.82rem;
}

.stream-preview.is-complete .stream-preview__header strong {
  color: #34d399;
}

.stream-preview__image {
  position: relative;
  min-height: 0;
  overflow: hidden;
  background: #05070b;
}

.stream-preview__image img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.stream-preview__start {
  position: absolute;
  inset: 0;
  display: grid;
  place-content: center;
  justify-items: center;
  gap: 0.35rem;
  color: #f9fafb;
  background: rgba(5, 7, 11, 0.42);
  backdrop-filter: blur(1px);
}

.stream-preview__start svg {
  width: 3.35rem;
  height: 3.35rem;
  padding: 0.95rem;
  color: #111827;
  background: var(--deck-yellow);
  border-radius: 999px;
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.26);
}

.stream-preview__start strong {
  color: #f9fafb;
  font-size: 1rem;
}

.stream-preview__start span {
  color: #fbbf24;
  font: 600 0.55rem/1 var(--deck-font-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.stream-preview__footer {
  min-height: 2.35rem;
  border-top: 1px solid #263044;
}

.stream-preview__steps {
  display: flex;
  gap: 0.32rem;
}

.stream-preview__steps i {
  width: 1.65rem;
  height: 0.28rem;
  background: #374151;
  border-radius: 999px;
  transition: background 180ms ease-out;
}

.stream-preview__steps i.is-reached {
  background: #fbbf24;
}

.stream-preview.is-complete .stream-preview__steps i {
  background: #34d399;
}

.stream-preview__terminal {
  display: inline-flex;
  align-items: center;
  gap: 0.28rem;
  color: #34d399;
}

.stream-frame-enter-active {
  transition:
    opacity 260ms ease-out,
    filter 260ms ease-out;
}

.stream-frame-enter-from {
  opacity: 0;
  filter: blur(5px);
}

@media (prefers-reduced-motion: reduce) {
  .stream-frame-enter-active,
  .stream-preview__steps i {
    transition-duration: 1ms;
  }
}
</style>
