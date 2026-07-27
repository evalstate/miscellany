<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";

const cycle = ref(0);
const settledReels = ref([false, false, false]);
const showMcp = ref(false);
const mcpSettled = ref(false);
const mcpPhase = ref(0);
const timers: ReturnType<typeof setTimeout>[] = [];

function schedule(callback: () => void, delay: number) {
  timers.push(setTimeout(callback, delay));
}

function clearTimers() {
  while (timers.length) clearTimeout(timers.pop());
}

function settleReel(index: number) {
  const next = [...settledReels.value];
  next[index] = true;
  settledReels.value = next;
}

function replay() {
  clearTimers();
  settledReels.value = [false, false, false];
  showMcp.value = false;
  mcpSettled.value = false;
  mcpPhase.value = 0;
  cycle.value += 1;

  schedule(() => {
    showMcp.value = true;
    mcpPhase.value = 0;
  }, 6500);
  schedule(() => {
    mcpPhase.value = 1;
  }, 6560);
  schedule(() => {
    mcpPhase.value = 2;
  }, 7960);
  schedule(() => {
    mcpPhase.value = 3;
  }, 8140);
  schedule(() => {
    mcpPhase.value = 4;
    mcpSettled.value = true;
  }, 8300);
}

onMounted(replay);
onBeforeUnmount(clearTimers);
</script>

<template>
  <div
    :key="cycle"
    class="icon-machine"
    role="img"
    aria-label="Hugging Face loves AAIF, then MCP"
    title="Click to replay"
    @click.stop="replay"
  >
    <div
      class="icon-reel reel-huggy"
      :class="{ 'is-settled': settledReels[0] }"
    >
      <div
        class="icon-track initial-track"
        @animationend.self="settleReel(0)"
      >
        <div class="icon-cell">
          <img src="/brand/aaif-symbol-black.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/mcp-symbol-black.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/heart.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/hugging-face.svg" alt="Hugging Face" />
        </div>
      </div>
    </div>

    <div
      class="icon-reel reel-heart"
      :class="{ 'is-settled': settledReels[1] }"
    >
      <div
        class="icon-track initial-track"
        @animationend.self="settleReel(1)"
      >
        <div class="icon-cell">
          <img src="/brand/hugging-face.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/aaif-symbol-black.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/mcp-symbol-black.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img class="heart-symbol" src="/brand/heart.svg" alt="loves" />
        </div>
      </div>
    </div>

    <div
      class="icon-reel reel-foundation"
      :class="{ 'is-settled': settledReels[2] }"
    >
      <div
        v-if="!showMcp"
        key="aaif"
        class="icon-track initial-track track-aaif"
        @animationend.self="settleReel(2)"
      >
        <div class="icon-cell">
          <img src="/brand/heart.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/mcp-symbol-black.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/hugging-face.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img
            class="aaif-symbol"
            src="/brand/aaif-symbol-black.svg"
            alt="Agentic AI Foundation"
          />
        </div>
      </div>

      <div
        v-else
        key="mcp"
        class="icon-track track-mcp"
        :class="[
          `mcp-phase-${mcpPhase}`,
          { 'mcp-settled': mcpSettled },
        ]"
      >
        <div class="icon-cell">
          <img
            class="aaif-symbol"
            src="/brand/aaif-symbol-black.svg"
            alt=""
          />
        </div>
        <div class="icon-cell">
          <img src="/brand/hugging-face.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img src="/brand/heart.svg" alt="" />
        </div>
        <div class="icon-cell">
          <img
            class="mcp-symbol"
            src="/brand/mcp-symbol-black.svg"
            alt="Model Context Protocol"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.icon-machine {
  --cell-height: 520px;
  display: grid;
  width: min(1080px, 96%);
  height: var(--cell-height);
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.3rem;
  align-items: center;
  margin-inline: auto;
  cursor: pointer;
}

.icon-reel {
  position: relative;
  height: var(--cell-height);
  overflow: visible;
  clip-path: inset(0 -92px);
  mask-image: linear-gradient(
    transparent 0%,
    #000 14%,
    #000 86%,
    transparent 100%
  );
}

.icon-track {
  display: grid;
  grid-auto-rows: var(--cell-height);
  animation: reel-in 3s cubic-bezier(0.16, 0.76, 0.22, 1) forwards;
  will-change: transform, filter;
}

.reel-heart .icon-track {
  animation-duration: 3.45s;
}

.track-aaif {
  animation-duration: 3.9s;
}

.track-mcp {
  filter: blur(7px);
  transform: translateY(0);
  animation: none;
  transition: none;
}

.track-mcp.mcp-phase-1 {
  filter: blur(2px);
  transform: translateY(calc(var(--cell-height) * -3.12));
  transition:
    transform 1.4s cubic-bezier(0.18, 0.8, 0.2, 1),
    filter 1.1s ease-out;
}

.track-mcp.mcp-phase-2 {
  filter: blur(0);
  transform: translateY(calc(var(--cell-height) * -2.94));
  transition: transform 180ms ease-out;
}

.track-mcp.mcp-phase-3 {
  transform: translateY(calc(var(--cell-height) * -3.035));
  transition: transform 160ms ease-in-out;
}

.track-mcp.mcp-phase-4 {
  filter: blur(0);
  transform: translateY(calc(var(--cell-height) * -3));
  transition: transform 140ms ease-out;
}

.icon-cell {
  display: grid;
  place-items: center;
  padding: 12px;
  transition: opacity 260ms ease-out;
}

.icon-cell img {
  display: block;
  width: min(420px, 112%);
  height: min(420px, 112%);
  object-fit: contain;
}

.reel-huggy img {
  width: min(448px, 122%);
  height: min(448px, 122%);
  scale: 1.35;
}

.aaif-symbol,
.mcp-symbol {
  width: min(410px, 114%) !important;
  height: min(410px, 114%) !important;
}

.aaif-symbol {
  scale: 1.2;
}

.mcp-symbol {
  scale: 1.27;
}

.heart-symbol {
  width: min(430px, 118%) !important;
  height: min(410px, 112%) !important;
  scale: 1.18;
}

.icon-reel.is-settled .initial-track .icon-cell:not(:last-child),
.track-mcp.mcp-settled .icon-cell:not(:last-child) {
  opacity: 0;
}

@keyframes reel-in {
  0% {
    filter: blur(8px);
    transform: translateY(0);
  }
  58% {
    filter: blur(7px);
    transform: translateY(calc(var(--cell-height) * -2.15));
  }
  76% {
    filter: blur(3px);
    transform: translateY(calc(var(--cell-height) * -2.72));
  }
  86% {
    transform: translateY(calc(var(--cell-height) * -3.12));
  }
  91% {
    filter: blur(0);
    transform: translateY(calc(var(--cell-height) * -2.94));
  }
  96% {
    transform: translateY(calc(var(--cell-height) * -3.035));
  }
  100% {
    filter: blur(0);
    transform: translateY(calc(var(--cell-height) * -3));
  }
}

.icon-reel.is-settled .initial-track .icon-cell:last-child img,
.track-mcp.mcp-settled .icon-cell:last-child img {
  animation: winner-bounce 420ms cubic-bezier(0.2, 0.86, 0.25, 1.15) both;
}

@keyframes winner-bounce {
  0% {
    transform: scale(0.92);
  }
  55% {
    transform: scale(1.055);
  }
  78% {
    transform: scale(0.985);
  }
  100% {
    transform: scale(1);
  }
}

@media (prefers-reduced-motion: reduce) {
  .icon-track,
  .track-mcp {
    animation-duration: 1ms;
    transition-duration: 1ms;
  }
}
</style>
