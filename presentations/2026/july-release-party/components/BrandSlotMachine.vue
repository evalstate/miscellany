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
    const next = [...settledReels.value];
    next[2] = false;
    settledReels.value = next;
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
    settleReel(2);
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
  --reel-height: 520px;
  --cell-height: 170px;
  --start-y: calc((var(--reel-height) - var(--cell-height)) / 2);
  --rest-y: calc(var(--start-y) - var(--cell-height) * 3);
  display: grid;
  width: min(1080px, 96%);
  height: var(--reel-height);
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.3rem;
  align-items: center;
  margin-inline: auto;
  cursor: pointer;
}

.icon-reel {
  position: relative;
  height: var(--reel-height);
  overflow: hidden;
  clip-path: none;
  mask-image: linear-gradient(
    transparent 0%,
    rgba(0, 0, 0, 0.74) 10%,
    #000 26%,
    #000 74%,
    rgba(0, 0, 0, 0.74) 90%,
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
  transform: translateY(var(--start-y));
  animation: none;
  transition: none;
}

.track-mcp.mcp-phase-1 {
  transform: translateY(calc(var(--rest-y) - 22px));
  transition: transform 1.4s cubic-bezier(0.18, 0.8, 0.2, 1);
}

.track-mcp.mcp-phase-2 {
  transform: translateY(calc(var(--rest-y) + 12px));
  transition: transform 180ms ease-out;
}

.track-mcp.mcp-phase-3 {
  transform: translateY(calc(var(--rest-y) - 6px));
  transition: transform 160ms ease-in-out;
}

.track-mcp.mcp-phase-4 {
  transform: translateY(var(--rest-y));
  transition: transform 140ms ease-out;
}

.icon-cell {
  display: grid;
  place-items: center;
  padding: 10px;
  transition:
    opacity 300ms ease-out,
    scale 420ms cubic-bezier(0.2, 0.86, 0.25, 1.15);
}

.icon-cell img {
  display: block;
  width: 132px;
  height: 132px;
  object-fit: contain;
}

.reel-huggy img {
  width: 144px;
  height: 144px;
}

.aaif-symbol,
.mcp-symbol {
  width: 126px !important;
  height: 126px !important;
}

.heart-symbol {
  width: 136px !important;
  height: 126px !important;
}

.icon-reel.is-settled .initial-track .icon-cell:not(:last-child),
.track-mcp.mcp-settled .icon-cell:not(:last-child) {
  opacity: 0;
}

.icon-reel:not(.is-settled) .icon-cell img,
.track-mcp:not(.mcp-settled) .icon-cell img {
  animation: symbol-rotate 620ms linear infinite;
}

@keyframes reel-in {
  0% {
    transform: translateY(var(--start-y));
  }
  58% {
    transform: translateY(
      calc(var(--start-y) - var(--cell-height) * 2.15)
    );
  }
  76% {
    transform: translateY(
      calc(var(--start-y) - var(--cell-height) * 2.72)
    );
  }
  86% {
    transform: translateY(calc(var(--rest-y) - 22px));
  }
  91% {
    transform: translateY(calc(var(--rest-y) + 12px));
  }
  96% {
    transform: translateY(calc(var(--rest-y) - 6px));
  }
  100% {
    transform: translateY(var(--rest-y));
  }
}

.icon-reel.is-settled .initial-track .icon-cell:last-child img,
.track-mcp.mcp-settled .icon-cell:last-child img {
  scale: 1.72;
  animation: winner-bounce 420ms cubic-bezier(0.2, 0.86, 0.25, 1.15) both;
}

@keyframes symbol-rotate {
  to {
    transform: rotate(1turn);
  }
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
