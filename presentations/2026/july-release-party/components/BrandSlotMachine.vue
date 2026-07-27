<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";

const cycle = ref(0);
const initialSettled = ref(false);
const showMcp = ref(false);
const mcpSettled = ref(false);
let shiftTimer: ReturnType<typeof setTimeout> | undefined;
let settleTimer: ReturnType<typeof setTimeout> | undefined;
let initialSettleTimer: ReturnType<typeof setTimeout> | undefined;

function replay() {
  if (shiftTimer) clearTimeout(shiftTimer);
  if (settleTimer) clearTimeout(settleTimer);
  if (initialSettleTimer) clearTimeout(initialSettleTimer);
  initialSettled.value = false;
  showMcp.value = false;
  mcpSettled.value = false;
  cycle.value += 1;
  initialSettleTimer = setTimeout(() => {
    initialSettled.value = true;
  }, 1900);
  shiftTimer = setTimeout(() => {
    showMcp.value = true;
    settleTimer = setTimeout(() => {
      mcpSettled.value = true;
    }, 50);
  }, 3200);
}

onMounted(replay);
onBeforeUnmount(() => {
  if (shiftTimer) clearTimeout(shiftTimer);
  if (settleTimer) clearTimeout(settleTimer);
  if (initialSettleTimer) clearTimeout(initialSettleTimer);
});
</script>

<template>
  <div
    :key="cycle"
    class="icon-machine"
    :class="{ 'is-initial-settled': initialSettled }"
    role="img"
    aria-label="Hugging Face loves AAIF, then MCP"
    title="Click to replay"
    @click.stop="replay"
  >
    <div class="icon-reel reel-huggy">
      <div class="icon-track initial-track">
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

    <div class="icon-reel reel-heart">
      <div class="icon-track initial-track">
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

    <div class="icon-reel reel-foundation">
      <div
        v-if="!showMcp"
        key="aaif"
        class="icon-track initial-track track-aaif"
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
        :class="{ 'mcp-settled': mcpSettled }"
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
  --cell-height: 360px;
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
  overflow: hidden;
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
  animation: reel-in 1.45s cubic-bezier(0.16, 0.76, 0.22, 1) forwards;
  will-change: transform, filter;
}

.reel-heart .icon-track {
  animation-delay: 160ms;
}

.track-aaif {
  animation-delay: 320ms;
}

.track-mcp {
  filter: blur(7px);
  transform: translateY(0);
  animation: none;
  transition:
    transform 1.15s cubic-bezier(0.18, 0.8, 0.2, 1),
    filter 0.85s ease-out;
}

.track-mcp.mcp-settled {
  filter: blur(0);
  transform: translateY(calc(var(--cell-height) * -3));
}

.icon-cell {
  display: grid;
  place-items: center;
  padding: 12px;
  transition: opacity 260ms ease-out;
}

.icon-cell img {
  display: block;
  width: min(318px, 94%);
  height: min(318px, 94%);
  object-fit: contain;
}

.reel-huggy img {
  width: min(338px, 98%);
  height: min(338px, 98%);
}

.aaif-symbol,
.mcp-symbol {
  width: min(308px, 92%) !important;
  height: min(308px, 92%) !important;
}

.heart-symbol {
  width: min(320px, 94%) !important;
  height: min(300px, 90%) !important;
}

.icon-machine.is-initial-settled
  .initial-track
  .icon-cell:not(:last-child),
.track-mcp.mcp-settled .icon-cell:not(:last-child) {
  opacity: 0;
}

@keyframes reel-in {
  0% {
    filter: blur(8px);
    transform: translateY(0);
  }
  70% {
    filter: blur(3px);
  }
  88% {
    transform: translateY(calc(var(--cell-height) * -3 - 18px));
  }
  100% {
    filter: blur(0);
    transform: translateY(calc(var(--cell-height) * -3));
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
