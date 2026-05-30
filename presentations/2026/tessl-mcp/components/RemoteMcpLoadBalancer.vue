<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'

const activeServer = ref(1)
const animationKey = ref(0)
const isAnimating = ref(false)

const serverTargets = [
  { id: 1, name: 'MCP Server 01', y: 105 },
  { id: 2, name: 'MCP Server 02', y: 215 },
  { id: 3, name: 'MCP Server 03', y: 325 },
]

const selectedServer = computed(() => serverTargets[activeServer.value - 1])
const selectedPath = computed(() => `M 535 215 C 620 215, 660 ${selectedServer.value.y}, 770 ${selectedServer.value.y}`)

async function fireRequest() {
  activeServer.value = (activeServer.value % 3) + 1
  isAnimating.value = false
  animationKey.value += 1
  await nextTick()
  isAnimating.value = true

  window.setTimeout(() => {
    isAnimating.value = false
  }, 3400)
}
</script>

<template>
  <section
    class="remote-mcp"
    :class="{ 'remote-mcp--running': isAnimating }"
    aria-label="Remote MCP request routed through a load balancer"
  >
    <svg class="remote-mcp__links" viewBox="0 0 1000 430" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="mcp-link-glow" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="rgba(245, 164, 0, 0)" />
          <stop offset="48%" stop-color="rgba(255, 198, 73, 0.92)" />
          <stop offset="100%" stop-color="rgba(106, 163, 247, 0.72)" />
        </linearGradient>
        <filter id="mcp-packet-glow" x="-120%" y="-120%" width="340%" height="340%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <path class="remote-mcp__link remote-mcp__link--base" d="M 185 215 C 285 215, 350 215, 430 215" />
      <path class="remote-mcp__link remote-mcp__link--base" d="M 535 215 C 620 215, 660 105, 770 105" />
      <path class="remote-mcp__link remote-mcp__link--base" d="M 535 215 C 620 215, 660 215, 770 215" />
      <path class="remote-mcp__link remote-mcp__link--base" d="M 535 215 C 620 215, 660 325, 770 325" />

      <path
        :key="`client-link-${animationKey}`"
        class="remote-mcp__link remote-mcp__link--travel remote-mcp__link--to-lb"
        d="M 185 215 C 285 215, 350 215, 430 215"
      />
      <path
        :key="`server-link-${animationKey}`"
        class="remote-mcp__link remote-mcp__link--travel remote-mcp__link--to-server"
        :d="selectedPath"
      />

      <circle :key="`packet-one-${animationKey}`" class="remote-mcp__packet remote-mcp__packet--one" r="8" filter="url(#mcp-packet-glow)">
        <animateMotion dur="0.82s" begin="0s" fill="freeze" path="M 185 215 C 285 215, 350 215, 430 215" />
      </circle>
      <circle :key="`packet-two-${animationKey}`" class="remote-mcp__packet remote-mcp__packet--two" r="8" filter="url(#mcp-packet-glow)">
        <animateMotion dur="0.9s" begin="1.42s" fill="freeze" :path="selectedPath" />
      </circle>
    </svg>

    <button class="remote-mcp__node remote-mcp__node--client" type="button" @click="fireRequest">
      <span class="remote-mcp__node-glow" />
      <span class="remote-mcp__eyebrow">click to send</span>
      <strong>MCP Client</strong>
      <small>JSON-RPC request</small>
    </button>

    <div class="remote-mcp__node remote-mcp__node--lb">
      <span class="remote-mcp__node-glow" />
      <span class="remote-mcp__eyebrow">remote edge</span>
      <strong>Load Balancer</strong>
      <small>route + health check</small>
    </div>

    <div
      v-for="server in serverTargets"
      :key="server.id"
      class="remote-mcp__node remote-mcp__node--server"
      :class="{
        'remote-mcp__node--active-server': activeServer === server.id,
        [`remote-mcp__node--server-${server.id}`]: true,
      }"
    >
      <span class="remote-mcp__node-glow" />
      <span class="remote-mcp__eyebrow">worker {{ server.id }}</span>
      <strong>{{ server.name }}</strong>
      <small>tools · resources · prompts</small>
    </div>

    <div class="remote-mcp__legend">
      <span class="remote-mcp__legend-dot" />
      <span>request pulse follows the selected route</span>
    </div>
  </section>
</template>

<style scoped>
.remote-mcp {
  container-type: size;
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 8px);
  background:
    radial-gradient(circle at 15% 52%, rgba(245, 164, 0, 0.12), transparent 22%),
    radial-gradient(circle at 82% 50%, rgba(106, 163, 247, 0.10), transparent 28%),
    linear-gradient(135deg, rgba(245, 164, 0, 0.055), transparent 38%),
    rgba(20, 22, 27, 0.82);
  box-shadow: var(--deck-shadow);
}

.remote-mcp::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(245, 164, 0, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(245, 164, 0, 0.03) 1px, transparent 1px);
  background-size: 38px 38px;
  mask-image: radial-gradient(circle at 50% 50%, black, transparent 78%);
}

.remote-mcp__links {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.remote-mcp__link {
  fill: none;
  stroke-linecap: round;
  stroke-width: 3.5;
}

.remote-mcp__link--base {
  stroke: rgba(185, 179, 165, 0.22);
  stroke-dasharray: 6 12;
}

.remote-mcp__link--travel {
  opacity: 0;
  stroke: url(#mcp-link-glow);
  stroke-width: 6;
  stroke-dasharray: 0 900;
  filter: drop-shadow(0 0 10px rgba(255, 198, 73, 0.62));
}

.remote-mcp--running .remote-mcp__link--to-lb {
  animation: link-travel 0.82s ease-out both;
}

.remote-mcp--running .remote-mcp__link--to-server {
  animation: link-travel 0.9s ease-out 1.42s both;
}

.remote-mcp__packet {
  opacity: 0;
  fill: var(--deck-accent-hi);
}

.remote-mcp--running .remote-mcp__packet--one {
  animation: packet-visible 0.82s ease-out both;
}

.remote-mcp--running .remote-mcp__packet--two {
  animation: packet-visible 0.9s ease-out 1.42s both;
  fill: var(--deck-info);
}

.remote-mcp__node {
  position: absolute;
  z-index: 2;
  width: clamp(142px, 17cqw, 190px);
  min-height: clamp(84px, 22cqh, 112px);
  display: grid;
  align-content: center;
  gap: 0.24rem;
  padding: clamp(0.72rem, 2.1cqh, 0.98rem);
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 6px);
  background:
    linear-gradient(145deg, rgba(245, 164, 0, 0.105), rgba(20, 22, 27, 0.82) 48%),
    rgba(20, 22, 27, 0.92);
  color: var(--deck-text);
  font-family: var(--deck-font-mono);
  text-align: left;
  box-shadow: 0 18px 38px rgba(0, 0, 0, 0.28);
}

button.remote-mcp__node {
  cursor: pointer;
}

button.remote-mcp__node:hover {
  border-color: rgba(255, 198, 73, 0.58);
  transform: translateY(-1px);
}

.remote-mcp__node--client {
  left: 4.5%;
  top: 50%;
  transform: translateY(-50%);
}

button.remote-mcp__node--client:hover {
  transform: translateY(calc(-50% - 1px));
}

.remote-mcp__node--lb {
  left: 43.5%;
  top: 50%;
  transform: translate(-50%, -50%);
  background:
    linear-gradient(145deg, rgba(106, 163, 247, 0.09), rgba(20, 22, 27, 0.86) 50%),
    rgba(20, 22, 27, 0.94);
}

.remote-mcp__node--server {
  right: 4.5%;
}

.remote-mcp__node--server-1 {
  top: 8%;
}

.remote-mcp__node--server-2 {
  top: 50%;
  transform: translateY(-50%);
}

.remote-mcp__node--server-3 {
  bottom: 8%;
}

.remote-mcp__node--active-server {
  border-color: rgba(106, 163, 247, 0.58);
}

.remote-mcp__eyebrow {
  color: var(--deck-dim);
  font-size: clamp(0.48rem, 1.8cqh, 0.62rem);
  font-weight: 850;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.remote-mcp__node strong {
  color: var(--deck-text);
  font-size: clamp(0.94rem, 4.2cqh, 1.28rem);
  line-height: 1.05;
  letter-spacing: -0.04em;
}

.remote-mcp__node small {
  color: var(--deck-muted);
  font-size: clamp(0.54rem, 2cqh, 0.68rem);
  line-height: 1.2;
}

.remote-mcp__node-glow {
  position: absolute;
  inset: -1px;
  border-radius: inherit;
  opacity: 0;
  pointer-events: none;
  box-shadow:
    0 0 0 1px rgba(255, 198, 73, 0.52),
    0 0 28px rgba(245, 164, 0, 0.55),
    inset 0 0 24px rgba(255, 198, 73, 0.14);
}

.remote-mcp--running .remote-mcp__node--client .remote-mcp__node-glow {
  animation: node-pulse 0.72s ease-out both;
}

.remote-mcp--running .remote-mcp__node--lb .remote-mcp__node-glow {
  animation: node-pulse 0.72s ease-out 0.82s both;
}

.remote-mcp--running .remote-mcp__node--active-server .remote-mcp__node-glow {
  animation: node-pulse-blue 0.86s ease-out 2.25s both;
}

.remote-mcp__legend {
  position: absolute;
  left: 50%;
  bottom: clamp(0.8rem, 3cqh, 1.2rem);
  z-index: 3;
  display: inline-flex;
  align-items: center;
  gap: 0.46rem;
  transform: translateX(-50%);
  padding: 0.38rem 0.68rem;
  border: 1px solid var(--deck-border);
  border-radius: 999px;
  background: color-mix(in srgb, var(--deck-bg) 82%, transparent);
  color: var(--deck-muted);
  font-size: clamp(0.52rem, 2cqh, 0.68rem);
  font-weight: 750;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  white-space: nowrap;
}

.remote-mcp__legend-dot {
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 999px;
  background: var(--deck-accent-hi);
  box-shadow: 0 0 14px rgba(255, 198, 73, 0.8);
}

@keyframes node-pulse {
  0% {
    opacity: 0;
    transform: scale(0.96);
  }
  32% {
    opacity: 1;
    transform: scale(1.045);
  }
  100% {
    opacity: 0;
    transform: scale(1.16);
  }
}

@keyframes node-pulse-blue {
  0% {
    opacity: 0;
    transform: scale(0.96);
    box-shadow:
      0 0 0 1px rgba(106, 163, 247, 0.48),
      0 0 28px rgba(106, 163, 247, 0.5),
      inset 0 0 24px rgba(106, 163, 247, 0.14);
  }
  32% {
    opacity: 1;
    transform: scale(1.045);
    box-shadow:
      0 0 0 1px rgba(106, 163, 247, 0.62),
      0 0 34px rgba(106, 163, 247, 0.64),
      inset 0 0 26px rgba(106, 163, 247, 0.18);
  }
  100% {
    opacity: 0;
    transform: scale(1.16);
    box-shadow:
      0 0 0 1px rgba(106, 163, 247, 0.36),
      0 0 40px rgba(106, 163, 247, 0.32),
      inset 0 0 26px rgba(106, 163, 247, 0.1);
  }
}

@keyframes link-travel {
  0% {
    opacity: 0;
    stroke-dasharray: 0 900;
  }
  12% {
    opacity: 1;
  }
  72% {
    opacity: 1;
    stroke-dasharray: 320 900;
  }
  100% {
    opacity: 0;
    stroke-dasharray: 420 900;
  }
}

@keyframes packet-visible {
  0%,
  100% {
    opacity: 0;
  }
  12%,
  78% {
    opacity: 1;
  }
}
</style>
