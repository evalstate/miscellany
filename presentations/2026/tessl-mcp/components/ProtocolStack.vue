<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref } from "vue";

const props = withDefaults(
  defineProps<{
    showDescriptions?: boolean;
  }>(),
  {
    showDescriptions: false,
  },
);

const capabilities = [
  {
    id: "tools",
    title: "Tools",
    icon: "wrench",
    description: "Invoke actions in the outside world",
    zone: "server",
  },
  {
    id: "resources",
    title: "Resources",
    icon: "file",
    description: "Expose context the model can read",
    zone: "server",
  },
  {
    id: "prompts",
    title: "Prompts",
    icon: "message",
    description: "Package reusable instructions",
    zone: "server",
  },
  {
    id: "roots",
    title: "Roots",
    icon: "roots",
    description: "Scope filesystem/project boundaries",
    zone: "client",
  },
  {
    id: "sampling",
    title: "Sampling",
    icon: "sparkles",
    description: "Let servers request model turns",
    zone: "client",
  },
  {
    id: "elicitation",
    title: "Elicitation",
    icon: "question",
    description: "Ask users for missing input",
    zone: "client",
  },
] as const;

type Capability = (typeof capabilities)[number];
type CapabilityId = Capability["id"];

const serverCapabilities = capabilities.filter(
  (capability) => capability.zone === "server",
);
const clientCapabilities = capabilities.filter(
  (capability) => capability.zone === "client",
);

const paths = {
  requestToServer: "M 500 260 C 470 226, 470 184, 500 148",
  responseToClient: "M 552 148 C 582 184, 582 226, 552 260",
  serverToClient: "M 624 148 C 672 190, 650 230, 552 260",
} as const;

const steps = [
  { label: "tools/call", path: paths.requestToServer, flash: undefined },
  { label: "Tools", path: paths.requestToServer, flash: "tools" },
  { label: "result", path: paths.responseToClient, flash: undefined },
  { label: "prompts/get", path: paths.requestToServer, flash: undefined },
  { label: "Prompts", path: paths.requestToServer, flash: "prompts" },
  { label: "prompt", path: paths.responseToClient, flash: undefined },
  {
    label: "sampling/createMessage",
    path: paths.serverToClient,
    flash: undefined,
  },
  { label: "Sampling", path: paths.serverToClient, flash: "sampling" },
] as const;

const activeStep = ref(-1);
const animationKey = ref(0);
const isAnimating = ref(false);
const timers: number[] = [];

const active = computed(() => steps[activeStep.value]);
const activeLabel = computed(() => active.value?.label ?? "click to send");
const activePath = computed(() => active.value?.path ?? paths.requestToServer);
const activeFlash = computed<CapabilityId | undefined>(
  () => active.value?.flash,
);

function clearTimers() {
  while (timers.length) window.clearTimeout(timers.pop());
}

async function pulseStep(index: number) {
  activeStep.value = index;
  isAnimating.value = false;
  animationKey.value += 1;
  await nextTick();
  isAnimating.value = activeFlash.value === undefined;
}

function play() {
  clearTimers();
  activeStep.value = -1;
  isAnimating.value = false;

  steps.forEach((_, index) => {
    timers.push(window.setTimeout(() => void pulseStep(index), index * 760));
  });

  timers.push(
    window.setTimeout(
      () => {
        activeStep.value = -1;
        isAnimating.value = false;
      },
      steps.length * 760 + 260,
    ),
  );
}

onBeforeUnmount(clearTimers);
</script>

<template>
  <button
    class="protocol-stack"
    type="button"
    :class="{ 'protocol-stack--running': isAnimating }"
    @click="play"
  >
    <svg
      class="protocol-flow"
      viewBox="0 0 1000 360"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="protocol-link-glow" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="rgba(106, 163, 247, 0.74)" />
          <stop offset="50%" stop-color="rgba(255, 198, 73, 0.92)" />
          <stop offset="100%" stop-color="rgba(106, 163, 247, 0.74)" />
        </linearGradient>
        <filter
          id="protocol-packet-glow"
          x="-120%"
          y="-120%"
          width="340%"
          height="340%"
        >
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <path
        class="protocol-flow__link protocol-flow__link--base"
        :d="paths.requestToServer"
      />
      <path
        :key="`travel-${animationKey}`"
        class="protocol-flow__link protocol-flow__link--travel"
        :d="activePath"
      />
      <circle
        :key="`packet-${animationKey}`"
        class="protocol-flow__packet"
        r="8"
        filter="url(#protocol-packet-glow)"
      >
        <animateMotion
          dur="0.64s"
          begin="0s"
          fill="freeze"
          :path="activePath"
        />
      </circle>
    </svg>

    <div class="protocol-grid protocol-grid--server">
      <ProtocolCapabilityCard
        v-for="item in serverCapabilities"
        :key="item.id"
        class="protocol-card-shell"
        :class="{ 'is-flashing': activeFlash === item.id }"
        :title="item.title"
        :icon="item.icon"
        :description="item.description"
        :show-description="props.showDescriptions"
      />
    </div>

    <div class="protocol-label protocol-label--server">
      <span>MCP Server</span>
    </div>

    <div class="protocol-exchange">
      <div class="protocol-exchange__line" />
      <div class="protocol-exchange__label">{{ activeLabel }}</div>
    </div>

    <div class="protocol-label protocol-label--client">
      <span>MCP Client</span>
    </div>

    <div class="protocol-grid protocol-grid--client">
      <ProtocolCapabilityCard
        v-for="item in clientCapabilities"
        :key="item.id"
        class="protocol-card-shell"
        :class="{ 'is-flashing': activeFlash === item.id }"
        :title="item.title"
        :icon="item.icon"
        :description="item.description"
        :show-description="props.showDescriptions"
      />
    </div>
  </button>
</template>

<style scoped>
.protocol-stack {
  container-type: size;
  --stack-gap: clamp(0.5rem, 2.1cqh, 0.85rem);
  --protocol-card-pad: clamp(0.58rem, 2.2cqh, 0.88rem)
    clamp(0.72rem, 2.6cqw, 1.1rem);
  --protocol-card-content-gap: clamp(0.58rem, 2.4cqw, 0.98rem);
  --protocol-card-height: clamp(58px, 22cqh, 86px);
  --protocol-icon-size: clamp(2rem, 11cqh, 3rem);
  --protocol-title-size: clamp(1.08rem, 4.4cqh, 1.62rem);
  --protocol-description-size: clamp(0.48rem, 1.8cqh, 0.62rem);
  --protocol-label-height: clamp(44px, 14cqh, 64px);
  --protocol-zone-pad: 0;
  --protocol-exchange-height: clamp(42px, 15cqh, 66px);
  width: 100%;
  height: 100%;
  min-height: 0;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-rows:
    minmax(0, 1fr)
    var(--protocol-label-height)
    var(--protocol-exchange-height)
    var(--protocol-label-height)
    minmax(0, 1fr);
  gap: var(--stack-gap);
  position: relative;
  color: inherit;
  font: inherit;
  text-align: left;
  background: transparent;
  border: 0;
  cursor: pointer;
}

.protocol-flow {
  position: absolute;
  inset: 0;
  z-index: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.protocol-flow__link {
  fill: none;
  stroke-linecap: round;
  stroke-width: 3.5;
}

.protocol-flow__link--base {
  stroke: rgba(185, 179, 165, 0.17);
  stroke-dasharray: 6 12;
}

.protocol-flow__link--travel {
  opacity: 0;
  stroke: url(#protocol-link-glow);
  stroke-width: 6;
  stroke-dasharray: 0 900;
  filter: drop-shadow(0 0 10px rgba(255, 198, 73, 0.62));
}

.protocol-stack--running .protocol-flow__link--travel {
  animation: protocol-link-travel 0.64s ease-out both;
}

.protocol-flow__packet {
  opacity: 0;
  fill: var(--deck-accent-hi);
}

.protocol-stack--running .protocol-flow__packet {
  animation: protocol-packet-visible 0.64s ease-out both;
}

.protocol-label,
.protocol-grid,
.protocol-exchange {
  position: relative;
}

.protocol-label,
.protocol-grid {
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 5px);
}

.protocol-label {
  z-index: 3;
}

.protocol-grid {
  z-index: 2;
}

.protocol-exchange {
  z-index: 1;
}

.protocol-label {
  min-width: 0;
  padding: 0 1.1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background:
    linear-gradient(90deg, rgba(245, 164, 0, 0.08), transparent 46%),
    rgba(20, 22, 27, 0.46);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.protocol-label--client {
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.065), transparent 46%),
    rgba(20, 22, 27, 0.42);
}

.protocol-label span {
  color: var(--deck-text);
  font-size: clamp(1rem, 4.2cqh, 1.42rem);
  font-weight: 850;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.protocol-grid {
  min-height: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--stack-gap);
  align-items: center;
  box-sizing: border-box;
  border-color: transparent;
}

.protocol-grid--server {
  margin-bottom: calc(var(--stack-gap) * -1.2);
  padding-bottom: calc(var(--stack-gap) * 0.8);
}

.protocol-grid--client {
  margin-top: calc(var(--stack-gap) * -1.2);
  padding-top: calc(var(--stack-gap) * 0.8);
}

.protocol-card-shell {
  transition:
    filter 180ms ease,
    transform 180ms ease;
}

.protocol-card-shell.is-flashing {
  filter: drop-shadow(0 0 18px rgba(255, 198, 73, 0.46));
  transform: translateY(-1px) scale(1.018);
}

.protocol-card-shell.is-flashing :deep(.protocol-card) {
  border-color: rgba(255, 198, 73, 0.78);
  background:
    radial-gradient(
      circle at 50% 45%,
      rgba(255, 198, 73, 0.2),
      transparent 58%
    ),
    rgba(20, 22, 27, 0.94);
}

.protocol-card-shell.is-flashing :deep(.protocol-card__icon) {
  color: var(--deck-accent-hi);
}

.protocol-exchange {
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 0.8rem;
}

.protocol-exchange__line {
  grid-column: 1 / -1;
  grid-row: 1;
  height: 2px;
  background: linear-gradient(
    90deg,
    transparent,
    var(--deck-border-2) 22%,
    var(--deck-accent-line) 50%,
    var(--deck-border-2) 78%,
    transparent
  );
}

.protocol-exchange__label {
  grid-column: 2;
  grid-row: 1;
  min-width: 9.5rem;
  padding: clamp(0.24rem, 1cqh, 0.38rem) clamp(0.62rem, 2cqw, 0.9rem);
  text-align: center;
  border: 1px solid var(--deck-border-2);
  border-radius: 999px;
  background: color-mix(in srgb, var(--deck-bg) 86%, transparent);
  color: var(--deck-muted);
  font-size: clamp(0.52rem, 2cqh, 0.68rem);
  font-weight: 850;
  letter-spacing: 0.08em;
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.24);
}

@keyframes protocol-link-travel {
  0% {
    opacity: 0;
    stroke-dasharray: 0 900;
  }
  12% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    stroke-dasharray: 420 900;
  }
}

@keyframes protocol-packet-visible {
  0%,
  100% {
    opacity: 0;
  }
  14%,
  82% {
    opacity: 1;
  }
}
</style>
