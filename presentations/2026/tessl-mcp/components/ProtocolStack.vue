<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";

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

type Direction = "client" | "server";
type Tone = "request" | "response";
type Point = { x: number; y: number };
type PathSpec = { a: Point; c1: Point; c2: Point; b: Point };
type Step = {
  label: string;
  path: keyof typeof paths;
  flash?: CapabilityId;
  tone: Tone;
};

const paths = {
  clientToServer: {
    a: { x: 500, y: 252 },
    c1: { x: 452, y: 220 },
    c2: { x: 452, y: 168 },
    b: { x: 500, y: 126 },
  },
  serverToClient: {
    a: { x: 552, y: 126 },
    c1: { x: 606, y: 168 },
    c2: { x: 606, y: 220 },
    b: { x: 552, y: 252 },
  },
} as const satisfies Record<string, PathSpec>;

const clientSteps: Step[] = [
  {
    label: "tools/call",
    path: "clientToServer",
    tone: "request",
  },
  {
    label: "Tools",
    path: "clientToServer",
    flash: "tools",
    tone: "request",
  },
  {
    label: "result",
    path: "serverToClient",
    tone: "response",
  },
];

const serverSteps: Step[] = [
  {
    label: "sampling/createMessage",
    path: "serverToClient",
    tone: "request",
  },
  {
    label: "Sampling",
    path: "serverToClient",
    flash: "sampling",
    tone: "request",
  },
  {
    label: "result",
    path: "clientToServer",
    tone: "response",
  },
];

const activeStep = ref(-1);
const animationKey = ref(0);
const packetProgress = ref(0);
const activeDirection = ref<Direction | undefined>();
const timers: number[] = [];
let animationFrame = 0;

const activeSteps = computed(() =>
  activeDirection.value === "server" ? serverSteps : clientSteps,
);
const active = computed(() => activeSteps.value[activeStep.value]);
const activeFlash = computed<CapabilityId | undefined>(
  () => active.value?.flash,
);
const activePath = computed(() =>
  active.value ? paths[active.value.path] : paths.clientToServer,
);
const activePathData = computed(() => pathData(activePath.value));
const packetPoint = computed(() =>
  pointOnPath(activePath.value, packetProgress.value),
);

function clearTimers() {
  while (timers.length) window.clearTimeout(timers.pop());
  window.cancelAnimationFrame(animationFrame);
  animationFrame = 0;
  packetProgress.value = 0;
}

function animatePacket(start = performance.now()) {
  const elapsed = performance.now() - start;
  packetProgress.value = Math.min(elapsed / 660, 1);

  if (packetProgress.value < 1) {
    animationFrame = window.requestAnimationFrame(() => animatePacket(start));
  }
}

function pulseStep(index: number) {
  activeStep.value = index;
  packetProgress.value = 0;
  animationKey.value += 1;
  window.cancelAnimationFrame(animationFrame);
  animationFrame = window.requestAnimationFrame(() => animatePacket());
}

function play(direction: Direction) {
  clearTimers();
  activeDirection.value = direction;
  activeStep.value = -1;

  activeSteps.value.forEach((_, index) => {
    timers.push(window.setTimeout(() => pulseStep(index), index * 780));
  });

  timers.push(
    window.setTimeout(
      () => {
        activeStep.value = -1;
        activeDirection.value = undefined;
        packetProgress.value = 0;
      },
      activeSteps.value.length * 780 + 260,
    ),
  );
}

function pathData(path: PathSpec) {
  return `M ${path.a.x} ${path.a.y} C ${path.c1.x} ${path.c1.y}, ${path.c2.x} ${path.c2.y}, ${path.b.x} ${path.b.y}`;
}

function pointOnPath(path: PathSpec, t: number) {
  const u = 1 - t;

  return {
    x:
      u ** 3 * path.a.x +
      3 * u ** 2 * t * path.c1.x +
      3 * u * t ** 2 * path.c2.x +
      t ** 3 * path.b.x,
    y:
      u ** 3 * path.a.y +
      3 * u ** 2 * t * path.c1.y +
      3 * u * t ** 2 * path.c2.y +
      t ** 3 * path.b.y,
  };
}

onBeforeUnmount(clearTimers);
</script>

<template>
  <section
    class="protocol-stack"
    :class="{
      'protocol-stack--running': active,
      'protocol-stack--client-active': activeDirection === 'client',
      'protocol-stack--server-active': activeDirection === 'server',
    }"
    aria-label="MCP protocol bidirectional message flow"
  >
    <svg
      class="protocol-flow"
      viewBox="0 0 1000 360"
      preserveAspectRatio="none"
      role="img"
      aria-label="Messages can flow from MCP client to server and from server to client"
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
        :d="pathData(paths.clientToServer)"
      />
      <path
        class="protocol-flow__link protocol-flow__link--base"
        :d="pathData(paths.serverToClient)"
      />
      <path
        v-if="active"
        :key="`travel-${animationKey}`"
        class="protocol-flow__link protocol-flow__link--travel"
        :class="`protocol-flow__link--${active.tone}`"
        pathLength="1"
        :style="{ '--protocol-pulse-offset': 1 - packetProgress }"
        :d="activePathData"
      />
      <circle
        v-if="active"
        :key="`packet-${animationKey}`"
        class="protocol-flow__packet"
        :class="`protocol-flow__packet--${active.tone}`"
        r="8"
        :cx="packetPoint.x"
        :cy="packetPoint.y"
        filter="url(#protocol-packet-glow)"
      />
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

    <button class="protocol-label protocol-label--server" type="button" @click="play('server')">
      <span>MCP Server</span>
      <small>click to call client</small>
    </button>

    <button class="protocol-label protocol-label--client" type="button" @click="play('client')">
      <span>MCP Client</span>
      <small>click to call server</small>
    </button>

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
    <div
      v-if="active"
      :key="`message-${animationKey}`"
      class="protocol-message"
      :class="`protocol-message--${active.tone}`"
      aria-live="polite"
    >
      <span>in flight</span>
      <strong>{{ active.label }}</strong>
    </div>
  </section>
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
  width: 100%;
  height: 100%;
  min-height: 0;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-rows:
    minmax(0, 1fr)
    var(--protocol-label-height)
    var(--protocol-label-height)
    minmax(0, 1fr);
  gap: var(--stack-gap);
  position: relative;
  color: inherit;
  font: inherit;
  text-align: left;
  background: transparent;
  border: 0;
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
  opacity: 1;
  stroke: url(#protocol-link-glow);
  stroke-width: 6;
  stroke-dasharray: 0.24 1;
  stroke-dashoffset: var(--protocol-pulse-offset);
  filter: drop-shadow(0 0 10px rgba(255, 198, 73, 0.62));
}

.protocol-flow__link--response {
  stroke: var(--deck-info);
  filter: drop-shadow(0 0 10px rgba(106, 163, 247, 0.62));
}

.protocol-flow__packet {
  opacity: 1;
  fill: var(--deck-accent-hi);
}

.protocol-flow__packet--response {
  fill: var(--deck-info);
}

.protocol-label,
.protocol-grid {
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

.protocol-label {
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 5px);
  min-width: 0;
  padding: 0 1.1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: inherit;
  font: inherit;
  background:
    linear-gradient(90deg, rgba(245, 164, 0, 0.08), transparent 46%),
    rgba(20, 22, 27, 0.46);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
  cursor: pointer;
}

.protocol-label--client {
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.065), transparent 46%),
    rgba(20, 22, 27, 0.42);
}

.protocol-label:hover,
.protocol-stack--server-active .protocol-label--server,
.protocol-stack--client-active .protocol-label--client {
  border-color: rgba(255, 198, 73, 0.62);
  background:
    radial-gradient(circle at 10% 50%, rgba(255, 198, 73, 0.12), transparent 34%),
    rgba(20, 22, 27, 0.62);
}

.protocol-label--client:hover,
.protocol-stack--client-active .protocol-label--client {
  border-color: rgba(106, 163, 247, 0.7);
  background:
    radial-gradient(circle at 10% 50%, rgba(106, 163, 247, 0.14), transparent 34%),
    rgba(20, 22, 27, 0.62);
}

.protocol-label span {
  color: var(--deck-text);
  font-size: clamp(1rem, 4.2cqh, 1.42rem);
  font-weight: 850;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.protocol-label small {
  color: var(--deck-dim);
  font-size: clamp(0.48rem, 1.7cqh, 0.62rem);
  font-weight: 850;
  letter-spacing: 0.14em;
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

.protocol-message {
  position: absolute;
  left: 50%;
  top: 50%;
  z-index: 5;
  min-width: 16rem;
  padding: 0.52rem 0.72rem;
  display: grid;
  gap: 0.18rem;
  transform: translate(-50%, -50%);
  border: 1px solid rgba(255, 198, 73, 0.46);
  border-radius: calc(var(--deck-radius) + 4px);
  background: rgba(11, 12, 15, 0.86);
  box-shadow: 0 16px 34px rgba(0, 0, 0, 0.32);
  animation: protocol-message 760ms ease both;
}

.protocol-message--response {
  border-color: rgba(106, 163, 247, 0.54);
}

.protocol-message span {
  color: var(--deck-dim);
  font-size: clamp(0.46rem, 1.6cqh, 0.58rem);
  font-weight: 850;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.protocol-message strong {
  color: var(--deck-text);
  font-size: clamp(0.76rem, 2.6cqh, 0.96rem);
  font-weight: 850;
  letter-spacing: -0.03em;
}

@keyframes protocol-message {
  0%,
  100% {
    opacity: 0;
    transform: translate(-50%, calc(-50% + 6px)) scale(0.98);
  }
  16%,
  82% {
    opacity: 1;
    transform: translate(-50%, -50%) scale(1);
  }
}
</style>
