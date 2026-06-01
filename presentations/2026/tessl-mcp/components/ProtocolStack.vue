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

type Channel = "up" | "down";
type Step = {
  label: string;
  channel: Channel;
  flash?: CapabilityId;
};

const scriptSteps: Step[] = [
  {
    label: "prompts/list",
    channel: "up",
    flash: "prompts",
  },
  {
    label: "prompts/GetListResult",
    channel: "down",
  },
  {
    label: "tools/call",
    channel: "up",
    flash: "tools",
  },
  {
    label: "progress/notification",
    channel: "down",
  },
  {
    label: "progress/notification",
    channel: "down",
  },
  {
    label: "progress/notification",
    channel: "down",
  },
  {
    label: "tools/CallToolResult",
    channel: "down",
  },
  {
    label: "sampling/createMessage",
    channel: "down",
    flash: "sampling",
  },
  {
    label: "result",
    channel: "up",
  },
];

const activeStep = ref(-1);
const animationKey = ref(0);
const isRunning = ref(false);
const timers: number[] = [];

const active = computed(() => scriptSteps[activeStep.value]);
const activeFlash = computed<CapabilityId | undefined>(
  () => active.value?.flash,
);
const operationLabel = computed(() => active.value?.label ?? "tools/call");
const activeChannel = computed<Channel | undefined>(() => active.value?.channel);

function clearTimers() {
  while (timers.length) window.clearTimeout(timers.pop());
}

function pulseStep(index: number) {
  activeStep.value = index;
  animationKey.value += 1;
}

function play() {
  clearTimers();
  isRunning.value = true;
  activeStep.value = -1;

  scriptSteps.forEach((_, index) => {
    timers.push(window.setTimeout(() => pulseStep(index), index * 1120));
  });

  timers.push(
    window.setTimeout(
      () => {
        activeStep.value = -1;
        isRunning.value = false;
      },
      scriptSteps.length * 1120 + 160,
    ),
  );
}

onBeforeUnmount(clearTimers);
</script>

<template>
  <section
    class="protocol-stack"
    :class="{
      'protocol-stack--running': isRunning,
      'protocol-stack--channel-up': activeChannel === 'up',
      'protocol-stack--channel-down': activeChannel === 'down',
    }"
    aria-label="MCP protocol bidirectional message flow"
    @click="play"
  >
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

    <button class="protocol-label protocol-label--server" type="button" @click.stop="play">
      <span>MCP Server</span>
    </button>

    <div class="protocol-process-gap">
      <div class="protocol-arrow-pair" aria-hidden="true">
        <div
          :key="`up-${animationKey}`"
          class="protocol-block-arrow protocol-block-arrow--up protocol-block-arrow--connected"
        >
          <span class="protocol-message-dot" />
        </div>
        <div
          :key="`down-${animationKey}`"
          class="protocol-block-arrow protocol-block-arrow--down protocol-block-arrow--connected"
        >
          <span class="protocol-message-dot" />
        </div>
      </div>
      <div class="protocol-operation" aria-live="polite">
        <span>operation</span>
        <strong>{{ operationLabel }}</strong>
      </div>
    </div>

    <button class="protocol-label protocol-label--client" type="button" @click.stop="play">
      <span>MCP Client</span>
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
  </section>
</template>

<style scoped>
.protocol-stack {
  container-type: size;
  --stack-gap: clamp(0.5rem, 2.1cqh, 0.85rem);
  --protocol-card-pad: clamp(0.62rem, 2.2cqh, 0.9rem)
    clamp(0.76rem, 2.65cqw, 1.12rem);
  --protocol-card-content-gap: clamp(0.62rem, 2.45cqw, 1rem);
  --protocol-card-height: clamp(60px, 22cqh, 84px);
  --protocol-icon-size: clamp(2.08rem, 10.8cqh, 3rem);
  --protocol-title-size: clamp(1.1rem, 4.3cqh, 1.56rem);
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
    minmax(var(--protocol-card-height), 1fr)
    var(--protocol-label-height)
    clamp(62px, 20cqh, 88px)
    var(--protocol-label-height)
    minmax(var(--protocol-card-height), 1fr);
  gap: var(--stack-gap);
  position: relative;
  color: inherit;
  font: inherit;
  text-align: left;
  background: transparent;
  border: 0;
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

.protocol-process-gap {
  position: relative;
  z-index: 4;
  min-height: 0;
  overflow: visible;
  border-inline: 1px solid rgba(185, 179, 165, 0.08);
  background:
    linear-gradient(90deg, transparent, rgba(185, 179, 165, 0.1), transparent)
      50% 50% / 46% 1px no-repeat;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 0.46fr);
  align-items: center;
  gap: clamp(1rem, 3.2cqw, 2rem);
}

.protocol-arrow-pair {
  position: absolute;
  top: calc(var(--stack-gap) * -1);
  left: 47%;
  z-index: 5;
  display: flex;
  align-items: stretch;
  justify-content: center;
  gap: clamp(0.9rem, 2.2cqw, 1.28rem);
  height: calc(100% + var(--stack-gap) * 2);
  transform: translateX(-50%);
  pointer-events: none;
}

.protocol-block-arrow {
  --arrow-color: rgba(106, 163, 247, 0.82);
  --arrow-glow: rgba(106, 163, 247, 0.34);
  --arrow-outline: rgba(225, 234, 255, 0.76);
  --arrow-head-extend: clamp(12px, 2.4cqh, 20px);
  position: relative;
  width: clamp(74px, 8.8cqw, 102px);
  height: 100%;
  background: var(--arrow-color);
  clip-path: polygon(
    33% 0,
    67% 0,
    67% 80%,
    96% 80%,
    50% 100%,
    4% 80%,
    33% 80%
  );
  filter:
    drop-shadow(2px 0 0 var(--arrow-outline))
    drop-shadow(-2px 0 0 var(--arrow-outline))
    drop-shadow(0 2px 0 var(--arrow-outline))
    drop-shadow(0 -2px 0 var(--arrow-outline))
    drop-shadow(0 0 16px var(--arrow-glow));
  opacity: 0.68;
  overflow: hidden;
  transition:
    filter 160ms ease,
    opacity 160ms ease;
}

.protocol-block-arrow--connected {
  --arrow-outline: rgba(235, 241, 255, 0.9);
  opacity: 0.72;
}

.protocol-block-arrow--disconnected {
  --arrow-outline: rgba(185, 179, 165, 0.32);
  opacity: 0.34;
  filter:
    drop-shadow(2px 0 0 var(--arrow-outline))
    drop-shadow(-2px 0 0 var(--arrow-outline))
    drop-shadow(0 2px 0 var(--arrow-outline))
    drop-shadow(0 -2px 0 var(--arrow-outline));
}

.protocol-message-dot {
  position: absolute;
  left: 50%;
  top: 8%;
  width: clamp(12px, 2.1cqw, 18px);
  aspect-ratio: 1;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow:
    0 0 0 5px color-mix(in srgb, var(--arrow-color) 35%, transparent),
    0 0 24px color-mix(in srgb, var(--arrow-color) 74%, white 12%);
  opacity: 0;
  transform: translate(-50%, 0);
}

.protocol-block-arrow--up {
  --arrow-color: rgba(255, 198, 73, 0.82);
  --arrow-glow: rgba(255, 198, 73, 0.34);
  --arrow-outline: rgba(255, 232, 166, 0.92);
  height: calc(100% + var(--arrow-head-extend));
  margin-top: calc(var(--arrow-head-extend) * -1);
  transform: rotate(180deg);
}

.protocol-block-arrow--down {
  height: calc(100% + var(--arrow-head-extend));
}

.protocol-stack--channel-down .protocol-block-arrow--down {
  animation: protocol-arrow-down 760ms ease both;
}

.protocol-stack--channel-down .protocol-block-arrow--down .protocol-message-dot {
  animation: protocol-dot-down 720ms cubic-bezier(0.2, 0.72, 0.2, 1) both;
}

.protocol-stack--channel-up .protocol-block-arrow--up {
  animation: protocol-arrow-up 760ms ease both;
}

.protocol-stack--channel-up .protocol-block-arrow--up .protocol-message-dot {
  animation: protocol-dot-down 720ms cubic-bezier(0.2, 0.72, 0.2, 1) both;
}

.protocol-operation {
  grid-column: 2;
  justify-self: end;
  z-index: 3;
  width: min(100%, 310px);
  padding: clamp(0.72rem, 2.4cqh, 0.98rem) clamp(0.9rem, 2.1cqw, 1.2rem);
  display: grid;
  gap: 0.28rem;
  border: 1px solid rgba(255, 198, 73, 0.4);
  border-radius: calc(var(--deck-radius) + 5px);
  background: rgba(11, 12, 15, 0.84);
  box-shadow: 0 18px 38px rgba(0, 0, 0, 0.28);
}

.protocol-operation span {
  color: var(--deck-dim);
  font-size: clamp(0.5rem, 1.8cqh, 0.66rem);
  font-weight: 850;
  letter-spacing: 0.15em;
  line-height: 1;
  text-transform: uppercase;
}

.protocol-operation strong {
  color: var(--deck-text);
  font-size: clamp(1.02rem, 4.1cqh, 1.42rem);
  font-weight: 950;
  letter-spacing: -0.045em;
  line-height: 1.05;
  white-space: nowrap;
}

.protocol-stack--running .protocol-operation {
  border-color: rgba(255, 198, 73, 0.58);
  box-shadow:
    0 18px 38px rgba(0, 0, 0, 0.3),
    0 0 24px rgba(255, 198, 73, 0.12);
}

.protocol-stack--response .protocol-operation {
  border-color: rgba(106, 163, 247, 0.58);
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
  animation: protocol-card-pulse 520ms ease 620ms both;
}

.protocol-card-shell.is-flashing :deep(.protocol-card) {
  animation: protocol-card-surface-pulse 520ms ease 620ms both;
}

.protocol-card-shell.is-flashing :deep(.protocol-card__icon) {
  animation: protocol-card-icon-pulse 520ms ease 620ms both;
}

@keyframes protocol-arrow-down {
  0%,
  100% {
    opacity: 0.72;
    filter:
      drop-shadow(2px 0 0 rgba(235, 241, 255, 0.9))
      drop-shadow(-2px 0 0 rgba(235, 241, 255, 0.9))
      drop-shadow(0 2px 0 rgba(235, 241, 255, 0.9))
      drop-shadow(0 -2px 0 rgba(235, 241, 255, 0.9))
      drop-shadow(0 0 16px rgba(106, 163, 247, 0.3));
    transform: translateY(-1%) scaleY(0.98);
  }
  18%,
  78% {
    opacity: 1;
    filter:
      drop-shadow(2px 0 0 rgba(245, 248, 255, 0.96))
      drop-shadow(-2px 0 0 rgba(245, 248, 255, 0.96))
      drop-shadow(0 2px 0 rgba(245, 248, 255, 0.96))
      drop-shadow(0 -2px 0 rgba(245, 248, 255, 0.96))
      drop-shadow(0 0 24px rgba(106, 163, 247, 0.68));
    transform: translateY(0) scaleY(1);
  }
}

@keyframes protocol-arrow-up {
  0%,
  100% {
    opacity: 0.72;
    filter:
      drop-shadow(2px 0 0 rgba(255, 232, 166, 0.92))
      drop-shadow(-2px 0 0 rgba(255, 232, 166, 0.92))
      drop-shadow(0 2px 0 rgba(255, 232, 166, 0.92))
      drop-shadow(0 -2px 0 rgba(255, 232, 166, 0.92))
      drop-shadow(0 0 16px rgba(255, 198, 73, 0.3));
    transform: rotate(180deg) translateY(-1%) scaleY(0.98);
  }
  18%,
  78% {
    opacity: 1;
    filter:
      drop-shadow(2px 0 0 rgba(255, 239, 191, 0.98))
      drop-shadow(-2px 0 0 rgba(255, 239, 191, 0.98))
      drop-shadow(0 2px 0 rgba(255, 239, 191, 0.98))
      drop-shadow(0 -2px 0 rgba(255, 239, 191, 0.98))
      drop-shadow(0 0 24px rgba(255, 198, 73, 0.68));
    transform: rotate(180deg) translateY(0) scaleY(1);
  }
}

@keyframes protocol-dot-down {
  0% {
    opacity: 0;
    transform: translate(-50%, -30%) scale(0.72);
  }
  18%,
  78% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: translate(-50%, 620%) scale(1.08);
  }
}

@keyframes protocol-card-pulse {
  0%,
  100% {
    filter: none;
    transform: none;
  }
  45% {
    filter: drop-shadow(0 0 20px rgba(255, 198, 73, 0.52));
    transform: translateY(-1px) scale(1.02);
  }
}

@keyframes protocol-card-surface-pulse {
  0%,
  100% {
    border-color: var(--deck-border);
  }
  45% {
    border-color: rgba(255, 198, 73, 0.82);
    background:
      radial-gradient(
        circle at 50% 45%,
        rgba(255, 198, 73, 0.22),
        transparent 58%
      ),
      rgba(20, 22, 27, 0.94);
  }
}

@keyframes protocol-card-icon-pulse {
  0%,
  100% {
    color: var(--deck-muted);
    transform: none;
  }
  45% {
    color: var(--deck-accent-hi);
    transform: scale(1.06);
  }
}
</style>
