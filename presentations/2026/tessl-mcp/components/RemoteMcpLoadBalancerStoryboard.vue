<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";

type Anchor = "left" | "right";

type NodeSpec = {
  id: string;
  role: string;
  title: string;
  detail: string;
  x: number;
  y: number;
  w: number;
  h: number;
};

type EdgeSpec = {
  id: string;
  from: string;
  to: string;
  fromAnchor: Anchor;
  toAnchor: Anchor;
};

type RouteLeg = {
  edgeId: string;
  reverse?: boolean;
};

type StoryStep = {
  id: string;
  route: RouteLeg[];
  label: string;
  caption: string;
  tone?: "request" | "result" | "notify";
};

const nodes: NodeSpec[] = [
  {
    id: "client",
    role: "endpoint",
    title: "Client",
    detail: "",
    x: 56,
    y: 180,
    w: 198,
    h: 112,
  },
  {
    id: "lb",
    role: "remote edge",
    title: "Load balancer",
    detail: "",
    x: 386,
    y: 185,
    w: 238,
    h: 102,
  },
  {
    id: "server-a",
    role: "server",
    title: "Server 01",
    detail: "",
    x: 748,
    y: 72,
    w: 206,
    h: 92,
  },
  {
    id: "server-b",
    role: "server",
    title: "Server 02",
    detail: "",
    x: 748,
    y: 190,
    w: 206,
    h: 92,
  },
  {
    id: "server-c",
    role: "server",
    title: "Server 03",
    detail: "",
    x: 748,
    y: 308,
    w: 206,
    h: 92,
  },
];

const edges: EdgeSpec[] = [
  { id: "client-lb", from: "client", to: "lb", fromAnchor: "right", toAnchor: "left" },
  { id: "lb-a", from: "lb", to: "server-a", fromAnchor: "right", toAnchor: "left" },
  { id: "lb-b", from: "lb", to: "server-b", fromAnchor: "right", toAnchor: "left" },
  { id: "lb-c", from: "lb", to: "server-c", fromAnchor: "right", toAnchor: "left" },
];

const initializeSteps: StoryStep[] = [
  {
    id: "initialize-request",
    route: [{ edgeId: "client-lb" }, { edgeId: "lb-b" }],
    label: "InitializeRequest",
    caption: "Client initializes through the load balancer",
    tone: "request",
  },
  {
    id: "initialize-result",
    route: [
      { edgeId: "lb-b", reverse: true },
      { edgeId: "client-lb", reverse: true },
    ],
    label: "InitializeResult + Session ID",
    caption: "Server responds through the load balancer",
    tone: "result",
  },
  {
    id: "initialized-notification",
    route: [{ edgeId: "client-lb" }, { edgeId: "lb-b" }],
    label: "notifications/initialized",
    caption: "Client acknowledges initialization",
    tone: "notify",
  },
];

const requestSteps: StoryStep[] = [
  {
    id: "tools-list-request",
    route: [{ edgeId: "client-lb" }, { edgeId: "lb-b" }],
    label: "tools/list",
    caption: "Client asks for available tools",
    tone: "request",
  },
  {
    id: "tools-list-result",
    route: [
      { edgeId: "lb-b", reverse: true },
      { edgeId: "client-lb", reverse: true },
    ],
    label: "ListToolsResultResponse",
    caption: "Server returns the tool list",
    tone: "result",
  },
  {
    id: "prompts-list-request",
    route: [{ edgeId: "client-lb" }, { edgeId: "lb-b" }],
    label: "prompts/list",
    caption: "Client asks for available prompts",
    tone: "request",
  },
  {
    id: "prompts-list-result",
    route: [
      { edgeId: "lb-b", reverse: true },
      { edgeId: "client-lb", reverse: true },
    ],
    label: "ListPromptsResultResponse",
    caption: "Server returns the prompt list",
    tone: "result",
  },
];

const toolCallSteps: StoryStep[] = [
  {
    id: "tool-call-request",
    route: [{ edgeId: "client-lb" }, { edgeId: "lb-b" }],
    label: "tools/call",
    caption: "Client invokes a tool",
    tone: "request",
  },
  {
    id: "tool-call-progress-one",
    route: [
      { edgeId: "lb-b", reverse: true },
      { edgeId: "client-lb", reverse: true },
    ],
    label: "notifications/progress 33%",
    caption: "Server reports progress",
    tone: "notify",
  },
  {
    id: "tool-call-progress-two",
    route: [
      { edgeId: "lb-b", reverse: true },
      { edgeId: "client-lb", reverse: true },
    ],
    label: "notifications/progress 80%",
    caption: "Server reports more progress",
    tone: "notify",
  },
  {
    id: "tool-call-result",
    route: [
      { edgeId: "lb-b", reverse: true },
      { edgeId: "client-lb", reverse: true },
    ],
    label: "CallToolResult",
    caption: "Server completes the tool call",
    tone: "result",
  },
];

const nodeById = new Map(nodes.map((node) => [node.id, node]));
const edgeById = new Map(edges.map((edge) => [edge.id, edge]));

const event = ref<"idle" | "initialize" | "request" | "tool">("idle");
const initialized = ref(false);
const animationKey = ref(0);
const activeStepIndex = ref(-1);
const packetProgress = ref(0);
const diagnostics = ref(false);
const timers: number[] = [];
let animationFrame = 0;
const PULSE_DURATION_MS = 1500;
const STEP_INTERVAL_MS = 1800;

const activeSteps = computed(() => {
  if (event.value === "initialize") return initializeSteps;
  if (event.value === "request") return requestSteps;
  if (event.value === "tool") return toolCallSteps;
  return [];
});

const activeStep = computed(() => activeSteps.value[activeStepIndex.value]);
const isPlaying = computed(() => event.value !== "idle");
const serverKnowsClient = computed(
  () =>
    initialized.value ||
    (event.value === "initialize" &&
      (activeStepIndex.value > 0 ||
        (activeStepIndex.value === 0 && packetProgress.value >= 0.92))),
);
const clientKnowsServer = computed(
  () =>
    initialized.value ||
    (event.value === "initialize" &&
      (activeStepIndex.value > 1 ||
        (activeStepIndex.value === 1 && packetProgress.value >= 0.92))),
);
const isLocked = computed(
  () =>
    initialized.value ||
    (event.value === "initialize" &&
      (activeStepIndex.value > 2 ||
        (activeStepIndex.value === 2 && packetProgress.value >= 0.92))),
);

const status = computed(() => {
  if (activeStep.value) return activeStep.value.caption;
  if (event.value === "initialize") return "Initializing session…";
  if (event.value === "request") return "Listing tools and prompts with established session state…";
  if (event.value === "tool") return "Calling a tool with progress notifications…";
  if (initialized.value) return "Initialized: each endpoint now has capability state from the other side.";
  return "Not initialized: no endpoint capability state has been established.";
});

const activeNodeIds = computed(() => {
  if (!activeLeg.value) return new Set<string>();

  const { edge, reverse, progress } = activeLeg.value;
  const source = reverse ? edge.to : edge.from;
  const target = reverse ? edge.from : edge.to;

  if (progress < 0.18) return new Set([source]);
  if (progress > 0.82) return new Set([target]);
  return new Set<string>();
});

const activeLeg = computed(() => {
  const route = activeStep.value?.route;
  if (!route?.length) return undefined;

  const scaled = packetProgress.value * route.length;
  const index =
    packetProgress.value >= 1
      ? route.length - 1
      : Math.min(route.length - 1, Math.floor(scaled));
  const leg = route[index];
  const edge = edgeById.get(leg.edgeId);

  if (!edge) return undefined;

  return {
    edge,
    reverse: leg.reverse ?? false,
    progress: packetProgress.value >= 1 ? 1 : scaled - index,
  };
});

const pulsePath = computed(() => {
  if (!activeLeg.value) return "";

  const tail = Math.max(0, activeLeg.value.progress - 0.28);
  const head = Math.min(1, activeLeg.value.progress);

  return cubicSegmentPath(activeLeg.value.edge, activeLeg.value.reverse, tail, head);
});

const diagnosticFrame = computed(() => Math.round(packetProgress.value * 60));
const diagnosticStep = computed(() =>
  activeStep.value ? `${activeStepIndex.value + 1}/${activeSteps.value.length}` : "idle",
);

function anchorPoint(node: NodeSpec, anchor: Anchor) {
  return {
    x: anchor === "left" ? node.x : node.x + node.w,
    y: node.y + node.h / 2,
  };
}

function edgePath(edge: EdgeSpec, reverse = false) {
  const { a, c1, c2, b } = edgePoints(edge, reverse);

  return `M ${a.x} ${a.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${b.x} ${b.y}`;
}

function edgePoints(edge: EdgeSpec, reverse = false) {
  const from = nodeById.get(reverse ? edge.to : edge.from)!;
  const to = nodeById.get(reverse ? edge.from : edge.to)!;
  const a = anchorPoint(from, reverse ? edge.toAnchor : edge.fromAnchor);
  const b = anchorPoint(to, reverse ? edge.fromAnchor : edge.toAnchor);
  const dx = Math.abs(b.x - a.x);
  const direction = b.x >= a.x ? 1 : -1;

  return {
    a,
    c1: { x: a.x + direction * dx * 0.45, y: a.y },
    c2: { x: b.x - direction * dx * 0.45, y: b.y },
    b,
  };
}

function cubicPoint(
  a: { x: number; y: number },
  c1: { x: number; y: number },
  c2: { x: number; y: number },
  b: { x: number; y: number },
  t: number,
) {
  const u = 1 - t;

  return {
    x: u ** 3 * a.x + 3 * u ** 2 * t * c1.x + 3 * u * t ** 2 * c2.x + t ** 3 * b.x,
    y: u ** 3 * a.y + 3 * u ** 2 * t * c1.y + 3 * u * t ** 2 * c2.y + t ** 3 * b.y,
  };
}

function cubicDerivative(
  a: { x: number; y: number },
  c1: { x: number; y: number },
  c2: { x: number; y: number },
  b: { x: number; y: number },
  t: number,
) {
  const u = 1 - t;

  return {
    x:
      3 * u ** 2 * (c1.x - a.x) +
      6 * u * t * (c2.x - c1.x) +
      3 * t ** 2 * (b.x - c2.x),
    y:
      3 * u ** 2 * (c1.y - a.y) +
      6 * u * t * (c2.y - c1.y) +
      3 * t ** 2 * (b.y - c2.y),
  };
}

function cubicSegmentPath(edge: EdgeSpec, reverse = false, start: number, end: number) {
  const { a, c1, c2, b } = edgePoints(edge, reverse);
  const t0 = Math.max(0, Math.min(1, start));
  const t1 = Math.max(t0 + 0.001, Math.min(1, end));
  const p0 = cubicPoint(a, c1, c2, b, t0);
  const p1 = cubicPoint(a, c1, c2, b, t1);
  const d0 = cubicDerivative(a, c1, c2, b, t0);
  const d1 = cubicDerivative(a, c1, c2, b, t1);
  const dt = t1 - t0;
  const s1 = { x: p0.x + (d0.x * dt) / 3, y: p0.y + (d0.y * dt) / 3 };
  const s2 = { x: p1.x - (d1.x * dt) / 3, y: p1.y - (d1.y * dt) / 3 };

  return `M ${p0.x} ${p0.y} C ${s1.x} ${s1.y}, ${s2.x} ${s2.y}, ${p1.x} ${p1.y}`;
}

function clearTimers() {
  while (timers.length) {
    window.clearTimeout(timers.pop());
  }
  window.cancelAnimationFrame(animationFrame);
  animationFrame = 0;
  packetProgress.value = 0;
}

function freeze() {
  while (timers.length) {
    window.clearTimeout(timers.pop());
  }
  window.cancelAnimationFrame(animationFrame);
  animationFrame = 0;
}

function animatePacket(start = performance.now()) {
  const elapsed = performance.now() - start;
  packetProgress.value = Math.min(elapsed / PULSE_DURATION_MS, 1);

  if (packetProgress.value < 1) {
    animationFrame = window.requestAnimationFrame(() => animatePacket(start));
  }
}

function play(nextEvent: "initialize" | "request" | "tool") {
  clearTimers();
  event.value = nextEvent;
  activeStepIndex.value = -1;
  animationKey.value += 1;

  const steps =
    nextEvent === "initialize"
      ? initializeSteps
      : nextEvent === "request"
        ? requestSteps
        : toolCallSteps;

  for (const index of steps.keys()) {
    timers.push(
      window.setTimeout(() => {
        activeStepIndex.value = index;
        packetProgress.value = 0;
        animationKey.value += 1;
        window.cancelAnimationFrame(animationFrame);
        animationFrame = window.requestAnimationFrame(() => animatePacket());
      }, index * STEP_INTERVAL_MS),
    );
  }

  timers.push(
    window.setTimeout(
      () => {
        if (nextEvent === "initialize") initialized.value = true;
        activeStepIndex.value = -1;
        packetProgress.value = 0;
        event.value = "idle";
      },
      steps.length * STEP_INTERVAL_MS + 650,
    ),
  );
}

function seekStep(delta: number) {
  freeze();
  const steps = activeSteps.value.length ? activeSteps.value : initializeSteps;
  event.value = event.value === "idle" ? "initialize" : event.value;
  activeStepIndex.value = Math.max(
    0,
    Math.min(steps.length - 1, activeStepIndex.value + delta),
  );
  packetProgress.value = 0.5;
  animationKey.value += 1;
}

function seekFrame(delta: number) {
  freeze();
  if (event.value === "idle") {
    event.value = "initialize";
    activeStepIndex.value = 0;
  }
  packetProgress.value = Math.max(0, Math.min(1, packetProgress.value + delta / 60));
  animationKey.value += 1;
}

function reset() {
  clearTimers();
  event.value = "idle";
  initialized.value = false;
  activeStepIndex.value = -1;
  packetProgress.value = 0;
  animationKey.value += 1;
}

onBeforeUnmount(clearTimers);
</script>

<template>
  <section
    class="remote-mcp-story"
    :class="[
      `remote-mcp-story--${event}`,
      {
        'remote-mcp-story--playing': isPlaying,
        'remote-mcp-story--server-knows-client': serverKnowsClient,
        'remote-mcp-story--client-knows-server': clientKnowsServer,
        'remote-mcp-story--locked': isLocked,
        'remote-mcp-story--initialized': initialized,
      },
    ]"
    aria-labelledby="remote-mcp-story-title"
    aria-describedby="remote-mcp-story-desc"
  >
    <div class="remote-mcp-story__controls" aria-label="Remote MCP animation controls">
      <button type="button" :aria-pressed="event === 'initialize'" @click="play('initialize')">
        Initialize
      </button>
      <button type="button" :aria-pressed="event === 'request'" @click="play('request')">
        Later request
      </button>
      <button type="button" :aria-pressed="event === 'tool'" @click="play('tool')">
        Tool call
      </button>
      <button type="button" @click="reset">Reset state</button>
      <button type="button" :aria-pressed="diagnostics" @click="diagnostics = !diagnostics">
        Diag
      </button>
    </div>

    <div v-if="diagnostics" class="remote-mcp-story__diagnostics">
      <div>
        <strong>{{ event }}</strong>
        <span>step {{ diagnosticStep }}</span>
        <span>frame {{ diagnosticFrame }}/60</span>
        <span>{{ Math.round(packetProgress * 100) }}%</span>
      </div>
      <div>
        <button type="button" @click="freeze">freeze</button>
        <button type="button" @click="seekStep(-1)">step −</button>
        <button type="button" @click="seekStep(1)">step +</button>
        <button type="button" @click="seekFrame(-5)">frame −</button>
        <button type="button" @click="seekFrame(5)">frame +</button>
      </div>
    </div>

    <p v-if="!activeStep" class="remote-mcp-story__status" aria-live="polite">{{ status }}</p>

    <svg
      class="remote-mcp-story__canvas"
      viewBox="0 0 1000 470"
      role="img"
      aria-labelledby="remote-mcp-story-title"
      aria-describedby="remote-mcp-story-desc"
    >
      <title id="remote-mcp-story-title">Remote MCP through a load balancer</title>
      <desc id="remote-mcp-story-desc">
        A client connects through a load balancer to a pool of MCP servers. The initialize animation
        shows client capabilities and server capabilities becoming retained state at opposite ends.
      </desc>

      <defs>
        <marker id="remote-mcp-story-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(185, 179, 165, 0.42)" />
        </marker>
      </defs>

      <path
        v-for="edge in edges"
        :key="edge.id"
        class="remote-mcp-story__edge"
        :class="{ 'remote-mcp-story__edge--selected': edge.id === 'lb-b' }"
        :d="edgePath(edge)"
        marker-end="url(#remote-mcp-story-arrow)"
      />

      <path
        v-if="activeStep"
        :key="`pulse-glow-${animationKey}-${activeStep.id}`"
        class="remote-mcp-story__pulse remote-mcp-story__pulse--glow"
        :class="`remote-mcp-story__pulse--${activeStep.tone ?? 'request'}`"
        :d="pulsePath"
      />

      <path
        v-if="activeStep"
        :key="`pulse-core-${animationKey}-${activeStep.id}`"
        class="remote-mcp-story__pulse remote-mcp-story__pulse--core"
        :class="`remote-mcp-story__pulse--${activeStep.tone ?? 'request'}`"
        :d="pulsePath"
      />

      <g
        v-if="activeStep"
        :key="`label-${animationKey}-${activeStep.id}`"
        class="remote-mcp-story__message"
        :class="`remote-mcp-story__message--${activeStep.tone ?? 'request'}`"
        transform="translate(220 110)"
      >
        <rect width="344" height="52" rx="14" />
        <text class="remote-mcp-story__message-label" x="18" y="22">in flight</text>
        <text class="remote-mcp-story__message-value" x="18" y="42">{{ activeStep.label }}</text>
      </g>

      <g
        v-for="node in nodes"
        :key="node.id"
        class="remote-mcp-story__node"
        :class="[
          `remote-mcp-story__node--${node.id}`,
          {
            'remote-mcp-story__node--active-server': node.id === 'server-b',
            'remote-mcp-story__node--endpoint': node.id === 'client' || node.id === 'server-b',
            'remote-mcp-story__node--active-hop': activeNodeIds.has(node.id),
            'remote-mcp-story__node--locked-endpoint': isLocked && (node.id === 'client' || node.id === 'server-b'),
          },
        ]"
        :transform="`translate(${node.x} ${node.y})`"
      >
        <rect class="remote-mcp-story__node-glow" :width="node.w" :height="node.h" rx="16" />
        <rect class="remote-mcp-story__node-box" :width="node.w" :height="node.h" rx="16" />
        <text class="remote-mcp-story__role" x="18" y="28">{{ node.role }}</text>
        <text class="remote-mcp-story__title" x="18" y="61">{{ node.title }}</text>
        <text v-if="node.detail" class="remote-mcp-story__detail" x="18" y="88">{{ node.detail }}</text>
      </g>

      <g class="remote-mcp-story__state remote-mcp-story__state--client" transform="translate(48 322)">
        <rect width="276" height="68" rx="16" />
        <text class="remote-mcp-story__state-label" x="18" y="24">client state</text>
        <text class="remote-mcp-story__state-value" x="18" y="49">server capabilities</text>
        <text class="remote-mcp-story__state-extra" x="18" y="64">MCP-Session-Id</text>
      </g>

      <g class="remote-mcp-story__state remote-mcp-story__state--server" transform="translate(678 400)">
        <rect width="276" height="68" rx="16" />
        <text class="remote-mcp-story__state-label" x="18" y="24">server state</text>
        <text class="remote-mcp-story__state-value" x="18" y="49">client capabilities</text>
        <text class="remote-mcp-story__state-extra" x="18" y="64">MCP-Session-Id</text>
      </g>
    </svg>
  </section>
</template>

<style scoped>
.remote-mcp-story {
  container-type: size;
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 8px);
  background:
    radial-gradient(circle at 15% 50%, rgba(245, 164, 0, 0.14), transparent 25%),
    radial-gradient(circle at 83% 50%, rgba(106, 163, 247, 0.11), transparent 28%),
    rgba(20, 22, 27, 0.86);
  box-shadow: var(--deck-shadow);
}

.remote-mcp-story::before {
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

.remote-mcp-story__controls {
  position: absolute;
  top: clamp(0.7rem, 2.8cqh, 1rem);
  right: clamp(0.7rem, 2.8cqw, 1.1rem);
  z-index: 2;
  display: flex;
  gap: 0.42rem;
}

.remote-mcp-story__controls button {
  padding: 0.42rem 0.62rem;
  color: var(--deck-muted);
  border: 1px solid var(--deck-border-2);
  border-radius: 999px;
  background: rgba(11, 12, 15, 0.72);
  font: 800 clamp(0.62rem, 2.1cqh, 0.76rem) / 1 var(--deck-font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.remote-mcp-story__controls button:hover,
.remote-mcp-story__controls button[aria-pressed="true"] {
  color: var(--deck-text);
  border-color: rgba(255, 198, 73, 0.62);
  background: rgba(245, 164, 0, 0.14);
}

.remote-mcp-story__diagnostics {
  position: absolute;
  top: clamp(3.45rem, 10cqh, 4.15rem);
  right: clamp(0.7rem, 2.8cqw, 1.1rem);
  z-index: 5;
  min-width: 18.5rem;
  padding: 0.52rem 0.6rem;
  border: 1px solid rgba(106, 163, 247, 0.38);
  border-radius: var(--deck-radius);
  background: rgba(11, 12, 15, 0.84);
  color: var(--deck-muted);
  font: 750 0.62rem / 1.35 var(--deck-font-mono);
  letter-spacing: 0.04em;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.32);
}

.remote-mcp-story__diagnostics > div {
  display: flex;
  flex-wrap: wrap;
  gap: 0.38rem 0.52rem;
  align-items: center;
}

.remote-mcp-story__diagnostics > div + div {
  margin-top: 0.42rem;
}

.remote-mcp-story__diagnostics strong {
  color: var(--deck-info);
  text-transform: uppercase;
}

.remote-mcp-story__diagnostics button {
  padding: 0.22rem 0.38rem;
  color: var(--deck-muted);
  border: 1px solid var(--deck-border-2);
  border-radius: 999px;
  background: rgba(20, 22, 27, 0.84);
  font: inherit;
  cursor: pointer;
}

.remote-mcp-story__diagnostics button:hover {
  color: var(--deck-text);
  border-color: rgba(106, 163, 247, 0.62);
}

.remote-mcp-story__status {
  position: absolute;
  left: clamp(0.85rem, 2.7cqw, 1.1rem);
  bottom: clamp(0.7rem, 2.5cqh, 1rem);
  z-index: 2;
  max-width: 46rem;
  margin: 0;
  padding: 0.44rem 0.7rem;
  color: var(--deck-muted);
  border: 1px solid var(--deck-border);
  border-radius: 999px;
  background: rgba(11, 12, 15, 0.68);
  font-size: clamp(0.58rem, 1.9cqh, 0.72rem);
  font-weight: 750;
}

.remote-mcp-story__canvas {
  position: relative;
  z-index: 1;
  width: 100%;
  height: 100%;
}

.remote-mcp-story__edge {
  fill: none;
  stroke: rgba(185, 179, 165, 0.22);
  stroke-dasharray: 6 12;
  stroke-linecap: round;
  stroke-width: 3.5;
}

.remote-mcp-story__edge--selected {
  stroke: rgba(106, 163, 247, 0.28);
}

.remote-mcp-story__pulse {
  fill: none;
  opacity: 1;
  stroke-linecap: round;
}

.remote-mcp-story__pulse--glow {
  stroke: rgba(255, 198, 73, 0.28);
  stroke-width: 20;
}

.remote-mcp-story__pulse--core {
  stroke: var(--deck-accent-hi);
  stroke-width: 9.5;
}

.remote-mcp-story__pulse--result.remote-mcp-story__pulse--glow {
  stroke: rgba(106, 163, 247, 0.28);
}

.remote-mcp-story__pulse--result.remote-mcp-story__pulse--core {
  stroke: var(--deck-info);
}

.remote-mcp-story__pulse--notify.remote-mcp-story__pulse--glow {
  stroke: rgba(106, 209, 156, 0.28);
}

.remote-mcp-story__pulse--notify.remote-mcp-story__pulse--core {
  stroke: var(--deck-ok);
}

.remote-mcp-story__message {
  opacity: 1;
  filter: drop-shadow(0 14px 26px rgba(0, 0, 0, 0.34));
  animation: remote-mcp-story-message 220ms ease-out both;
}

.remote-mcp-story__message rect {
  fill: rgba(11, 12, 15, 0.86);
  stroke: rgba(255, 198, 73, 0.42);
  stroke-width: 1.2;
}

.remote-mcp-story__message--result rect {
  stroke: rgba(106, 163, 247, 0.55);
}

.remote-mcp-story__message--notify rect {
  stroke: rgba(106, 209, 156, 0.52);
}

.remote-mcp-story__message-label {
  fill: var(--deck-dim);
  font: 850 10px / 1 var(--deck-font-mono);
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.remote-mcp-story__message-value {
  fill: var(--deck-text);
  font: 850 16px / 1 var(--deck-font-mono);
  letter-spacing: -0.035em;
}

.remote-mcp-story__node-box,
.remote-mcp-story__state rect {
  fill: rgba(20, 22, 27, 0.94);
  stroke: var(--deck-border-2);
  stroke-width: 1.2;
}

.remote-mcp-story__node--lb .remote-mcp-story__node-box,
.remote-mcp-story__node--server-b .remote-mcp-story__node-box {
  fill: rgba(18, 24, 32, 0.95);
}

.remote-mcp-story__node--active-server .remote-mcp-story__node-box {
  stroke: rgba(106, 163, 247, 0.58);
}

.remote-mcp-story__node-glow {
  opacity: 0;
  fill: transparent;
  stroke: rgba(255, 198, 73, 0.62);
  stroke-width: 2;
  filter: drop-shadow(0 0 18px rgba(255, 198, 73, 0.68));
  transform-box: fill-box;
  transform-origin: center;
}

.remote-mcp-story__node--active-hop .remote-mcp-story__node-glow {
  animation: remote-mcp-story-node 1.18s ease-out both;
}

.remote-mcp-story__node--locked-endpoint .remote-mcp-story__node-box {
  stroke: var(--deck-ok);
  stroke-width: 3.4;
  filter: drop-shadow(0 0 18px rgba(106, 209, 156, 0.42));
}

.remote-mcp-story__role,
.remote-mcp-story__state-label,
.remote-mcp-story__step-labels {
  fill: var(--deck-dim);
  font: 850 12px / 1 var(--deck-font-mono);
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.remote-mcp-story__title {
  fill: var(--deck-text);
  font: 750 25px / 1 var(--deck-font-mono);
  letter-spacing: -0.06em;
}

.remote-mcp-story__detail {
  fill: var(--deck-muted);
  font: 650 13px / 1 var(--deck-font-mono);
}

.remote-mcp-story__state {
  opacity: 0.18;
  transition:
    opacity 320ms ease,
    filter 320ms ease,
    transform 320ms ease;
}

.remote-mcp-story__state rect {
  fill: rgba(11, 12, 15, 0.72);
  stroke: rgba(245, 164, 0, 0.34);
}

.remote-mcp-story--client-knows-server .remote-mcp-story__state--client,
.remote-mcp-story--server-knows-client .remote-mcp-story__state--server {
  opacity: 1;
  filter: drop-shadow(0 0 18px rgba(255, 198, 73, 0.18));
}

.remote-mcp-story__state-value {
  fill: var(--deck-accent-hi);
  font: 820 18px / 1 var(--deck-font-mono);
  letter-spacing: -0.04em;
}

.remote-mcp-story__state-extra {
  fill: var(--deck-muted);
  font: 760 13px / 1 var(--deck-font-mono);
  letter-spacing: -0.035em;
}

@keyframes remote-mcp-story-message {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

@keyframes remote-mcp-story-node {
  0% {
    opacity: 0;
    transform: scale(0.97);
  }
  35% {
    opacity: 1;
    transform: scale(1.04);
  }
  100% {
    opacity: 0;
    transform: scale(1.13);
  }
}

</style>
