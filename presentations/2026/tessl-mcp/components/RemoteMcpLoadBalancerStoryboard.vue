<script setup lang="ts">
import { computed, nextTick, ref } from "vue";

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

const nodes: NodeSpec[] = [
  {
    id: "client",
    role: "MCP client",
    title: "Client",
    detail: "initiates JSON-RPC",
    x: 56,
    y: 180,
    w: 188,
    h: 112,
  },
  {
    id: "lb",
    role: "remote edge",
    title: "Load balancer",
    detail: "routes each HTTP request",
    x: 406,
    y: 185,
    w: 188,
    h: 102,
  },
  {
    id: "server-a",
    role: "worker 1",
    title: "Server 01",
    detail: "healthy",
    x: 766,
    y: 72,
    w: 184,
    h: 92,
  },
  {
    id: "server-b",
    role: "worker 2",
    title: "Server 02",
    detail: "selected",
    x: 766,
    y: 190,
    w: 184,
    h: 92,
  },
  {
    id: "server-c",
    role: "worker 3",
    title: "Server 03",
    detail: "healthy",
    x: 766,
    y: 308,
    w: 184,
    h: 92,
  },
];

const edges: EdgeSpec[] = [
  { id: "client-lb", from: "client", to: "lb", fromAnchor: "right", toAnchor: "left" },
  { id: "lb-a", from: "lb", to: "server-a", fromAnchor: "right", toAnchor: "left" },
  { id: "lb-b", from: "lb", to: "server-b", fromAnchor: "right", toAnchor: "left" },
  { id: "lb-c", from: "lb", to: "server-c", fromAnchor: "right", toAnchor: "left" },
];

const initializeSteps = [
  ["client-lb", "initialize"],
  ["lb-b", "route"],
  ["lb-b", "result"],
  ["client-lb", "initialized"],
] as const;

const requestSteps = [
  ["client-lb", "tools/call"],
  ["lb-b", "route"],
] as const;

const nodeById = new Map(nodes.map((node) => [node.id, node]));
const edgeById = new Map(edges.map((edge) => [edge.id, edge]));

const event = ref<"idle" | "initialize" | "request">("idle");
const initialized = ref(false);
const animationKey = ref(0);

const activeSteps = computed(() => {
  if (event.value === "initialize") return initializeSteps;
  if (event.value === "request") return requestSteps;
  return [];
});

const status = computed(() => {
  if (event.value === "initialize") return "Initializing session: capabilities are exchanged and retained as endpoint state.";
  if (event.value === "request") return "A later request can be routed to a different worker unless that state is shared or removed.";
  if (initialized.value) return "Initialized: each endpoint now has capability state from the other side.";
  return "Not initialized: no endpoint capability state has been established.";
});

function anchorPoint(node: NodeSpec, anchor: Anchor) {
  return {
    x: anchor === "left" ? node.x : node.x + node.w,
    y: node.y + node.h / 2,
  };
}

function edgePath(edge: EdgeSpec, reverse = false) {
  const from = nodeById.get(reverse ? edge.to : edge.from)!;
  const to = nodeById.get(reverse ? edge.from : edge.to)!;
  const a = anchorPoint(from, reverse ? edge.toAnchor : edge.fromAnchor);
  const b = anchorPoint(to, reverse ? edge.fromAnchor : edge.toAnchor);
  const dx = Math.abs(b.x - a.x);

  return `M ${a.x} ${a.y} C ${a.x + dx * 0.45} ${a.y}, ${b.x - dx * 0.45} ${b.y}, ${b.x} ${b.y}`;
}

function pathFor(edgeId: string, label = "") {
  const edge = edgeById.get(edgeId)!;
  return edgePath(edge, label === "result" || label === "initialized");
}

async function play(nextEvent: "initialize" | "request") {
  event.value = "idle";
  animationKey.value += 1;
  await nextTick();
  event.value = nextEvent;

  window.setTimeout(() => {
    if (nextEvent === "initialize") initialized.value = true;
    event.value = "idle";
  }, nextEvent === "initialize" ? 5200 : 2600);
}

function reset() {
  event.value = "idle";
  initialized.value = false;
  animationKey.value += 1;
}
</script>

<template>
  <section
    class="remote-mcp-story"
    :class="[`remote-mcp-story--${event}`, { 'remote-mcp-story--initialized': initialized }]"
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
      <button type="button" @click="reset">Reset state</button>
    </div>

    <p class="remote-mcp-story__status" aria-live="polite">{{ status }}</p>

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
        <linearGradient id="remote-mcp-story-pulse" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="rgba(255, 198, 73, 0)" />
          <stop offset="48%" stop-color="rgba(255, 198, 73, 0.96)" />
          <stop offset="100%" stop-color="rgba(106, 163, 247, 0.72)" />
        </linearGradient>
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
        v-for="(step, index) in activeSteps"
        :key="`${animationKey}-${step[0]}-${step[1]}-${index}`"
        class="remote-mcp-story__pulse"
        :style="{ '--step-delay': `${index * 1.12}s` }"
        :d="pathFor(step[0], step[1])"
      />

      <circle
        v-for="(step, index) in activeSteps"
        :key="`${animationKey}-packet-${step[0]}-${step[1]}-${index}`"
        class="remote-mcp-story__packet"
        r="7.5"
        :style="{ '--step-delay': `${index * 1.12}s` }"
      >
        <animateMotion dur="0.86s" :begin="`${index * 1.12}s`" fill="freeze" :path="pathFor(step[0], step[1])" />
      </circle>

      <g
        v-for="node in nodes"
        :key="node.id"
        class="remote-mcp-story__node"
        :class="[
          `remote-mcp-story__node--${node.id}`,
          {
            'remote-mcp-story__node--active-server': node.id === 'server-b',
            'remote-mcp-story__node--endpoint': node.id === 'client' || node.id === 'server-b',
          },
        ]"
        :transform="`translate(${node.x} ${node.y})`"
      >
        <rect class="remote-mcp-story__node-glow" :width="node.w" :height="node.h" rx="16" />
        <rect class="remote-mcp-story__node-box" :width="node.w" :height="node.h" rx="16" />
        <text class="remote-mcp-story__role" x="18" y="28">{{ node.role }}</text>
        <text class="remote-mcp-story__title" x="18" y="61">{{ node.title }}</text>
        <text class="remote-mcp-story__detail" x="18" y="88">{{ node.detail }}</text>
      </g>

      <g class="remote-mcp-story__state remote-mcp-story__state--client" transform="translate(48 326)">
        <rect width="204" height="82" rx="14" />
        <text class="remote-mcp-story__state-label" x="16" y="27">client state</text>
        <text class="remote-mcp-story__state-value" x="16" y="57">server capabilities</text>
      </g>

      <g class="remote-mcp-story__state remote-mcp-story__state--server" transform="translate(748 404)">
        <rect width="220" height="54" rx="14" />
        <text class="remote-mcp-story__state-label" x="16" y="20">server state</text>
        <text class="remote-mcp-story__state-value" x="16" y="43">client capabilities</text>
      </g>

      <g class="remote-mcp-story__step-labels" aria-hidden="true">
        <text x="314" y="166">initialize</text>
        <text x="630" y="154">server capabilities</text>
        <text x="301" y="323">initialized</text>
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
  opacity: 0;
  stroke: url("#remote-mcp-story-pulse");
  stroke-dasharray: 0 900;
  stroke-linecap: round;
  stroke-width: 7;
  filter: drop-shadow(0 0 10px rgba(255, 198, 73, 0.6));
  animation: remote-mcp-story-travel 0.86s ease-out var(--step-delay) both;
}

.remote-mcp-story__packet {
  opacity: 0;
  fill: var(--deck-accent-hi);
  filter: drop-shadow(0 0 10px rgba(255, 198, 73, 0.8));
  animation: remote-mcp-story-packet 0.86s ease-out var(--step-delay) both;
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
}

.remote-mcp-story--initialize .remote-mcp-story__node--client .remote-mcp-story__node-glow,
.remote-mcp-story--initialize .remote-mcp-story__node--lb .remote-mcp-story__node-glow,
.remote-mcp-story--initialize .remote-mcp-story__node--server-b .remote-mcp-story__node-glow,
.remote-mcp-story--request .remote-mcp-story__node--client .remote-mcp-story__node-glow,
.remote-mcp-story--request .remote-mcp-story__node--lb .remote-mcp-story__node-glow,
.remote-mcp-story--request .remote-mcp-story__node--server-b .remote-mcp-story__node-glow {
  animation: remote-mcp-story-node 0.9s ease-out both;
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
  opacity: 0.34;
  transition: opacity 260ms ease;
}

.remote-mcp-story__state rect {
  fill: rgba(11, 12, 15, 0.62);
  stroke: rgba(245, 164, 0, 0.24);
}

.remote-mcp-story--initialized .remote-mcp-story__state,
.remote-mcp-story--initialize .remote-mcp-story__state {
  opacity: 1;
}

.remote-mcp-story__state-value {
  fill: var(--deck-accent-hi);
  font: 800 18px / 1 var(--deck-font-mono);
  letter-spacing: -0.04em;
}

.remote-mcp-story__step-labels {
  opacity: 0;
  fill: var(--deck-accent-hi);
}

.remote-mcp-story--initialize .remote-mcp-story__step-labels {
  animation: remote-mcp-story-labels 5.2s ease both;
}

@keyframes remote-mcp-story-travel {
  0% {
    opacity: 0;
    stroke-dasharray: 0 900;
  }
  15%,
  74% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    stroke-dasharray: 340 900;
  }
}

@keyframes remote-mcp-story-packet {
  0%,
  100% {
    opacity: 0;
  }
  12%,
  82% {
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

@keyframes remote-mcp-story-labels {
  0%,
  8%,
  92%,
  100% {
    opacity: 0;
  }
  16%,
  84% {
    opacity: 1;
  }
}
</style>
