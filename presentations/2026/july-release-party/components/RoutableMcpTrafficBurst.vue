<script setup lang="ts">
type Anchor = "left" | "right";

type NodeSpec = {
  id: string;
  role: string;
  title: string;
  x: number;
  y: number;
  w: number;
  h: number;
};

type Packet = {
  id: number;
  target: string;
  delay: number;
  duration: number;
  lane: number;
  tone: "hot" | "cool" | "ok";
};

const width = 1200;
const height = 520;

const nodes: NodeSpec[] = [
  { id: "client", role: "clients", title: "MCP clients", x: 52, y: 205, w: 214, h: 110 },
  { id: "edge", role: "http edge", title: "Router", x: 450, y: 204, w: 218, h: 112 },
  { id: "server-a", role: "sandbox runtime", title: "Sandbox 01", x: 875, y: 52, w: 214, h: 78 },
  { id: "server-b", role: "sandbox runtime", title: "Sandbox 02", x: 875, y: 154, w: 214, h: 78 },
  { id: "server-c", role: "sandbox runtime", title: "Sandbox 03", x: 875, y: 256, w: 214, h: 78 },
  { id: "server-d", role: "sandbox runtime", title: "Sandbox 04", x: 875, y: 358, w: 214, h: 78 },
];

const nodeById = new Map(nodes.map((node) => [node.id, node]));
const targets = ["server-c", "server-a", "server-d", "server-b", "server-a", "server-c", "server-d", "server-b"];
const tones = ["hot", "cool", "ok"] as const;
const packets: Packet[] = Array.from({ length: 68 }, (_, id) => ({
  id,
  target: targets[id % targets.length],
  delay: -(id * 0.115 + (id % 5) * 0.021),
  duration: 2.05 + (id % 7) * 0.08,
  lane: (id % 5) - 2,
  tone: tones[id % tones.length],
}));

function anchorPoint(node: NodeSpec, anchor: Anchor, lane = 0) {
  return {
    x: anchor === "left" ? node.x : node.x + node.w,
    y: node.y + node.h / 2 + lane * 4.4,
  };
}

function segment(fromId: string, toId: string, lane = 0) {
  const from = nodeById.get(fromId)!;
  const to = nodeById.get(toId)!;
  const a = anchorPoint(from, "right", lane);
  const b = anchorPoint(to, "left", lane);
  const dx = b.x - a.x;

  return `M ${a.x} ${a.y} C ${a.x + dx * 0.46} ${a.y}, ${b.x - dx * 0.42} ${b.y}, ${b.x} ${b.y}`;
}

function routePath(packet: Packet) {
  const client = nodeById.get("client")!;
  const edge = nodeById.get("edge")!;
  const server = nodeById.get(packet.target)!;
  const a = anchorPoint(client, "right", packet.lane);
  const b = anchorPoint(edge, "left", packet.lane);
  const c = anchorPoint(edge, "right", packet.lane);
  const d = anchorPoint(server, "left", packet.lane);
  const dx1 = b.x - a.x;
  const dx2 = d.x - c.x;

  return [
    `M ${a.x} ${a.y}`,
    `C ${a.x + dx1 * 0.44} ${a.y}, ${b.x - dx1 * 0.44} ${b.y}, ${b.x} ${b.y}`,
    `L ${c.x} ${c.y}`,
    `C ${c.x + dx2 * 0.4} ${c.y}, ${d.x - dx2 * 0.44} ${d.y}, ${d.x} ${d.y}`,
  ].join(" ");
}
</script>

<template>
  <section class="routable-burst" aria-labelledby="routable-burst-title">
    <svg class="routable-burst__canvas" :viewBox="`0 0 ${width} ${height}`" role="img">
      <title id="routable-burst-title">High-volume sandbox tool calls routed by explicit state handles</title>
      <defs>
        <filter id="routable-burst-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3.2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <linearGradient id="routable-burst-edge" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="rgba(255, 198, 73, 0.28)" />
          <stop offset="55%" stop-color="rgba(106, 163, 247, 0.34)" />
          <stop offset="100%" stop-color="rgba(106, 209, 156, 0.3)" />
        </linearGradient>
        <marker id="routable-burst-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(106, 209, 156, 0.5)" />
        </marker>
      </defs>

      <g class="routable-burst__edges">
        <path :d="segment('client', 'edge')" marker-end="url(#routable-burst-arrow)" />
        <path
          v-for="target in ['server-a', 'server-b', 'server-c', 'server-d']"
          :key="target"
          :d="segment('edge', target)"
          marker-end="url(#routable-burst-arrow)"
        />
      </g>

      <path
        v-for="packet in packets"
        :id="`routable-burst-route-${packet.id}`"
        :key="`route-${packet.id}`"
        class="routable-burst__route"
        :d="routePath(packet)"
      />

      <g class="routable-burst__packets" filter="url(#routable-burst-glow)">
        <circle
          v-for="packet in packets"
          :key="packet.id"
          r="7.5"
          class="routable-burst__packet"
          :class="`routable-burst__packet--${packet.tone}`"
        >
          <animateMotion
            :dur="`${packet.duration}s`"
            :begin="`${packet.delay}s`"
            repeatCount="indefinite"
            rotate="auto"
          >
            <mpath :href="`#routable-burst-route-${packet.id}`" />
          </animateMotion>
        </circle>
      </g>

      <g class="routable-burst__junction" transform="translate(690 260)">
        <circle r="13" />
        <circle r="4.5" />
        <text x="22" y="-10">routed by</text>
        <text x="22" y="10">Sandbox-Id</text>
      </g>

      <g class="routable-burst__callout routable-burst__callout--cache" transform="translate(56 334)">
        <path d="M 100 0 L 100 -22" />
        <rect width="220" height="54" rx="14" />
        <text x="18" y="22">clients keep</text>
        <text x="18" y="42">Cached Tool List</text>
      </g>

      <g
        v-for="node in nodes"
        :key="node.id"
        class="routable-burst__node"
        :class="`routable-burst__node--${node.id}`"
        :transform="`translate(${node.x} ${node.y})`"
      >
        <rect :width="node.w" :height="node.h" rx="16" />
        <text class="routable-burst__role" x="18" y="27">{{ node.role }}</text>
        <text class="routable-burst__title" x="18" :y="node.h > 90 ? 68 : 55">{{ node.title }}</text>
      </g>

      <g class="routable-burst__callout routable-burst__callout--method" transform="translate(436 344)">
        <path d="M 122 0 L 122 -30 L 226 -92" />
        <rect width="246" height="54" rx="14" />
        <text x="18" y="22">each call exposes</text>
        <text x="18" y="42">method + target + handle</text>
      </g>

      <g class="routable-burst__metrics" transform="translate(74 64)">
        <text class="routable-burst__metrics-kicker" x="0" y="0">stateless hot path</text>
        <text class="routable-burst__metrics-main" x="0" y="43">High volume, no protocol session</text>
        <text class="routable-burst__metrics-sub" x="0" y="76">explicit state routes each independent tool call</text>
      </g>
    </svg>
  </section>
</template>

<style scoped>
.routable-burst {
  container-type: size;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 8px);
  background:
    radial-gradient(circle at 20% 42%, rgba(255, 198, 73, 0.08), transparent 28%),
    radial-gradient(circle at 82% 50%, rgba(106, 209, 156, 0.08), transparent 32%),
    rgba(255, 255, 255, 0.86);
  box-shadow: var(--deck-shadow);
}

.routable-burst__canvas {
  width: 100%;
  height: 100%;
}

.routable-burst__edges path {
  fill: none;
  stroke: url(#routable-burst-edge);
  stroke-dasharray: 7 11;
  stroke-linecap: round;
  stroke-width: 3.4;
}

.routable-burst__junction circle:first-child {
  fill: rgba(106, 163, 247, 0.12);
  stroke: rgba(106, 209, 156, 0.44);
  stroke-width: 1.8;
}

.routable-burst__junction circle:last-child {
  fill: var(--deck-ok);
}

.routable-burst__junction text {
  fill: var(--deck-ok);
  font: 850 13px / 1 var(--deck-font-mono);
  letter-spacing: -0.025em;
}

.routable-burst__route {
  fill: none;
  stroke: transparent;
}

.routable-burst__packet {
  opacity: 0.92;
}

.routable-burst__packet--hot {
  fill: var(--deck-accent-hi);
}

.routable-burst__packet--cool {
  fill: var(--deck-info);
}

.routable-burst__packet--ok {
  fill: var(--deck-ok);
}

.routable-burst__node rect,
.routable-burst__callout rect {
  fill: rgba(243, 244, 246, 0.82);
  stroke: var(--deck-border-2);
  stroke-width: 1.3;
}

.routable-burst__node--edge rect {
  fill: rgba(18, 24, 32, 0.94);
  stroke: rgba(106, 163, 247, 0.62);
  stroke-width: 2.4;
  filter: drop-shadow(0 0 14px rgba(106, 163, 247, 0.18));
}

.routable-burst__node--edge .routable-burst__role {
  fill: #d1d5db;
}

.routable-burst__node--edge .routable-burst__title {
  fill: #ffffff;
}

.routable-burst__node--server-a rect,
.routable-burst__node--server-b rect,
.routable-burst__node--server-c rect,
.routable-burst__node--server-d rect {
  stroke: rgba(106, 209, 156, 0.48);
}

.routable-burst__role,
.routable-burst__callout text:first-of-type,
.routable-burst__metrics-kicker {
  fill: var(--deck-dim);
  font: 850 13px / 1 var(--deck-font-mono);
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.routable-burst__title {
  fill: var(--deck-text);
  font: 760 27px / 1 var(--deck-font-mono);
  letter-spacing: -0.058em;
}

.routable-burst__callout rect {
  stroke: rgba(255, 198, 73, 0.46);
}

.routable-burst__callout path {
  fill: none;
  stroke: rgba(255, 198, 73, 0.34);
  stroke-dasharray: 4 7;
  stroke-linecap: round;
  stroke-width: 1.5;
}

.routable-burst__callout text:last-of-type {
  fill: var(--deck-accent-hi);
  font: 850 17px / 1 var(--deck-font-mono);
  letter-spacing: -0.04em;
}

.routable-burst__metrics-main {
  fill: var(--deck-text);
  font: 850 36px / 1 var(--deck-font-mono);
  letter-spacing: -0.07em;
}

.routable-burst__metrics-sub {
  fill: var(--deck-muted);
  font: 730 17px / 1 var(--deck-font-mono);
  letter-spacing: -0.035em;
}
</style>
