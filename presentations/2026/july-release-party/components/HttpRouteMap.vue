<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    mode?: "problem" | "solution";
  }>(),
  { mode: "solution" },
);

type NodeId = "regionA" | "regionB" | "endpoint" | "client";
type Side = "top" | "right" | "bottom" | "left";
type Tone = "blue" | "red";

type DiagramNode = {
  id: NodeId;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  rx: number;
  tone?: Tone;
  kind?: "service" | "endpoint" | "client";
};

type Edge = {
  id: string;
  from: NodeId;
  to: NodeId;
  fromSide: Side;
  toSide: Side;
  tone: Tone;
  dashed?: boolean;
  fromOffsetX?: number;
  toOffsetX?: number;
};

const nodes: Record<NodeId, DiagramNode> = {
  regionA: {
    id: "regionA",
    label: "Hub Query",
    x: 48,
    y: 44,
    w: 180,
    h: 64,
    rx: 18,
    tone: "blue",
    kind: "service",
  },
  regionB: {
    id: "regionB",
    label: "Image Gen",
    x: 412,
    y: 44,
    w: 180,
    h: 64,
    rx: 18,
    tone: "red",
    kind: "service",
  },
  endpoint: {
    id: "endpoint",
    label: "HF MCP Edge",
    x: 48,
    y: 160,
    w: 544,
    h: 64,
    rx: 32,
    kind: "endpoint",
  },
  client: {
    id: "client",
    label: "Client",
    x: 230,
    y: 330,
    w: 180,
    h: 68,
    rx: 34,
    tone: "blue",
    kind: "client",
  },
};

const orderedNodes = [
  nodes.regionA,
  nodes.regionB,
  nodes.endpoint,
  nodes.client,
];

const edges: Edge[] = [
  {
    id: "region-a-request",
    from: "regionA",
    to: "endpoint",
    fromSide: "bottom",
    toSide: "top",
    tone: "blue",
    fromOffsetX: -8,
    toOffsetX: -190,
  },
  {
    id: "region-a-response",
    from: "endpoint",
    to: "regionA",
    fromSide: "top",
    toSide: "bottom",
    tone: "blue",
    dashed: true,
    fromOffsetX: -174,
    toOffsetX: 8,
  },
  {
    id: "region-b-request",
    from: "regionB",
    to: "endpoint",
    fromSide: "bottom",
    toSide: "top",
    tone: "red",
    fromOffsetX: -8,
    toOffsetX: 174,
  },
  {
    id: "region-b-response",
    from: "endpoint",
    to: "regionB",
    fromSide: "top",
    toSide: "bottom",
    tone: "red",
    dashed: true,
    fromOffsetX: 190,
    toOffsetX: 8,
  },
  {
    id: "client-blue-request",
    from: "client",
    to: "endpoint",
    fromSide: "top",
    toSide: "bottom",
    tone: "blue",
    fromOffsetX: -28,
    toOffsetX: -28,
  },
  {
    id: "client-blue-response",
    from: "endpoint",
    to: "client",
    fromSide: "bottom",
    toSide: "top",
    tone: "blue",
    dashed: true,
    fromOffsetX: -14,
    toOffsetX: -14,
  },
  {
    id: "client-red-request",
    from: "client",
    to: "endpoint",
    fromSide: "top",
    toSide: "bottom",
    tone: "red",
    fromOffsetX: 28,
    toOffsetX: 28,
  },
  {
    id: "client-red-response",
    from: "endpoint",
    to: "client",
    fromSide: "bottom",
    toSide: "top",
    tone: "red",
    dashed: true,
    fromOffsetX: 42,
    toOffsetX: 42,
  },
];

function anchor(node: DiagramNode, side: Side, offsetX = 0) {
  if (side === "top") return { x: node.x + node.w / 2 + offsetX, y: node.y };
  if (side === "bottom")
    return { x: node.x + node.w / 2 + offsetX, y: node.y + node.h };
  if (side === "left") return { x: node.x, y: node.y + node.h / 2 };
  return { x: node.x + node.w, y: node.y + node.h / 2 };
}

function edgePath(edge: Edge) {
  const from = anchor(nodes[edge.from], edge.fromSide, edge.fromOffsetX);
  const to = anchor(nodes[edge.to], edge.toSide, edge.toOffsetX);
  return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
}
</script>

<template>
  <section
    class="http-route-map deck-panel"
    :class="`http-route-map--${props.mode}`"
  >
    <svg
      class="http-route-map__svg"
      viewBox="0 0 640 420"
      role="img"
      aria-label="Regions and client connecting through a global endpoint"
    >
      <defs>
        <marker
          id="http-route-arrow-blue"
          markerWidth="12"
          markerHeight="12"
          refX="10"
          refY="6"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path d="M 0 0 L 12 6 L 0 12 z" />
        </marker>
        <marker
          id="http-route-arrow-red"
          markerWidth="12"
          markerHeight="12"
          refX="10"
          refY="6"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path d="M 0 0 L 12 6 L 0 12 z" />
        </marker>
      </defs>

      <g class="http-route-map__nodes">
        <g
          v-for="node in orderedNodes"
          :key="node.id"
          class="http-route-node"
          :class="[
            `http-route-node--${node.kind}`,
            node.tone && `http-route-node--${node.tone}`,
          ]"
          :transform="`translate(${node.x} ${node.y})`"
        >
          <rect :width="node.w" :height="node.h" :rx="node.rx" />
          <text :x="node.w / 2" :y="node.h / 2">{{ node.label }}</text>
        </g>
      </g>

      <g class="http-route-map__edges">
        <path
          v-for="edge in edges"
          :key="edge.id"
          class="http-route-edge"
          :class="[
            `http-route-edge--${edge.tone}`,
            edge.dashed && 'http-route-edge--dashed',
          ]"
          :d="edgePath(edge)"
          :marker-end="`url(#http-route-arrow-${edge.tone})`"
        />
      </g>

    </svg>

    <div class="http-route-note">
      <template v-if="props.mode === 'problem'">
        Edge only sees <strong>POST /mcp</strong>; routing fields are buried in JSON.
      </template>
      <template v-else>
        Edge sees <strong>Mcp-Name</strong> and <strong>Mcp-Param-Pipeline</strong>.
      </template>
    </div>
  </section>
</template>

<style scoped>
.http-route-map {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  gap: 0.72rem;
  padding: 0.95rem;
  overflow: hidden;
  background:
    radial-gradient(
      circle at 20% 18%,
      rgba(106, 163, 247, 0.06),
      transparent 28%
    ),
    radial-gradient(
      circle at 80% 28%,
      rgba(240, 107, 90, 0.1),
      transparent 30%
    ),
    rgba(255, 255, 255, 0.74);
}

.http-route-map__svg {
  display: block;
  width: 100%;
  height: 100%;
}

.http-route-node rect {
  fill: rgba(243, 244, 246, 0.44);
  stroke: var(--deck-border-2);
  stroke-width: 1.4;
}

.http-route-node--blue rect {
  stroke: var(--deck-info-line);
}

.http-route-node--red rect {
  stroke: rgba(240, 107, 90, 0.34);
}

.http-route-node--endpoint rect {
  fill: rgba(17, 24, 39, 0.06);
}

.http-route-node--client rect {
  fill: rgba(106, 163, 247, 0.1);
  stroke: rgba(106, 163, 247, 0.72);
  stroke-width: 2.4;
}

.http-route-node text {
  fill: var(--deck-text);
  font-family: var(--deck-font-mono);
  font-size: 28px;
  font-weight: 750;
  text-anchor: middle;
  dominant-baseline: central;
}

.http-route-node--endpoint text {
  fill: var(--deck-muted);
}

.http-route-node--client text {
  fill: var(--deck-info);
}

.http-route-edge {
  fill: none;
  stroke-width: 5;
  stroke-linecap: round;
}

.http-route-map--problem .http-route-edge {
  opacity: 0.36;
  stroke: var(--deck-dim);
}

.http-route-map--problem .http-route-edge--dashed {
  opacity: 0.24;
}

.http-route-edge--blue {
  stroke: var(--deck-info);
}

.http-route-edge--red {
  stroke: var(--deck-no);
}

.http-route-edge--dashed {
  stroke-dasharray: 7 7;
}

#http-route-arrow-blue path {
  fill: var(--deck-info);
}

#http-route-arrow-red path {
  fill: var(--deck-no);
}

.http-route-note {
  box-sizing: border-box;
  width: 100%;
  padding: 0.74rem 0.86rem;
  color: var(--deck-muted);
  border: 1px solid rgba(17, 24, 39, 0.28);
  border-radius: var(--deck-radius);
  background: rgba(243, 244, 246, 0.36);
  font-family: var(--deck-font-mono);
  font-size: 0.88rem;
  line-height: 1.3;
}

.http-route-note strong {
  color: var(--deck-text);
}
</style>
