<script setup lang="ts">
// Sankey of fast-agent chess-best-move trials: image read -> how the move was
// found -> verifier reward. Counts come from data/chess-flows.json
// (analysis/chess-intro/flows.py). Ribbon colour is the "how" column; outcome
// is carried by position and by a lighter ribbon for reward 0.
import { computed } from "vue";
import flows from "../data/chess-flows.json";

const W = 868;
const H = 322;
const NODE = 12;
const GAP = 28;
const X = [190, 452, 700];

const cols = [
  [
    { id: "image", label: "Reads the image" },
    { id: "noimage", label: "No image read" },
  ],
  [
    { id: "stockfish", label: "Runs Stockfish" },
    { id: "pychess", label: "python-chess only" },
    { id: "none", label: "No chess library" },
  ],
  [
    { id: "pass", label: "Reward 1" },
    { id: "fail", label: "Reward 0" },
  ],
];
const count = (a: string, b: string) => flows.links.find((l) => l.path.length === 2 && l.path[0] === a && l.path[1] === b)?.n ?? 0;
const TOP = 22;
const BOTTOM = 26;
const scale = (H - TOP - BOTTOM - GAP * 2) / flows.n;

const layout = computed(() => {
  const nodes: Record<string, { x: number; y: number; h: number; n: number; label: string; col: number; out: number; in: number }> = {};
  cols.forEach((col, c) => {
    const sizes = col.map((nd) => (c === 0 ? cols[1].reduce((s, t) => s + count(nd.id, t.id), 0) : cols[c - 1].reduce((s, t) => s + count(t.id, nd.id), 0)));
    const total = sizes.reduce((s, n) => s + n * scale, 0) + GAP * (col.length - 1);
    let y = TOP + (H - TOP - BOTTOM - total) / 2;
    col.forEach((nd, i) => {
      nodes[nd.id] = { x: X[c], y, h: sizes[i] * scale, n: sizes[i], label: nd.label, col: c, out: 0, in: 0 };
      y += sizes[i] * scale + GAP;
    });
  });
  const ribbons: { d: string; cls: string; tip: string }[] = [];
  for (let c = 0; c < 2; c++)
    for (const a of cols[c])
      for (const b of cols[c + 1]) {
        const n = count(a.id, b.id);
        if (!n) continue;
        const s = nodes[a.id];
        const t = nodes[b.id];
        const h = n * scale;
        const y0 = s.y + s.out;
        const y1 = t.y + t.in;
        s.out += h;
        t.in += h;
        const x0 = s.x + NODE;
        const x1 = t.x;
        const m = (x0 + x1) / 2;
        const engine = c === 0 ? b.id : a.id;
        ribbons.push({
          d: `M${x0},${y0}C${m},${y0} ${m},${y1} ${x1},${y1}L${x1},${y1 + h}C${m},${y1 + h} ${m},${y0 + h} ${x0},${y0 + h}Z`,
          cls: `is-${engine}${b.id === "fail" ? " is-fail" : ""}`,
          tip: `${a.label} → ${b.label}: ${n} trials`,
        });
      }
  return { nodes, ribbons };
});
const node = (id: string) => layout.value.nodes[id];
const won = (id: string) => count(id, "pass");
</script>

<template>
  <svg class="fa-sankey" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="How fast-agent trials approached chess-best-move">
    <path v-for="r in layout.ribbons" :key="r.tip" :d="r.d" class="fa-sankey-ribbon" :class="r.cls"><title>{{ r.tip }}</title></path>
    <rect v-for="(n, id) in layout.nodes" :key="id" :x="n.x" :y="n.y" :width="NODE" :height="Math.max(2, n.h)" class="fa-sankey-node" />

    <g v-for="c in cols[0]" :key="c.id">
      <text class="fa-sankey-label" :x="node(c.id).x - 10" :y="node(c.id).y + node(c.id).h / 2 - 2" text-anchor="end">{{ c.label }}</text>
      <text class="fa-sankey-sub" :x="node(c.id).x - 10" :y="node(c.id).y + node(c.id).h / 2 + 16" text-anchor="end">{{ node(c.id).n }} trials</text>
    </g>
    <text class="fa-sankey-sub" :x="node('noimage').x - 10" :y="node('noimage').y + node('noimage').h / 2 + 33" text-anchor="end">{{ flows.asciiNoImage }} print the board as ASCII</text>

    <g v-for="c in cols[1]" :key="c.id">
      <text class="fa-sankey-label" :x="node(c.id).x + NODE / 2" :y="node(c.id).y - 9" text-anchor="middle">
        {{ c.label }}
        <tspan class="fa-sankey-sub" dx="6">{{ won(c.id) }} of {{ node(c.id).n }} rewarded</tspan>
      </text>
    </g>

    <g v-for="c in cols[2]" :key="c.id">
      <text class="fa-sankey-label" :x="node(c.id).x + NODE + 10" :y="node(c.id).y + node(c.id).h / 2 - 2">{{ c.label }}</text>
      <text class="fa-sankey-sub" :x="node(c.id).x + NODE + 10" :y="node(c.id).y + node(c.id).h / 2 + 16">{{ node(c.id).n }} trials</text>
    </g>
  </svg>
</template>

<style scoped>
.fa-sankey {
  display: block;
  width: 100%;
  height: auto;
  font-family: var(--font-read);
}
.fa-sankey-label {
  font-size: 16px;
  font-weight: 700;
  fill: var(--petrol);
  /* labels in the middle column sit over ribbons: an ivory halo keeps them readable */
  paint-order: stroke;
  stroke: var(--ivory);
  stroke-width: 5px;
  stroke-linejoin: round;
}
.fa-sankey-sub {
  font-size: 13px;
  font-weight: 500;
  fill: var(--text-muted);
  font-variant-numeric: tabular-nums;
  paint-order: stroke;
  stroke: var(--ivory);
  stroke-width: 5px;
  stroke-linejoin: round;
}
.fa-sankey-node { fill: var(--petrol); }
.fa-sankey-ribbon { opacity: 0.62; }
.fa-sankey-ribbon.is-fail { opacity: 0.26; }
.fa-sankey-ribbon:hover { opacity: 0.85; }
.is-stockfish { fill: var(--teal); }
.is-pychess { fill: var(--petrol); }
.is-none { fill: var(--orange); }
</style>
