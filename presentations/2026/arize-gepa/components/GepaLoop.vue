<script setup lang="ts">
const nodes = [
  {
    id: 'task',
    title: 'Task',
    detail: 'Prompt vN',
    x: 92,
    y: 42,
    w: 220,
    h: 116,
  },
  {
    id: 'score',
    title: 'Score',
    detail: 'Output gets scored',
    x: 552,
    y: 42,
    w: 220,
    h: 116,
  },
  {
    id: 'reflect',
    title: 'Reflect',
    detail: 'Generate new prompt',
    x: 322,
    y: 238,
    w: 220,
    h: 116,
  },
] as const

const edges = [
  {
    from: 'task',
    to: 'score',
    label: 'run task → output',
    d: 'M 320 100 C 390 58, 474 58, 544 100',
    labelX: 432,
    labelY: 54,
  },
  {
    from: 'score',
    to: 'reflect',
    label: 'feedback signal',
    d: 'M 625 163 C 650 224, 598 286, 550 296',
    labelX: 640,
    labelY: 230,
  },
  {
    from: 'reflect',
    to: 'task',
    label: 'new prompt',
    d: 'M 314 296 C 266 286, 214 224, 239 163',
    labelX: 224,
    labelY: 230,
  },
] as const
</script>

<template>
  <figure class="gepa-loop" aria-label="GEPA loop: Task flows to Score, Score flows to Reflect, Reflect generates a new prompt for Task.">
    <svg viewBox="0 0 864 424" role="img">
      <defs>
        <marker
          id="gepa-arrow"
          markerWidth="9"
          markerHeight="9"
          refX="7.5"
          refY="4.5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M 1.5 1.5 L 8 4.5 L 1.5 7.5 z" />
        </marker>
      </defs>

      <g class="gepa-loop__edges">
        <g v-for="edge in edges" :key="`${edge.from}-${edge.to}`" class="gepa-loop__edge">
          <path :d="edge.d" />
          <text :x="edge.labelX" :y="edge.labelY">{{ edge.label }}</text>
        </g>
      </g>

      <g class="gepa-loop__nodes">
        <g
          v-for="node in nodes"
          :key="node.id"
          class="gepa-loop__node"
          :class="`gepa-loop__node--${node.id}`"
          :transform="`translate(${node.x} ${node.y})`"
        >
          <rect :width="node.w" :height="node.h" rx="24" />
          <text class="gepa-loop__title" :x="node.w / 2" y="52">{{ node.title }}</text>
          <text class="gepa-loop__detail" :x="node.w / 2" y="82">{{ node.detail }}</text>
        </g>
      </g>
    </svg>
  </figure>
</template>
