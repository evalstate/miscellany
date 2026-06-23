<script setup lang="ts">
const components = [
  {
    id: 'theme',
    title: 'Required items',
    weight: '45%',
    detail: 'Maya · lighthouse · lamp',
    x: 42,
    y: 42,
    w: 310,
    h: 110,
  },
  {
    id: 'story',
    title: 'Story length',
    weight: '25%',
    detail: '≤ 140 words',
    x: 42,
    y: 180,
    w: 310,
    h: 110,
  },
  {
    id: 'prompt',
    title: 'Prompt length',
    weight: '30%',
    detail: 'short prompts win',
    x: 42,
    y: 318,
    w: 310,
    h: 110,
  },
] as const

const edges = components.map((node) => ({
  id: node.id,
  d: `M ${node.x + node.w + 18} ${node.y + node.h / 2} C 430 ${node.y + node.h / 2}, 524 235, 592 235`,
}))
</script>

<template>
  <figure class="story-score-breakdown" aria-label="Story score components from the latest scorer feed into one GEPA score.">
    <svg viewBox="0 0 920 470" role="img">
      <defs>
        <marker
          id="story-score-arrow"
          markerWidth="10"
          markerHeight="10"
          refX="8.5"
          refY="5"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M 1.5 1.5 L 9 5 L 1.5 8.5 z" />
        </marker>
      </defs>

      <g class="story-score-breakdown__edges">
        <path v-for="edge in edges" :key="edge.id" :d="edge.d" />
      </g>

      <g class="story-score-breakdown__components">
        <g
          v-for="component in components"
          :key="component.id"
          class="story-score-breakdown__component"
          :class="`story-score-breakdown__component--${component.id}`"
          :transform="`translate(${component.x} ${component.y})`"
        >
          <rect :width="component.w" :height="component.h" rx="22" />
          <text class="story-score-breakdown__weight" x="24" y="43">{{ component.weight }}</text>
          <text class="story-score-breakdown__title" x="24" y="75">{{ component.title }}</text>
          <text class="story-score-breakdown__detail" x="24" y="96">{{ component.detail }}</text>
        </g>
      </g>

      <g class="story-score-breakdown__score" transform="translate(604 92)">
        <rect width="274" height="286" rx="32" />
        <text class="story-score-breakdown__score-label" x="137" y="64">GEPA score</text>
        <text class="story-score-breakdown__score-value" x="137" y="153">0.0–1.0</text>
        <line x1="44" y1="184" x2="230" y2="184" />
        <text class="story-score-breakdown__formula" x="137" y="219">weighted sum</text>
        <text class="story-score-breakdown__formula story-score-breakdown__formula--muted" x="137" y="249">45% + 25% + 30%</text>
      </g>

    </svg>
  </figure>
</template>

<style scoped>
.story-score-breakdown {
  width: min(52.5rem, 100%);
  margin: 0.25rem auto 0;
}

.story-score-breakdown svg {
  width: 100%;
  height: auto;
  overflow: visible;
}

.story-score-breakdown__edges path {
  fill: none;
  stroke: rgba(142, 232, 255, 0.64);
  stroke-width: 3.5;
  stroke-linecap: round;
  marker-end: url("#story-score-arrow");
}

.story-score-breakdown marker path {
  fill: rgba(142, 232, 255, 0.9);
}

.story-score-breakdown__component rect,
.story-score-breakdown__score rect {
  fill: rgba(255, 255, 255, 0.08);
  stroke: rgba(246, 248, 255, 0.2);
  stroke-width: 2;
  filter: drop-shadow(0 22px 48px rgba(0, 0, 0, 0.22));
}

.story-score-breakdown__component--prompt rect {
  stroke: rgba(142, 232, 255, 0.74);
}

.story-score-breakdown__component--story rect {
  stroke: rgba(196, 181, 253, 0.78);
}

.story-score-breakdown__component--theme rect {
  stroke: rgba(255, 184, 107, 0.74);
}

.story-score-breakdown__weight {
  fill: #8ee8ff;
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -0.04em;
}

.story-score-breakdown__title {
  fill: #f6f8ff;
  font-size: 25px;
  font-weight: 850;
  letter-spacing: -0.035em;
}

.story-score-breakdown__detail {
  fill: rgba(246, 248, 255, 0.76);
  font-size: 16.5px;
  font-weight: 820;
}

.story-score-breakdown__score rect {
  fill: rgba(142, 232, 255, 0.1);
  stroke: rgba(142, 232, 255, 0.72);
}

.story-score-breakdown__score-label {
  fill: rgba(246, 248, 255, 0.7);
  font-size: 20px;
  font-weight: 830;
  letter-spacing: 0.08em;
  text-anchor: middle;
  text-transform: uppercase;
}

.story-score-breakdown__score-value {
  fill: #ffb86b;
  font-size: 58px;
  font-weight: 860;
  letter-spacing: 0;
  text-anchor: middle;
}

.story-score-breakdown__score line {
  stroke: rgba(246, 248, 255, 0.2);
  stroke-width: 2;
}

.story-score-breakdown__formula {
  fill: #f6f8ff;
  font-size: 19px;
  font-weight: 820;
  letter-spacing: 0.04em;
  text-anchor: middle;
  text-transform: uppercase;
}

.story-score-breakdown__formula--muted {
  fill: rgba(246, 248, 255, 0.68);
  font-size: 16px;
}

</style>
