<script setup lang="ts">
import { computed } from "vue";
import activityData from "../data-viz/mcp_weekly_init_tool_calls.json";

type Row = {
  week_start: string;
  week_end: string;
  iso_week: string;
  init_requests: number;
  tool_calls: number;
  snapshot_count: number;
  partial_week: boolean;
};

const rows = (activityData.rows as Row[]).toSorted((a, b) =>
  a.week_start.localeCompare(b.week_start),
);

const width = 1000;
const height = 520;
const plot = { left: 86, right: 96, top: 66, bottom: 84 };
const plotWidth = width - plot.left - plot.right;
const plotHeight = height - plot.top - plot.bottom;
const barWidth = Math.max(
  3,
  Math.min(9, (plotWidth / Math.max(1, rows.length)) * 0.34),
);

function toDate(value: string) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day));
}

function niceMax(value: number) {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const scaled = value / power;
  const nice = scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return nice * power;
}

const startDate = toDate(rows[0].week_start);
const endDate = toDate(rows.at(-1)!.week_start);
const startMs = startDate.getTime();
const spanMs = Math.max(1, endDate.getTime() - startMs);
const initMax = niceMax(Math.max(...rows.map((row) => row.init_requests)));
const toolMax = niceMax(Math.max(...rows.map((row) => row.tool_calls)));

function xForDate(value: string) {
  return plot.left + ((toDate(value).getTime() - startMs) / spanMs) * plotWidth;
}

function yForInit(value: number) {
  return plot.top + (1 - value / initMax) * plotHeight;
}

function yForTool(value: number) {
  return plot.top + (1 - value / toolMax) * plotHeight;
}

function formatCount(value: number) {
  if (value >= 1_000_000)
    return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
  if (value >= 1_000)
    return `${(value / 1_000).toFixed(value >= 100_000 ? 0 : 1)}k`;
  return `${value}`;
}

const initTicks = computed(() => [
  0,
  initMax * 0.25,
  initMax * 0.5,
  initMax * 0.75,
  initMax,
]);
const toolTicks = computed(() => [0, toolMax * 0.5, toolMax]);

const toolLine = computed(() =>
  rows
    .map(
      (row) =>
        `${xForDate(row.week_start).toFixed(1)},${yForTool(row.tool_calls).toFixed(1)}`,
    )
    .join(" "),
);

const monthTicks = computed(() => {
  const ticks: { label: string; x: number }[] = [];
  const cursor = new Date(
    Date.UTC(startDate.getUTCFullYear(), startDate.getUTCMonth(), 1),
  );
  const formatter = new Intl.DateTimeFormat("en", { month: "short" });
  let index = 0;
  while (cursor.getTime() <= endDate.getTime()) {
    const x = plot.left + ((cursor.getTime() - startMs) / spanMs) * plotWidth;
    if (index % 2 === 0 && x >= plot.left && x <= plot.left + plotWidth) {
      ticks.push({ label: formatter.format(cursor), x });
    }
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
    index += 1;
  }
  return ticks;
});

const latest = rows.at(-1)!;
const first = rows[0];
</script>

<template>
  <section class="activity-chart">
    <header class="activity-chart__header">
      <div>
        <h1>Weekly MCP activity</h1>
        <p>Initialization requests as bars · tool calls as line</p>
      </div>
      <div class="activity-chart__stat">
        <span>latest week</span>
        <strong>{{ formatCount(latest.init_requests) }}</strong>
        <em>initializations</em>
      </div>
    </header>

    <svg
      class="activity-chart__svg"
      :viewBox="`0 0 ${width} ${height}`"
      role="img"
    >
      <defs>
        <linearGradient id="activity-bar-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#ffc649" />
          <stop offset="100%" stop-color="rgba(245, 164, 0, 0.38)" />
        </linearGradient>
        <linearGradient id="activity-tool-line" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#8bb8ff" />
          <stop offset="100%" stop-color="#6aa3f7" />
        </linearGradient>
        <filter
          id="activity-line-glow"
          x="-40%"
          y="-40%"
          width="180%"
          height="180%"
        >
          <feGaussianBlur stdDeviation="3.2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect
        :x="plot.left"
        :y="plot.top"
        :width="plotWidth"
        :height="plotHeight"
        class="activity-chart__plot-bg"
      />

      <g class="activity-chart__grid">
        <line
          v-for="tick in initTicks"
          :key="`init-grid-${tick}`"
          :x1="plot.left"
          :x2="plot.left + plotWidth"
          :y1="yForInit(tick)"
          :y2="yForInit(tick)"
        />
        <line
          v-for="tick in monthTicks"
          :key="`month-grid-${tick.label}-${tick.x}`"
          :x1="tick.x"
          :x2="tick.x"
          :y1="plot.top"
          :y2="plot.top + plotHeight"
        />
      </g>

      <g class="activity-chart__bars">
        <rect
          v-for="row in rows"
          :key="row.week_start"
          :x="xForDate(row.week_start) - barWidth / 2"
          :y="yForInit(row.init_requests)"
          :width="barWidth"
          :height="plot.top + plotHeight - yForInit(row.init_requests)"
          rx="2"
          :class="{ 'is-partial': row.partial_week }"
        />
      </g>

      <polyline
        class="activity-chart__line"
        :points="toolLine"
        filter="url(#activity-line-glow)"
      />

      <g class="activity-chart__axis activity-chart__axis--left">
        <text :x="plot.left" :y="plot.top - 20">Initializations</text>
        <text
          v-for="tick in initTicks"
          :key="`init-label-${tick}`"
          :x="plot.left - 14"
          :y="yForInit(tick) + 4"
          text-anchor="end"
        >
          {{ formatCount(tick) }}
        </text>
      </g>

      <g class="activity-chart__axis activity-chart__axis--right">
        <text :x="plot.left + plotWidth" :y="plot.top - 20" text-anchor="end">
          Tool calls
        </text>
        <text
          v-for="tick in toolTicks"
          :key="`tool-label-${tick}`"
          :x="plot.left + plotWidth + 14"
          :y="yForTool(tick) + 4"
        >
          {{ formatCount(tick) }}
        </text>
      </g>

      <g class="activity-chart__x-axis">
        <text
          v-for="tick in monthTicks"
          :key="`month-label-${tick.label}-${tick.x}`"
          :x="tick.x"
          :y="plot.top + plotHeight + 42"
          text-anchor="middle"
        >
          {{ tick.label }}
        </text>
        <text
          :x="plot.left"
          :y="plot.top + plotHeight + 68"
          text-anchor="start"
        >
          {{ first.week_start }}
        </text>
        <text
          :x="plot.left + plotWidth"
          :y="plot.top + plotHeight + 68"
          text-anchor="end"
        >
          {{ latest.week_end }}
        </text>
      </g>

      <g class="activity-chart__legend">
        <rect
          :x="plot.left + plotWidth - 292"
          :y="plot.top + 14"
          width="12"
          height="38"
          rx="2"
        />
        <text :x="plot.left + plotWidth - 270" :y="plot.top + 38">
          initializations
        </text>
        <line
          :x1="plot.left + plotWidth - 132"
          :x2="plot.left + plotWidth - 96"
          :y1="plot.top + 33"
          :y2="plot.top + 33"
        />
        <text :x="plot.left + plotWidth - 86" :y="plot.top + 38">
          tool calls
        </text>
      </g>
    </svg>
  </section>
</template>

<style scoped>
.activity-chart {
  height: 100%;
  padding: 1.75rem 2rem 1.35rem;
  background:
    radial-gradient(
      circle at 78% 12%,
      rgba(106, 163, 247, 0.14),
      transparent 28%
    ),
    rgba(20, 22, 27, 0.74);
  border: 1px solid var(--deck-border);
  border-radius: var(--deck-radius);
  box-shadow: var(--deck-shadow);
}

.activity-chart__header {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 1.5rem;
  margin-bottom: 0.25rem;
}

.activity-chart h1 {
  margin: 0;
  font-size: 2.55rem;
}

.activity-chart p {
  margin: 0.45rem 0 0;
  color: var(--deck-muted);
  font-size: 0.92rem;
}

.activity-chart__stat {
  min-width: 170px;
  padding: 0.7rem 0.85rem;
  text-align: right;
  background: var(--deck-accent-bg);
  border: 1px solid var(--deck-accent-line);
  border-radius: var(--deck-radius-sm);
}

.activity-chart__stat span,
.activity-chart__stat em {
  display: block;
  color: var(--deck-dim);
  font-size: 0.58rem;
  font-style: normal;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.activity-chart__stat strong {
  display: block;
  color: var(--deck-accent-hi);
  font-size: 1.65rem;
  line-height: 1.1;
}

.activity-chart__svg {
  width: 100%;
  height: calc(100% - 82px);
  overflow: visible;
}

.activity-chart__plot-bg {
  fill: rgba(11, 12, 15, 0.34);
  stroke: var(--deck-border);
}

.activity-chart__grid line {
  stroke: rgba(240, 236, 226, 0.08);
  stroke-width: 1;
}

.activity-chart__bars rect {
  fill: url(#activity-bar-fill);
  opacity: 0.82;
}

.activity-chart__bars rect.is-partial {
  opacity: 0.5;
}

.activity-chart__line {
  fill: none;
  stroke: url(#activity-tool-line);
  stroke-width: 4;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.activity-chart__axis text,
.activity-chart__x-axis text,
.activity-chart__legend text {
  fill: var(--deck-dim);
  font-family: var(--deck-font-mono);
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.activity-chart__axis--left text:first-child {
  fill: var(--deck-accent-hi);
}

.activity-chart__axis--right text:first-child {
  fill: var(--deck-info);
}

.activity-chart__x-axis text {
  font-size: 13px;
}

.activity-chart__legend rect {
  fill: url(#activity-bar-fill);
}

.activity-chart__legend line {
  stroke: var(--deck-info);
  stroke-width: 4;
  stroke-linecap: round;
}
</style>
