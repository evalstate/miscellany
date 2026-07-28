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

const width = 1200;
const height = 560;
const plot = { left: 92, right: 74, top: 42, bottom: 62 };
const plotWidth = width - plot.left - plot.right;
const plotHeight = height - plot.top - plot.bottom;
const barWidth = Math.max(
  4,
  Math.min(11, (plotWidth / Math.max(1, rows.length)) * 0.34),
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

</script>

<template>
  <section class="activity-chart">
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
          <feGaussianBlur stdDeviation="1.8" result="blur" />
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
        <text :x="plot.left" :y="plot.top - 18">Initializations</text>
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

      <text
        class="activity-chart__line-label"
        :x="plot.left + plotWidth"
        :y="plot.top - 18"
        text-anchor="end"
      >
        Tool Calls
      </text>

      <g class="activity-chart__x-axis">
        <text
          v-for="tick in monthTicks"
          :key="`month-label-${tick.label}-${tick.x}`"
          :x="tick.x"
          :y="plot.top + plotHeight + 38"
          text-anchor="middle"
        >
          {{ tick.label }}
        </text>
      </g>
    </svg>
  </section>
</template>

<style scoped>
.activity-chart {
  --activity-chart-bar-fill: url(#activity-bar-fill);
  --activity-chart-line-stroke: url(#activity-tool-line);
}
</style>
