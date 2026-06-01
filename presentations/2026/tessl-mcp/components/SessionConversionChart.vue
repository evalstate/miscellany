<script setup lang="ts">
import { computed } from "vue";
import conversionData from "../data-viz/session_conversion_daily.json";

type Row = {
  day: string;
  sessions: number;
  converted_sessions: number;
  conversion_rate_pct: number;
  converted_sessions_3d_avg: number;
};

const rows = (conversionData.rows as Row[]).toSorted((a, b) =>
  a.day.localeCompare(b.day),
);

const width = 1000;
const height = 520;
const plot = { left: 82, right: 104, top: 62, bottom: 84 };
const plotWidth = width - plot.left - plot.right;
const plotHeight = height - plot.top - plot.bottom;

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

const startDate = toDate(rows[0].day);
const endDate = toDate(rows.at(-1)!.day);
const startMs = startDate.getTime();
const spanMs = Math.max(1, endDate.getTime() - startMs);
const rateMax = Math.max(
  4,
  Math.ceil(Math.max(...rows.map((row) => row.conversion_rate_pct))),
);
const convertedMax = niceMax(
  Math.max(...rows.map((row) => row.converted_sessions_3d_avg)),
);

function xForDay(value: string) {
  return plot.left + ((toDate(value).getTime() - startMs) / spanMs) * plotWidth;
}

function yForRate(value: number) {
  return plot.top + (1 - value / rateMax) * plotHeight;
}

function yForConverted(value: number) {
  return plot.top + (1 - value / convertedMax) * plotHeight;
}

function formatCount(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000)
    return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}k`;
  return `${Math.round(value)}`;
}

const rateTicks = computed(() =>
  Array.from({ length: rateMax + 1 }, (_, index) => index),
);
const convertedTicks = computed(() => [0, convertedMax * 0.5, convertedMax]);

const rateLine = computed(() =>
  rows
    .map(
      (row) =>
        `${xForDay(row.day).toFixed(1)},${yForRate(row.conversion_rate_pct).toFixed(1)}`,
    )
    .join(" "),
);

const convertedAverageLine = computed(() =>
  rows
    .map(
      (row) =>
        `${xForDay(row.day).toFixed(1)},${yForConverted(row.converted_sessions_3d_avg).toFixed(1)}`,
    )
    .join(" "),
);

const monthTicks = computed(() => {
  const ticks: { label: string; x: number }[] = [];
  const cursor = new Date(
    Date.UTC(startDate.getUTCFullYear(), startDate.getUTCMonth(), 1),
  );
  const formatter = new Intl.DateTimeFormat("en", { month: "short" });
  while (cursor.getTime() <= endDate.getTime()) {
    const x = plot.left + ((cursor.getTime() - startMs) / spanMs) * plotWidth;
    if (x >= plot.left && x <= plot.left + plotWidth)
      ticks.push({ label: formatter.format(cursor), x });
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
  }
  return ticks;
});

const first = rows[0];
const latest = rows.at(-1)!;
const peak = rows.reduce(
  (best, row) =>
    row.conversion_rate_pct > best.conversion_rate_pct ? row : best,
  first,
);
const overallRate =
  (rows.reduce((sum, row) => sum + row.converted_sessions, 0) /
    rows.reduce((sum, row) => sum + row.sessions, 0)) *
  100;
</script>

<template>
  <section class="conversion-chart">
    <div class="conversion-chart__stats">
      <div>
        <span>overall</span>
        <strong>{{ overallRate.toFixed(2) }}%</strong>
      </div>
      <div>
        <span>latest</span>
        <strong>{{ latest.conversion_rate_pct.toFixed(2) }}%</strong>
      </div>
    </div>

    <svg
      class="conversion-chart__svg"
      :viewBox="`0 0 ${width} ${height}`"
      role="img"
    >
      <defs>
        <linearGradient id="conversion-rate-stroke" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#ffc649" />
          <stop offset="100%" stop-color="#f5a400" />
        </linearGradient>
        <linearGradient
          id="conversion-volume-stroke"
          x1="0"
          x2="1"
          y1="0"
          y2="0"
        >
          <stop offset="0%" stop-color="#8bb8ff" />
          <stop offset="100%" stop-color="#6aa3f7" />
        </linearGradient>
        <filter
          id="conversion-line-glow"
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
        class="conversion-chart__plot-bg"
      />

      <g class="conversion-chart__grid">
        <line
          v-for="tick in rateTicks"
          :key="`rate-grid-${tick}`"
          :x1="plot.left"
          :x2="plot.left + plotWidth"
          :y1="yForRate(tick)"
          :y2="yForRate(tick)"
        />
        <line
          v-for="tick in monthTicks"
          :key="`month-grid-${tick.label}`"
          :x1="tick.x"
          :x2="tick.x"
          :y1="plot.top"
          :y2="plot.top + plotHeight"
        />
      </g>

      <polyline
        class="conversion-chart__volume-line"
        :points="convertedAverageLine"
      />
      <polyline
        class="conversion-chart__rate-line"
        :points="rateLine"
        filter="url(#conversion-line-glow)"
      />

      <g class="conversion-chart__dots">
        <circle
          v-for="row in rows"
          :key="row.day"
          :cx="xForDay(row.day)"
          :cy="yForRate(row.conversion_rate_pct)"
          r="2.8"
        />
      </g>

      <g class="conversion-chart__axis conversion-chart__axis--left">
        <text :x="plot.left" :y="plot.top - 20">Conversion rate</text>
        <text
          v-for="tick in rateTicks"
          :key="`rate-label-${tick}`"
          :x="plot.left - 14"
          :y="yForRate(tick) + 4"
          text-anchor="end"
        >
          {{ tick }}%
        </text>
      </g>

      <g class="conversion-chart__axis conversion-chart__axis--right">
        <text :x="plot.left + plotWidth" :y="plot.top - 20" text-anchor="end">
          3-day converted avg
        </text>
        <text
          v-for="tick in convertedTicks"
          :key="`converted-label-${tick}`"
          :x="plot.left + plotWidth + 14"
          :y="yForConverted(tick) + 4"
        >
          {{ formatCount(tick) }}
        </text>
      </g>

      <g class="conversion-chart__x-axis">
        <text
          v-for="tick in monthTicks"
          :key="`month-label-${tick.label}`"
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
          {{ first.day }}
        </text>
        <text
          :x="plot.left + plotWidth"
          :y="plot.top + plotHeight + 68"
          text-anchor="end"
        >
          {{ latest.day }}
        </text>
      </g>

      <g class="conversion-chart__legend">
        <line
          :x1="plot.left + plotWidth - 352"
          :x2="plot.left + plotWidth - 316"
          :y1="plot.top + 32"
          :y2="plot.top + 32"
          class="is-rate"
        />
        <text :x="plot.left + plotWidth - 306" :y="plot.top + 37">
          conversion rate
        </text>
        <line
          :x1="plot.left + plotWidth - 152"
          :x2="plot.left + plotWidth - 116"
          :y1="plot.top + 32"
          :y2="plot.top + 32"
          class="is-volume"
        />
        <text :x="plot.left + plotWidth - 106" :y="plot.top + 37">
          3-day avg
        </text>
      </g>

      <g class="conversion-chart__peak">
        <circle
          :cx="xForDay(peak.day)"
          :cy="yForRate(peak.conversion_rate_pct)"
          r="6"
        />
        <text
          :x="xForDay(peak.day) - 12"
          :y="yForRate(peak.conversion_rate_pct) - 16"
          text-anchor="end"
        >
          peak {{ peak.conversion_rate_pct.toFixed(2) }}%
        </text>
      </g>
    </svg>
  </section>
</template>

<style scoped>
.conversion-chart {
  --deck-chart-glow-x: 76%;
  --deck-chart-glow-y: 14%;
  --deck-chart-header-gap: 1.4rem;
  --deck-chart-subtitle-size: 0.9rem;
  --deck-chart-stat-size: 1.5rem;
}

.conversion-chart__stats div {
  padding: 0.68rem 0.8rem;
}

.conversion-chart__rate-line,
.conversion-chart__volume-line {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.conversion-chart__rate-line {
  stroke: url(#conversion-rate-stroke);
  stroke-width: 4;
}

.conversion-chart__volume-line {
  stroke: url(#conversion-volume-stroke);
  stroke-width: 3.5;
  opacity: 0.9;
}

.conversion-chart__dots circle {
  fill: var(--deck-accent-hi);
  stroke: var(--deck-bg);
  stroke-width: 1.5;
}

.conversion-chart__legend line {
  stroke-width: 4;
  stroke-linecap: round;
}

.conversion-chart__legend .is-rate {
  stroke: var(--deck-accent-hi);
}

.conversion-chart__legend .is-volume {
  stroke: var(--deck-info);
}

.conversion-chart__peak circle {
  fill: var(--deck-bg);
  stroke: var(--deck-accent-hi);
  stroke-width: 3;
}

.conversion-chart__peak text {
  fill: var(--deck-accent-hi);
}
</style>
