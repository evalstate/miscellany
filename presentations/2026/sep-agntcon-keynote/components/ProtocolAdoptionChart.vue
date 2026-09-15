<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId, watch } from "vue";
import { useNav, useSlideContext } from "@slidev/client";
import { useRoute } from 'vue-router';
import data from "../data/protocol-adoption/data.json";
import dataUrl from "../data/protocol-adoption/data.json?url";
import csvUrl from "../data/protocol-adoption/daily.csv?url";
import provenanceUrl from "../data/protocol-adoption/provenance.json?url";
import readmeUrl from "../data/protocol-adoption/README.md?url";

const props = withDefaults(
  defineProps<{
    capture?: boolean;
    title?: string;
    subtitle?: string;
    footer?: string;
    footnote?: string;
  }>(),
  {
    capture: false,
    title: "Protocol Adoption",
    subtitle: "2026-07-28 · share of tool calls",
    footer: "",
    footnote:
      "Recognized protocol versions only. Five hashes excluded. Lines illustrate daily observations, not intraday estimates.",
  },
);
const durationMs = 30000;
const lastIndex = data.dates.length - 1;
const position = ref(lastIndex);
const playing = ref(false);
const client = ref(data.series[0].id);
const viewport = ref<"full" | "seven">("full");
const speed = ref(4);
const reducedMotion = ref(false);
const browserPrint = ref(false);
const detailsOpen = ref(false);
const { $page, $nav, $renderContext } = useSlideContext();
const { isPrintMode } = useNav();
const deckRoute = useRoute();
const printMode = computed(
  () =>
    browserPrint.value || deckRoute.path === '/print' || isPrintMode.value || $renderContext.value === "print",
);
const active = computed(() => $nav.value.currentSlideNo === $page.value);
const renderedPosition = computed(() =>
  printMode.value ? lastIndex : position.value,
);
const index = computed(() => Math.floor(renderedPosition.value));
const selected = computed(() =>
  data.series.find((s) => s.id === client.value)!,
);
const row = computed(() => selected.value.rows[index.value]);
const bounds = computed(() =>
  viewport.value === "seven"
    ? [
        Math.max(0, renderedPosition.value - 6),
        Math.max(6, renderedPosition.value),
      ]
    : [0, lastIndex],
);
const state = computed(() =>
  Object.freeze({
    index: index.value,
    position: renderedPosition.value,
    viewportStart: bounds.value[0],
    viewportEnd: bounds.value[1],
    duration: durationMs,
    durationMs,
    playing: playing.value,
    client: client.value,
    viewport: viewport.value,
    speed: speed.value,
    reducedMotion: reducedMotion.value,
    date: row.value.date,
    partial: row.value.partial,
    dailyShare: row.value.daily_share,
    rollingShare: row.value.rolling_share,
    counts: Object.freeze({ ...row.value.counts }),
  }),
);
const uid = `protocol-adoption-${useId().replace(/[^a-zA-Z0-9_-]/g, "-")}`;
const purple = "#6430d8",
  slate = "#475569",
  amber = "#a56a00";
const pct = (v: number | null) =>
  v === null ? "—" : (v * 100).toFixed(1) + "%";
const y = (v: number) => 825 - v * 620;
const x = (i: number) =>
  80 +
  ((i - bounds.value[0]) / Math.max(1, bounds.value[1] - bounds.value[0])) *
    1420;
const ticks = computed(() => {
  const [lo, hi] = bounds.value;
  const stride =
    viewport.value === "seven" ? 1 : Math.max(1, Math.ceil((hi - lo) / 6));
  return data.dates.flatMap((date, i) =>
    i >= lo &&
    i <= hi &&
    (viewport.value === "seven" || i % stride === 0 || i === lastIndex)
      ? [{ i, label: date.slice(5), x: x(i) }]
      : [],
  );
});
// Purple rolling7 leads; muted slate daily follows, with amber partial observations.
// Preserve source drawing order: rolling first, then daily; line before its point.
// Supplied rates/denominators are never recomputed. Only guide geometry interpolates.
const seriesMarks = computed(() =>
  (["rolling_share", "daily_share"] as const).map((field) => ({
    field,
    marks: selected.value.rows.flatMap((r, i) => {
      const v = r[field];
      if (v === null) return [];
      const previous = selected.value.rows[i - 1];
      const partial = field === "daily_share" && r.partial;
      return [
        {
          i,
          v,
          date: r.date,
          partial,
          color: partial ? amber : field === "daily_share" ? slate : purple,
          previous: previous?.[field],
          connected:
            !!previous &&
            previous[field] !== null &&
            Date.parse(r.date) - Date.parse(previous.date) === 86400000,
        },
      ];
    }),
  })),
);
const plotted = computed(() =>
  seriesMarks.value.map((series) => ({
    field: series.field,
    marks: series.marks.map((mark) => {
      const [lo, hi] = bounds.value;
      const start = Math.max(mark.i - 1, lo),
        end = Math.min(mark.i, renderedPosition.value);
      const value = (t: number) =>
        mark.previous! + (mark.v - mark.previous!) * (t - (mark.i - 1));
      return {
        ...mark,
        dot: mark.i <= index.value && mark.i >= lo && mark.i <= hi,
        cx: x(mark.i),
        cy: y(mark.v),
        line:
          mark.connected && end > start && mark.i - 1 < hi
            ? {
                x1: x(start),
                y1: y(value(start)),
                x2: x(end),
                y2: y(value(end)),
              }
            : null,
      };
    }),
  })),
);
// Endpoint adornments are derived only from the reached, complete frame. Null
// remains unavailable; no look-ahead or synthetic zero is used for either rate.
const endpoints = computed(() =>
  renderedPosition.value === lastIndex
    ? plotted.value.flatMap(series => {
        const mark = series.marks.find(m => m.i === index.value && m.dot);
        return mark ? [{ ...mark, field: series.field }] : [];
      })
    : [],
);
const starPoints = (cx: number, cy: number) =>
  Array.from({ length: 10 }, (_, i) => {
    const angle = -Math.PI / 2 + i * Math.PI / 5;
    const radius = i % 2 === 0 ? 13 : 5.5;
    return `${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`;
  }).join(" ");
const status = computed(
  () =>
    `${playing.value ? "Playing" : "Paused"} · ${selected.value.label} · ${row.value.date}${row.value.partial ? " (partial)" : ""} · Daily ${pct(row.value.daily_share)} · Rolling7 ${pct(row.value.rolling_share)} (${row.value.rolling_start} – ${row.value.rolling_end})`,
);
let raf: number | null = null;
let lastTime: number | null = null;
let stepElapsed = 0;
let motionQuery: MediaQueryList | undefined;
let printQuery: MediaQueryList | undefined;
function pause() {
  playing.value = false;
  if (raf !== null) cancelAnimationFrame(raf);
  raf = null;
  lastTime = null;
}
function finite(value: number) {
  if (!Number.isFinite(value))
    throw new TypeError("A finite number is required");
  return value;
}
function seek(value: number) {
  finite(value);
  pause();
  position.value = Math.max(0, Math.min(lastIndex, value));
  stepElapsed = 0;
}
// Absolute source-clock time at configured speed. Independent of RAF history and
// reduced-motion preference: identical elapsed/configuration => identical SVG.
function renderAt(elapsedMs: number) {
  seek((finite(elapsedMs) * speed.value * lastIndex) / durationMs);
}
function reset() {
  seek(0);
}
function tick(now: number) {
  if (!playing.value) return;
  if (document.hidden || !active.value || printMode.value) {
    pause();
    return;
  }
  if (lastTime !== null) {
    const delta = (now - lastTime) * speed.value;
    if (reducedMotion.value) {
      stepElapsed += delta;
      const dayDuration = durationMs / Math.max(1, lastIndex);
      if (stepElapsed >= dayDuration) {
        position.value = Math.min(
          lastIndex,
          Math.floor(position.value) + Math.floor(stepElapsed / dayDuration),
        );
        stepElapsed %= dayDuration;
      }
    } else
      position.value = Math.min(
        lastIndex,
        position.value + (delta * lastIndex) / durationMs,
      );
  }
  lastTime = now;
  if (position.value >= lastIndex) {
    pause();
    return;
  }
  raf = requestAnimationFrame(tick);
}
function play() {
  if (
    playing.value ||
    typeof document === "undefined" ||
    document.hidden ||
    !active.value ||
    printMode.value
  )
    return;
  if (position.value >= lastIndex) {
    position.value = 0;
    stepElapsed = 0;
  }
  playing.value = true;
  lastTime = null;
  raf = requestAnimationFrame(tick);
}
function configure(options: {
  client?: string;
  viewport?: "full" | "seven";
  speed?: number;
}) {
  // Validate all fields before changing any state.
  if (
    options.client !== undefined &&
    !data.series.some((s) => s.id === options.client)
  )
    throw new RangeError("Unknown client identity");
  if (
    options.viewport !== undefined &&
    !["full", "seven"].includes(options.viewport)
  )
    throw new RangeError("Unknown viewport");
  if (
    options.speed !== undefined &&
    (!Number.isFinite(options.speed) || options.speed <= 0)
  )
    throw new RangeError("Speed must be finite and positive");
  if (options.client !== undefined) client.value = options.client;
  if (options.viewport !== undefined) viewport.value = options.viewport;
  if (options.speed !== undefined) {
    speed.value = options.speed;
    lastTime = null;
  }
}
function visibilityChanged() {
  if (document.hidden) pause();
}
function motionChanged() {
  pause();
  reducedMotion.value = motionQuery?.matches ?? false;
  stepElapsed = 0;
}
function beforePrint() {
  pause();
  browserPrint.value = true;
}
function afterPrint() {
  browserPrint.value = false;
}
function printChanged() {
  if (printQuery?.matches) beforePrint();
  else afterPrint();
}
watch(active, (value) => {
  if (!value) pause();
});
watch(printMode, (value) => {
  if (value) pause();
});
watch(
  () => props.capture,
  () => pause(),
);
onMounted(() => {
  motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  printQuery = window.matchMedia("print");
  motionChanged();
  printChanged();
  motionQuery.addEventListener("change", motionChanged);
  printQuery.addEventListener("change", printChanged);
  document.addEventListener("visibilitychange", visibilityChanged);
  window.addEventListener("beforeprint", beforePrint);
  window.addEventListener("afterprint", afterPrint);
});
onBeforeUnmount(() => {
  pause();
  motionQuery?.removeEventListener("change", motionChanged);
  printQuery?.removeEventListener("change", printChanged);
  document.removeEventListener("visibilitychange", visibilityChanged);
  window.removeEventListener("beforeprint", beforePrint);
  window.removeEventListener("afterprint", afterPrint);
});
defineExpose({
  seek,
  renderAt,
  play,
  pause,
  reset,
  configure,
  state,
  durationMs,
});
</script>

<template>
  <section
    class="protocol-adoption-chart"
    :data-state="playing ? 'playing' : 'paused'"
    :data-position="renderedPosition"
    :data-index="index"
    :data-client="client"
    :data-viewport="viewport"
    :data-partial="row.partial"
    :data-capture="capture"
    :data-reduced-motion="reducedMotion"
    aria-label="Protocol adoption dashboard"
  >
    <div class="pa-stage">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 1600 900"
        font-family="Arial,Helvetica,sans-serif"
        style="font-variant-numeric: tabular-nums"
        role="img"
        :aria-labelledby="`${uid}-title ${uid}-desc`"
        data-role="stage"
      >
        <title :id="`${uid}-title`">{{ title }} · {{ data.protocol }}</title>
        <desc :id="`${uid}-desc`">
          Muted slate daily tool-call share and purple count-weighted rolling seven
          completed days. Five hashes excluded; recognized protocol versions
          only. Amber hollow point and dotted segment denote a partial day. Data
          table and snapshot provenance follow controls.
        </desc>
        <rect width="1600" height="900" fill="white" />
        <rect x="45" y="36" width="9" height="48" rx="2" fill="#6430d8" />
        <g fill="#172032">
          <text x="80" y="77" style="font-size: 54px" font-weight="800">
            {{ title }}
          </text>
          <text
            data-role="clock"
            x="1520"
            y="150"
            text-anchor="end"
            style="font-size: 31px"
            font-weight="800"
          >
            {{ row.date + " UTC" }}
          </text>
          <text
            data-role="client-label"
            x="80"
            y="150"
            style="font-size: 29px"
            font-weight="700"
          >
            {{ selected.label }}
          </text>
          <g data-role="axes">
            <g v-for="p in [0, 20, 40, 60, 80, 100]" :key="p">
              <line
                x1="80"
                x2="1500"
                :y1="y(p / 100)"
                :y2="y(p / 100)"
                stroke="#e7e9ee"
              />
              <text
                x="65"
                :y="y(p / 100) + 6"
                style="font-size: 18px"
                fill="#64748b"
                text-anchor="end"
              >
                {{ p }}%
              </text>
            </g>
          </g>
          <g data-role="half-reference">
            <line x1="80" x2="1500" :y1="y(0.5)" :y2="y(0.5)"
              stroke="#94a3b8" stroke-dasharray="5 8" style="opacity: 0.45" />
            <text x="65" :y="y(0.5) + 6" text-anchor="end"
              fill="#64748b" style="font-size: 18px; opacity: 0.8">50%</text>
          </g>
          <defs>
            <clipPath :id="`${uid}-clip`" clipPathUnits="userSpaceOnUse">
              <rect x="70" y="195" width="1440" height="640" />
            </clipPath>
          </defs>
          <g data-role="lines" :clip-path="`url(#${uid}-clip)`">
            <g
              v-for="series in plotted"
              :key="series.field"
              :data-series="series.field"
            >
              <g
                v-for="mark in series.marks"
                :key="mark.i"
                :data-index="mark.i"
                :data-partial="mark.partial"
              >
                <line
                  v-if="mark.line"
                  v-bind="mark.line"
                  :stroke="mark.color"
                  :stroke-width="series.field === 'daily_share' ? 2.5 : 7.5"
                  :style="{ opacity: series.field === 'daily_share' && !mark.partial ? 0.6 : 1 }"
                  :stroke-dasharray="mark.partial ? '3 9' : undefined"
                  :stroke-linecap="mark.partial ? 'round' : undefined"
                  data-role="connection"
                />
                <circle
                  v-if="mark.dot && !(renderedPosition === lastIndex && mark.i === index)"
                  :data-index="mark.i"
                  :data-field="series.field"
                  :cx="mark.cx"
                  :cy="mark.cy"
                  :r="mark.partial ? 8 : series.field === 'daily_share' ? 2.5 : 5"
                  :fill="mark.partial ? 'white' : mark.color"
                  :stroke="mark.color"
                  :stroke-width="series.field === 'daily_share' ? 1.5 : 3"
                  data-role="point"
                >
                  <title>
                    {{ mark.date }}: {{ pct(mark.v)
                    }}{{ mark.partial ? " (partial)" : "" }}
                  </title>
                </circle>
              </g>
            </g>
          </g>
          <g data-role="ticks">
            <text
              v-for="t in ticks"
              :key="t.i"
              :x="t.x"
              y="865"
              style="font-size: 18px"
              fill="#64748b"
              text-anchor="middle"
            >
              {{ t.label }}
            </text>
          </g>
          <!-- Compact translucent inset: high client traces remain visible beneath
               the backing, rather than disappearing behind a full plot mask. -->
          <g data-role="kpi">
            <rect x="90" y="216" width="560" height="220" rx="12"
              fill="white" style="opacity: 0.88" />
            <text data-role="rolling" x="100" y="290"
              style="font-size: 104px" font-weight="800" :fill="purple">
              {{ pct(row.rolling_share) }}
            </text>
            <text x="490" y="254" style="font-size: 19px"
              font-weight="800" :fill="purple">ROLLING 7</text>
            <text x="490" y="281" style="font-size: 17px"
              fill="#64748b">count-weighted</text>
            <text x="100" y="330" style="font-size: 26px" fill="#475569">
              of tool calls · trailing 7 days
            </text>
            <text x="100" y="361" style="font-size: 19px" fill="#64748b">
              <tspan data-role="rolling-start">{{ row.rolling_start }}</tspan>
              <tspan> → </tspan>
              <tspan data-role="rolling-end">{{ row.rolling_end }} UTC</tspan>
              <tspan> · completed</tspan>
            </text>
            <text data-role="daily-heading" x="100" y="412"
              style="font-size: 20px" font-weight="700" :fill="slate">Daily</text>
            <text data-role="daily" x="162" y="414"
              style="font-size: 40px" font-weight="700" :fill="row.partial ? amber : slate">
              {{ pct(row.daily_share) }}
            </text>
            <text data-role="partial-label" x="320" y="412"
              style="font-size: 20px" :fill="amber">{{ row.partial ? "○ Partial day" : "" }}</text>
          </g>
          <!-- Callouts are separate from the KPI test roles and point counts.
               Daily sits above its endpoint, rolling below, including at 0/100%. -->
          <g v-for="mark in endpoints" :key="mark.field"
            data-role="endpoint" :data-field="mark.field">
            <polygon :points="starPoints(mark.cx, mark.cy)"
              :fill="mark.partial ? 'white' : mark.color" :stroke="mark.color"
              stroke-width="2.5" stroke-linejoin="round" />
            <text :x="mark.cx - 18"
              :y="mark.field === 'daily_share' ? mark.cy - 22 : (mark.cy > 790 ? 892 : mark.cy + 39)"
              text-anchor="end" :fill="mark.color" font-weight="800"
              style="font-size: 30px; paint-order: stroke; stroke: white; stroke-width: 6px; stroke-linejoin: round">
              {{ pct(mark.v) }}
            </text>
          </g>
        </g>
      </svg>
    </div>
    <div
      v-if="!capture && !printMode"
      class="pa-controls"
      data-role="controls"
      @keydown.stop
      @pointerdown.stop
      @click.stop
    >
      <div class="pa-toolbar">
        <button
          type="button"
          data-action="play"
          :aria-pressed="playing"
          @click="playing ? pause() : play()"
        >
          {{ playing ? "Pause" : "Play" }}
        </button>
        <button type="button" data-action="reset" @click="reset">Reset</button>
        <label
          >Client
          <select
            data-control="client"
            :value="client"
            @change="
              configure({ client: ($event.target as HTMLSelectElement).value })
            "
          >
            <option v-for="s in data.series" :key="s.id" :value="s.id">
              {{ s.label }}
            </option>
          </select></label
        >
        <label
          >Plot viewport
          <select
            data-control="viewport"
            :value="viewport"
            @change="
              configure({
                viewport: ($event.target as HTMLSelectElement).value as
                  'full' | 'seven',
              })
            "
          >
            <option value="full">Full history</option>
            <option value="seven">Sliding 7 days</option>
          </select></label
        >
        <label
          >Speed
          <select
            data-control="speed"
            :value="speed"
            @change="
              configure({
                speed: Number(($event.target as HTMLSelectElement).value),
              })
            "
          >
            <option
              v-for="s in [...new Set([0.5, 1, 2, 4, speed])].sort(
                (a, b) => a - b,
              )"
              :key="s"
              :value="s"
            >
              {{ s }}×
            </option>
          </select></label
        >
        <button
          type="button"
          data-action="provenance"
          :aria-expanded="detailsOpen"
          :aria-controls="`${uid}-details`"
          @click="detailsOpen = !detailsOpen"
        >
          Data &amp; provenance
        </button>
      </div>
      <label class="pa-scrub"
        >Observation (UTC)<input
          data-control="scrubber"
          type="range"
          min="0"
          :max="lastIndex"
          step="0.001"
          :value="renderedPosition"
          :aria-valuetext="row.date + (row.partial ? ' partial' : '')"
          @input="seek(Number(($event.target as HTMLInputElement).value))"
      /></label>
      <p class="pa-status" role="status" aria-live="off" data-role="status">
        {{ status }}
      </p>
    </div>
    <aside
      v-if="detailsOpen && !capture && !printMode"
      :id="`${uid}-details`"
      class="pa-details"
      aria-label="Data and provenance"
      @keydown.stop
      @pointerdown.stop
      @click.stop
    >
      <button type="button" @click="detailsOpen = false">
        Close data &amp; provenance
      </button>
      <p>
        <strong>Private aggregates: review before public sharing.</strong>
        {{ data.provenance.notice }}
      </p>
      <p>
        Plot viewport changes visible dates only, never the rolling metric.
        Rolling7 is count-weighted over exactly seven completed UTC calendar
        dates (preceding seven on a partial day). Null is unavailable, not zero.
        No minimum-volume suppression. Progressive straight connections are
        illustration, not intraday estimates. Date, rates and counts use
        floor(position). Reduced-motion playback steps daily.
      </p>
      <p>
        Publication cutoff: {{ data.provenance.snapshot_last_modified_utc }}.
        Revision discovered: {{ data.provenance.snapshot_observed_at_utc }}.
      </p>
      <nav aria-label="Download original source data">
        <a :href="dataUrl" download="data.json">data.json</a> ·
        <a :href="csvUrl" download="daily.csv">daily.csv</a> ·
        <a :href="provenanceUrl" download="provenance.json">provenance.json</a>
        · <a :href="readmeUrl" download="README.md">README</a>
      </nav>
      <div class="pa-table-wrap">
        <table>
          <caption>
            {{
              selected.label
            }}
            · shares and raw counts; — means unavailable
          </caption>
          <thead>
            <tr>
              <th
                v-for="heading in [
                  'Date UTC',
                  'Partial',
                  'Daily %',
                  'Rolling %',
                  'Rolling dates',
                  'All calls',
                  'Valid',
                  'Modern',
                  'Unknown/unreviewed',
                  'Rolling valid',
                  'Rolling modern',
                ]"
                :key="heading"
                scope="col"
              >
                {{ heading }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in selected.rows" :key="r.date">
              <th scope="row">{{ r.date }}</th>
              <td>{{ r.partial ? "Yes" : "No" }}</td>
              <td>{{ pct(r.daily_share) }}</td>
              <td>{{ pct(r.rolling_share) }}</td>
              <td>{{ r.rolling_start }} – {{ r.rolling_end }}</td>
              <td
                v-for="key in [
                  'all_calls',
                  'valid_calls',
                  'modern_calls',
                  'unknown_or_unreviewed_calls',
                  'rolling_valid_calls',
                  'rolling_modern_calls',
                ] as const"
                :key="key"
              >
                {{ r.counts[key] ?? "—" }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <details>
        <summary>Full snapshot provenance</summary>
        <pre>{{ JSON.stringify(data.provenance, null, 2) }}</pre>
      </details>
    </aside>
  </section>
</template>

<style scoped>
.protocol-adoption-chart {
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  position: relative;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: white;
  color: #172032;
  font-family: Arial, Helvetica, sans-serif;
  accent-color: #6430d8;
}
.protocol-adoption-chart * {
  box-sizing: border-box;
}
.pa-stage {
  flex: 1 1 0;
  min-height: 0;
  min-width: 0;
}
.pa-stage svg {
  display: block;
  width: 100%;
  height: 100%;
  font-family: Arial, Helvetica, sans-serif;
}
.pa-stage text {
  font-variant-numeric: tabular-nums;
}
.pa-controls {
  flex: 0 0 auto;
  padding: 4px 10px 6px;
  font-size: 11px;
  line-height: 1.3;
}
.pa-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px 9px;
}
.pa-controls label {
  display: flex;
  align-items: center;
  gap: 5px;
}
.pa-controls button,
.pa-controls select,
.pa-details button {
  font:
    600 11px Arial,
    sans-serif;
  padding: 5px 7px;
  border: 1px solid #ccd1da;
  border-radius: 5px;
  background: white;
  color: #172032;
  max-width: 100%;
}
.pa-controls button,
.pa-details button {
  cursor: pointer;
}
.pa-controls [data-action="play"] {
  background: #6430d8;
  color: white;
}
.pa-scrub {
  margin-top: 4px;
}
.pa-scrub input {
  flex: 1;
  min-width: 0;
}
.pa-status {
  margin: 2px 0 0;
  font-size: 10px;
  color: #475569;
}
.protocol-adoption-chart :is(button, select, input, summary, a):focus-visible {
  outline: 3px solid #9063eb;
  outline-offset: 2px;
}
.pa-details {
  position: absolute;
  inset: 8px;
  z-index: 2;
  overflow: auto;
  padding: 16px;
  background: white;
  border: 2px solid #6430d8;
  border-radius: 6px;
  color: #172032;
  font-size: 13px;
  line-height: 1.5;
}
.pa-details p {
  margin: 10px 0;
}
.pa-details a {
  color: #6430d8;
  text-decoration: underline;
}
.pa-details pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: 11px;
}
.pa-details summary {
  cursor: pointer;
}
.pa-table-wrap {
  overflow: auto;
  margin: 12px 0;
}
.pa-details table {
  border-collapse: collapse;
  width: 100%;
  font-size: 11px;
}
.pa-details th,
.pa-details td {
  padding: 6px;
  text-align: right;
  border-bottom: 1px solid #ddd;
}
.pa-details caption {
  text-align: left;
}
@media print {
  .pa-controls,
  .pa-details {
    display: none !important;
  }
}
</style>
