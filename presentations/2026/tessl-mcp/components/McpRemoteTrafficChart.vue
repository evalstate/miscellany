<script setup lang="ts">
import { computed } from 'vue'
import weeklyRows from '../data-viz/mcp_remote_share_weekly.json'

type ClientFamily = 'Codex' | 'Claude Code'
type RangeMode = 'client' | 'claude'

const props = withDefaults(
  defineProps<{
    client: ClientFamily
    rangeMode?: RangeMode
    title?: string
    subtitle?: string
  }>(),
  {
    rangeMode: 'client',
    title: undefined,
    subtitle: undefined,
  },
)

type WeeklyRow = {
  week_start: string
  week_end: string
  client_family: ClientFamily
  mcp_remote_share_pct: number
  usage_index_0_100: number
  mcp_remote_requests: number
  total_requests: number
}

const rows = weeklyRows as WeeklyRow[]
const width = 1000
const height = 520
const plot = {
  left: 74,
  right: 40,
  top: 54,
  bottom: 82,
}
const plotWidth = width - plot.left - plot.right
const plotHeight = height - plot.top - plot.bottom

const titleText = computed(() => props.title ?? `${props.client}: mcp-remote share`)

const subtitleText = computed(() => {
  if (props.subtitle) return props.subtitle
  if (props.rangeMode === 'claude' && props.client === 'Codex') {
    return 'Codex plotted on the Claude Code date range'
  }
  return 'Weekly buckets · usage index in background'
})

const clientRows = computed(() =>
  rows
    .filter((row) => row.client_family === props.client)
    .sort((a, b) => a.week_start.localeCompare(b.week_start)),
)

const claudeRows = computed(() =>
  rows
    .filter((row) => row.client_family === 'Claude Code')
    .sort((a, b) => a.week_start.localeCompare(b.week_start)),
)

const rangeRows = computed(() => {
  if (props.rangeMode === 'claude') return claudeRows.value
  return clientRows.value
})

function toDate(value: string) {
  const [year, month, day] = value.split('-').map(Number)
  return new Date(Date.UTC(year, month - 1, day))
}

const startDate = computed(() => toDate(rangeRows.value[0]?.week_start ?? clientRows.value[0]?.week_start))
const endDate = computed(() => toDate(rangeRows.value.at(-1)?.week_start ?? clientRows.value.at(-1)?.week_start))
const startMs = computed(() => startDate.value.getTime())
const endMs = computed(() => endDate.value.getTime())
const spanMs = computed(() => Math.max(1, endMs.value - startMs.value))

function xForDate(value: string) {
  const ms = toDate(value).getTime()
  return plot.left + ((ms - startMs.value) / spanMs.value) * plotWidth
}

function yForShare(value: number) {
  return plot.top + (1 - value / shareMax.value) * plotHeight
}

function yForUsage(value: number) {
  return plot.top + (1 - value / 100) * plotHeight
}

const visibleRows = computed(() =>
  clientRows.value.filter((row) => {
    const ms = toDate(row.week_start).getTime()
    return ms >= startMs.value && ms <= endMs.value
  }),
)

const shareMax = computed(() => {
  const max = Math.max(...visibleRows.value.map((row) => row.mcp_remote_share_pct), 1)
  if (max <= 10) return 10
  if (max <= 25) return 25
  if (max <= 50) return 50
  return 100
})

const shareTicks = computed(() => {
  if (shareMax.value === 10) return [0, 2.5, 5, 7.5, 10]
  if (shareMax.value === 25) return [0, 5, 10, 15, 20, 25]
  if (shareMax.value === 50) return [0, 10, 20, 30, 40, 50]
  return [0, 20, 40, 60, 80, 100]
})

function pointsFor(getY: (row: WeeklyRow) => number) {
  return visibleRows.value.map((row) => `${xForDate(row.week_start).toFixed(1)},${getY(row).toFixed(1)}`).join(' ')
}

const sharePolyline = computed(() => pointsFor((row) => yForShare(row.mcp_remote_share_pct)))

const usageArea = computed(() => {
  const points = visibleRows.value.map((row) => `${xForDate(row.week_start).toFixed(1)},${yForUsage(row.usage_index_0_100).toFixed(1)}`)
  if (points.length === 0) return ''
  const firstX = xForDate(visibleRows.value[0].week_start).toFixed(1)
  const lastX = xForDate(visibleRows.value.at(-1)!.week_start).toFixed(1)
  const baseY = plot.top + plotHeight
  return `M ${firstX} ${baseY} L ${points.join(' L ')} L ${lastX} ${baseY} Z`
})

const monthTicks = computed(() => {
  const ticks: { label: string; x: number }[] = []
  const cursor = new Date(Date.UTC(startDate.value.getUTCFullYear(), startDate.value.getUTCMonth(), 1))
  const formatter = new Intl.DateTimeFormat('en', { month: 'short' })
  let index = 0
  while (cursor.getTime() <= endMs.value) {
    const x = plot.left + ((cursor.getTime() - startMs.value) / spanMs.value) * plotWidth
    const showTick = index % 2 === 0 || cursor.getTime() >= endMs.value - 1000 * 60 * 60 * 24 * 34
    if (showTick && x >= plot.left - 1 && x <= plot.left + plotWidth + 1) {
      ticks.push({
        label: formatter.format(cursor),
        x,
      })
    }
    cursor.setUTCMonth(cursor.getUTCMonth() + 1)
    index += 1
  }
  return ticks
})

const latest = computed(() => visibleRows.value.at(-1))
const peakShare = computed(() => visibleRows.value.reduce((peak, row) => (row.mcp_remote_share_pct > peak.mcp_remote_share_pct ? row : peak), visibleRows.value[0]))
const latestShareLabel = computed(() => `${latest.value?.mcp_remote_share_pct.toFixed(1) ?? '—'}%`)
const peakShareLabel = computed(() => `${peakShare.value?.mcp_remote_share_pct.toFixed(1) ?? '—'}%`)
const latestUsageLabel = computed(() => `${latest.value?.usage_index_0_100.toFixed(1) ?? '—'}`)
const latestLabelX = computed(() => {
  if (!latest.value) return plot.left
  return Math.min(plot.left + plotWidth - 16, Math.max(plot.left + 90, xForDate(latest.value.week_start) - 12))
})
const latestLabelY = computed(() => {
  if (!latest.value) return plot.top
  return Math.max(plot.top + 30, Math.min(plot.top + plotHeight - 44, yForShare(latest.value.mcp_remote_share_pct) - 18))
})

</script>

<template>
  <section class="traffic-chart" :class="`traffic-chart--${props.client === 'Codex' ? 'codex' : 'claude'}`">
    <header class="traffic-chart__header">
      <div>
        <h1>{{ titleText }}</h1>
        <p>{{ subtitleText }}</p>
      </div>
      <div class="traffic-chart__badge">
        <span>mcp-remote share</span>
        <strong>{{ latestShareLabel }}</strong>
      </div>
    </header>

    <svg class="traffic-chart__svg" :viewBox="`0 0 ${width} ${height}`" role="img">
      <defs>
        <linearGradient :id="`usage-fill-${props.client.replaceAll(' ', '-')}-${props.rangeMode}`" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="rgba(106, 163, 247, 0.42)" />
          <stop offset="100%" stop-color="rgba(106, 163, 247, 0.03)" />
        </linearGradient>
        <linearGradient :id="`share-stroke-${props.client.replaceAll(' ', '-')}-${props.rangeMode}`" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#ffc649" />
          <stop offset="100%" stop-color="#f5a400" />
        </linearGradient>
        <filter :id="`share-glow-${props.client.replaceAll(' ', '-')}-${props.rangeMode}`" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect :x="plot.left" :y="plot.top" :width="plotWidth" :height="plotHeight" class="traffic-chart__plot-bg" />

      <g class="traffic-chart__grid">
        <line
          v-for="tick in shareTicks"
          :key="`share-${tick}`"
          :x1="plot.left"
          :x2="plot.left + plotWidth"
          :y1="yForShare(tick)"
          :y2="yForShare(tick)"
        />
        <line
          v-for="tick in monthTicks"
          :key="`month-line-${tick.label}-${tick.x}`"
          :x1="tick.x"
          :x2="tick.x"
          :y1="plot.top"
          :y2="plot.top + plotHeight"
        />
      </g>

      <path class="traffic-chart__usage-area" :d="usageArea" :fill="`url(#usage-fill-${props.client.replaceAll(' ', '-')}-${props.rangeMode})`" />
      <polyline
        class="traffic-chart__share-line"
        :points="sharePolyline"
        :stroke="`url(#share-stroke-${props.client.replaceAll(' ', '-')}-${props.rangeMode})`"
        :filter="`url(#share-glow-${props.client.replaceAll(' ', '-')}-${props.rangeMode})`"
      />

      <g class="traffic-chart__dots">
        <circle
          v-for="row in visibleRows"
          :key="row.week_start"
          :cx="xForDate(row.week_start)"
          :cy="yForShare(row.mcp_remote_share_pct)"
          r="3.5"
        />
      </g>

      <g class="traffic-chart__y-axis">
        <text
          v-for="tick in shareTicks"
          :key="`label-${tick}`"
          :x="plot.left - 14"
          :y="yForShare(tick) + 4"
          text-anchor="end"
        >
          {{ tick }}%
        </text>
      </g>

      <g class="traffic-chart__x-axis">
        <text
          v-for="tick in monthTicks"
          :key="`month-${tick.label}-${tick.x}`"
          :x="tick.x"
          :y="plot.top + plotHeight + 42"
          text-anchor="middle"
        >
          {{ tick.label }}
        </text>
      </g>

      <g v-if="latest" class="traffic-chart__latest">
        <line :x1="xForDate(latest.week_start)" :x2="xForDate(latest.week_start)" :y1="plot.top" :y2="plot.top + plotHeight" />
        <circle :cx="xForDate(latest.week_start)" :cy="yForShare(latest.mcp_remote_share_pct)" r="7" />
        <text :x="latestLabelX" :y="latestLabelY" text-anchor="end">
          {{ latestShareLabel }}
        </text>
      </g>

      <text class="traffic-chart__axis-title" :x="plot.left" :y="plot.top - 24">share of traffic using mcp-remote</text>
      <text class="traffic-chart__usage-label" :x="plot.left + plotWidth - 4" :y="plot.top + 24" text-anchor="end">opaque usage index</text>
    </svg>
  </section>
</template>

<style scoped>
.traffic-chart {
  position: relative;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 0.55rem;
  overflow: hidden;
  padding: 1rem 1.1rem 0.9rem;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 10px);
  background:
    radial-gradient(circle at 12% 12%, rgba(245, 164, 0, 0.12), transparent 28%),
    radial-gradient(circle at 78% 30%, rgba(106, 163, 247, 0.11), transparent 30%),
    rgba(20, 22, 27, 0.82);
  box-shadow: var(--deck-shadow);
}

.traffic-chart::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(245, 164, 0, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(245, 164, 0, 0.026) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.55), transparent 86%);
}

.traffic-chart > * {
  position: relative;
  z-index: 1;
}

.traffic-chart__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
}

.traffic-chart__header h1 {
  margin: 0;
  color: var(--deck-text);
  font-size: 2.25rem;
  line-height: 1;
  letter-spacing: -0.06em;
}

.traffic-chart__header p {
  margin: 0.42rem 0 0;
  color: var(--deck-dim);
  font-size: 0.66rem;
  font-weight: 850;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.traffic-chart__badge {
  min-width: 168px;
  padding: 0.58rem 0.78rem;
  border: 1px solid rgba(255, 198, 73, 0.38);
  border-radius: calc(var(--deck-radius) + 5px);
  background: rgba(11, 12, 15, 0.5);
  text-align: right;
}

.traffic-chart__badge span {
  display: block;
  color: var(--deck-dim);
  font-size: 0.54rem;
  font-weight: 850;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.traffic-chart__badge strong {
  display: block;
  margin-top: 0.12rem;
  color: var(--deck-accent-hi);
  font-size: 1.84rem;
  line-height: 1;
  letter-spacing: -0.06em;
}

.traffic-chart__svg {
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: visible;
}

.traffic-chart__plot-bg {
  fill: rgba(11, 12, 15, 0.34);
  stroke: rgba(58, 63, 74, 0.75);
  stroke-width: 1;
  rx: 12;
}

.traffic-chart__grid line {
  stroke: rgba(185, 179, 165, 0.13);
  stroke-width: 1;
}

.traffic-chart__usage-area {
  opacity: 0.92;
}

.traffic-chart__share-line {
  fill: none;
  stroke-width: 6.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.traffic-chart__dots circle {
  fill: var(--deck-accent-hi);
  stroke: rgba(11, 12, 15, 0.88);
  stroke-width: 1.5;
}

.traffic-chart__y-axis text,
.traffic-chart__x-axis text {
  fill: var(--deck-muted);
  font-family: var(--deck-font-mono);
  font-size: 15px;
  font-weight: 800;
}

.traffic-chart__axis-title,
.traffic-chart__usage-label {
  fill: var(--deck-dim);
  font-family: var(--deck-font-mono);
  font-size: 13px;
  font-weight: 900;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

.traffic-chart__latest line {
  stroke: rgba(255, 198, 73, 0.32);
  stroke-width: 2;
  stroke-dasharray: 6 8;
}

.traffic-chart__latest circle {
  fill: var(--deck-accent-hi);
  stroke: rgba(245, 164, 0, 0.38);
  stroke-width: 8;
}

.traffic-chart__latest text {
  fill: var(--deck-accent-hi);
  font-family: var(--deck-font-mono);
  font-size: 24px;
  font-weight: 950;
  letter-spacing: -0.06em;
}

</style>
