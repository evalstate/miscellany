<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId, watch } from 'vue'
import { useNav, useSlideContext } from '@slidev/client'
import { useRoute } from 'vue-router';
import data from '../data/tool-quality-version/data.json'
import changeData from '../data/tool-quality-version/changes.json'
import provenance from '../data/tool-quality-version/provenance.json'
import dataUrl from '../data/tool-quality-version/data.json?url'
import csvUrl from '../data/tool-quality-version/daily.csv?url'
import changesUrl from '../data/tool-quality-version/changes.json?url'
import provenanceUrl from '../data/tool-quality-version/provenance.json?url'
import readmeUrl from '../data/tool-quality-version/README.md?url'

const props = withDefaults(defineProps<{ capture?: boolean; title?: string; subtitle?: string; footer?: string }>(), {
  capture: false, title: 'Tool Error Rate', subtitle: 'hf_fs · daily quality-classified errors', footer: '',
})
const DAY = 86400000, WEEK = 7 * DAY
const START = Date.parse('2026-08-25T00:00:00Z'), END = Date.parse('2026-09-14T00:00:00Z')
const INTRO = 1200, DRAW = 30000, DROP = 450, durationMs = 33700
const clamp = (n: number, a = 0, b = 1) => Math.min(b, Math.max(a, n))
const iso = (t: number) => new Date(t).toISOString()
const date = (t: number) => new Date(t).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
const timeForDate = (t: number) => INTRO + (clamp(t, START, END) - START) / (END - START) * DRAW
const colors = { claude: '#6430d8', others: '#475569' }
type Series = keyof typeof colors
type Kind = 'claude' | 'server'
type Event = { time: number; date: string; version: string; kind: Kind; initial?: boolean }
const days = data.days.map(d => ({ ...d, time: Date.parse(d.date + 'T12:00:00Z') }))
const servers: Event[] = data.server_events.map(e => ({ ...e, time: Date.parse(e.date), kind: 'server' }))
const leaders: Event[] = days.filter((d, i) => i === 0 || d.claude_version !== days[i - 1].claude_version)
  .map(d => ({ time: d.time, date: iso(d.time), version: d.claude_version, kind: 'claude' }))
const events = [...leaders, ...servers].sort((a, b) => a.time - b.time)
  .map(e => ({ ...e, logicalTime: timeForDate(e.time) }))
const changes = changeData.map(c => {
  const event = servers.find(e => e.version === c.version)!
  return { ...c, time: event.time, logicalTime: timeForDate(event.time) }
})
const intervals: Record<Kind, Event[]> = { claude: leaders, server: [{ time: START, date: iso(START), version: data.initial_server_version, kind: 'server', initial: true }, ...servers] }
const ymax = Math.max(8, Math.ceil(Math.max(...days.flatMap(d => [d.claude.rate, d.others.rate]))))
const y = (v: number) => 840 - v / ymax * 540
const uid = `tqv-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
const clip = (name: string) => `url(#${uid}-${name})`
const elapsed = ref(durationMs), playing = ref(false), speed = ref(1), loop = ref(false)
const reduced = ref(false), browserPrint = ref(false), stage = ref<HTMLElement>()
const { $page, $nav, $renderContext } = useSlideContext()
const { isPrintMode } = useNav()
const deckRoute = useRoute();
const printMode = computed(() => browserPrint.value || deckRoute.path === '/print' || isPrintMode.value || $renderContext.value === 'print')
const active = computed(() => $nav.value.currentSlideNo === $page.value)
const clock = computed(() => printMode.value ? durationMs : elapsed.value)
const effectsOff = computed(() => reduced.value || printMode.value)
const t = computed(() => START + clamp((clock.value - INTRO) / DRAW) * (END - START))
const end = computed(() => Math.max(START + WEEK, t.value))
const start = computed(() => end.value - WEEK)
const x = (time: number) => 80 + (time - start.value) / WEEK * 1110
const dayIndex = computed(() => clock.value < INTRO ? -1 : days.findLastIndex(d => d.time <= t.value))
const observation = computed(() => days[dayIndex.value])
const change = computed(() => changes.findLast(c => clock.value >= c.logicalTime))
const state = computed(() => Object.freeze({
  position: clock.value, elapsed: clock.value, durationMs, playing: playing.value && !printMode.value,
  phase: clock.value < INTRO ? 'intro' : clock.value < INTRO + DRAW ? 'drawing' : clock.value < durationMs ? 'hold' : 'complete',
  state: playing.value ? 'playing' : clock.value >= durationMs ? 'complete' : clock.value === 0 ? 'idle' : 'paused',
  progress: (t.value - START) / (END - START), cursorTime: iso(t.value), viewportStart: iso(start.value), viewportEnd: iso(end.value),
  rolling: t.value > START + WEEK, dayIndex: dayIndex.value, day: observation.value?.date ?? null,
  clientVersion: leaders.findLast(e => e.time <= t.value)?.version ?? null,
  serverVersion: servers.findLast(e => e.time <= t.value)?.version ?? data.initial_server_version,
  activeChangeVersion: change.value?.version ?? null,
  currentEvents: Object.freeze(events.filter(e => clock.value >= e.logicalTime && clock.value < e.logicalTime + DROP).map(e => Object.freeze({ ...e }))),
}))
const ticks = computed(() => {
  const result: { time: number; x: number }[] = []
  for (let tick = Math.ceil(start.value / DAY) * DAY; tick < end.value; tick += DAY)
    if (x(tick) >= 107 && x(tick) <= 1163) result.push({ time: tick, x: x(tick) })
  return result
})
const traces = computed(() => (Object.keys(colors) as Series[]).reverse().map(key => {
  let path = ''
  for (let i = 0; i < days.length; i++) {
    const a = days[i], b = days[i + 1]
    if (a.time > t.value) continue
    if (b && b.time >= start.value) {
      const lo = Math.max(a.time, start.value), hi = Math.min(b.time, t.value, end.value)
      const value = (q: number) => a[key].rate + (b[key].rate - a[key].rate) * (q - a.time) / (b.time - a.time)
      if (hi >= lo) path += `M${x(lo)},${y(value(lo))}L${x(hi)},${y(value(hi))}`
    }
  }
  return { key, path, samples: days.filter(d => d.time <= t.value && d.time >= start.value && d.time <= end.value) }
}))
const guides = computed(() => events.map(e => {
  const origin = e.kind === 'claude' ? 198 : 273
  const p = effectsOff.value ? 1 : clamp((clock.value - e.logicalTime) / DROP)
  const settle = effectsOff.value ? 1 : clamp((clock.value - e.logicalTime) / 1200)
  const base = e.kind === 'server' ? .45 : .25
  return { ...e, origin, visible: clock.value >= e.logicalTime && e.time >= start.value && e.time <= t.value,
    landed: p >= 1, bottom: origin + (840 - origin) * (1 - (1 - p) ** 2), opacity: base + (.8 - base) * (1 - settle) }
}))
// Measure unscaled Arial text in SVG logical units. Re-run once mounted; no viewport-pixel assumptions.
const measureNode = ref<SVGTextElement>()
const measured = ref(false)
function textWidth(text: string, size: number, weight: number) {
  if (measured.value && measureNode.value) {
    measureNode.value.style.fontSize = `${size}px`
    measureNode.value.setAttribute('font-weight', String(weight))
    measureNode.value.textContent = text
    return measureNode.value.getComputedTextLength()
  }
  return text.length * size * .56
}
const ribbons = computed(() => (['claude', 'server'] as Kind[]).map(kind => ({ kind,
  segments: intervals[kind].map((e, i, list) => {
    const observedEnd = Math.max(e.time, Math.min(list[i + 1]?.time ?? END, t.value))
    const lo = Math.max(e.time, start.value), hi = Math.min(observedEnd, end.value)
    const w = Math.max(0, x(hi) - x(lo))
    const size = Math.min(22, Math.floor(22 * Math.max(0, w - 8) / Math.max(1, textWidth(e.version, 22, 700))))
    return { ...e, observedEnd, lo, hi, w, size, visible: e.time <= t.value && hi > lo, yy: kind === 'claude' ? 160 : 235,
      fill: kind === 'claude' ? ['#6430d8', '#8353e3'][i % 2] : ['#a56a00', '#c58a21'][i % 2] }
  }),
})))
function wrap(text: string, size: number, lineHeight: number, maxLines: number, weight: number) {
  let lines: string[] = []
  for (; size >= 12; size--) {
    lines = ['']
    for (const word of text.split(/\s+/)) {
      const prev = lines[lines.length - 1], next = prev ? `${prev} ${word}` : word
      if (prev && textWidth(next, size, weight) > 284) { lines[lines.length - 1] += ' '; lines.push(word) }
      else lines[lines.length - 1] = next
    }
    if (lines.length <= maxLines) break
  }
  return { lines, size, lineHeight }
}
const headline = computed(() => wrap(change.value?.title ?? '', 36, 42, 2, 800))
const detail = computed(() => wrap(change.value?.detail ?? '', 25, 32, 3, 400))
const noteOpacity = computed(() => !change.value ? 0 : effectsOff.value ? 1 : clamp((clock.value - change.value.logicalTime) / 350))
let raf: number | undefined, last: number | undefined
function pause() {
  playing.value = false
  if (raf !== undefined) cancelAnimationFrame(raf)
  raf = last = undefined
  return state.value
}
/** Both APIs take finite milliseconds on the ORIGINAL source clock, clamped to [0, 33700].
 * 0–1200 intro; 1200–31200 affine Aug25–Sep14 clock; 31200–33700 final hold.
 * Stops playback; all guide/fade effects are pure functions of this clock. Await Vue nextTick before capture.
 */
function renderAt(elapsedMs: number) {
  if (!Number.isFinite(elapsedMs)) throw new TypeError('Expected finite source-clock milliseconds')
  pause(); elapsed.value = clamp(elapsedMs, 0, durationMs)
  return state.value
}
function seek(position: number) { return renderAt(position) }
function reset() { return renderAt(0) }
function tick(now: number) {
  if (!playing.value) return
  if (document.hidden || !active.value || printMode.value) { pause(); return }
  if (last !== undefined) elapsed.value = Math.min(durationMs, elapsed.value + (now - last) * speed.value)
  last = now
  if (elapsed.value >= durationMs) {
    if (loop.value && !reduced.value) elapsed.value = 0
    else { pause(); return }
  }
  raf = requestAnimationFrame(tick)
}
function play() {
  if (playing.value || !active.value || printMode.value || typeof document === 'undefined' || document.hidden) return state.value
  if (elapsed.value >= durationMs) elapsed.value = 0
  playing.value = true; last = undefined; raf = requestAnimationFrame(tick)
  return state.value
}
async function fullscreen() { if (stage.value?.requestFullscreen) await stage.value.requestFullscreen().catch(() => {}) }
watch(active, value => { if (!value) pause() })
watch(printMode, value => { if (value) pause() })
watch(speed, () => { last = undefined })
let motionQuery: MediaQueryList | undefined, printQuery: MediaQueryList | undefined
function motionChanged() { reduced.value = motionQuery?.matches ?? false; if (reduced.value) renderAt(durationMs) }
function printChanged() { browserPrint.value = printQuery?.matches ?? false }
function visibilityChanged() { if (document.hidden) pause() }
function beforePrint() { browserPrint.value = true; pause() }
function afterPrint() { browserPrint.value = false }
onMounted(() => {
  measured.value = true
  motionQuery = matchMedia('(prefers-reduced-motion: reduce)'); motionChanged()
  printQuery = matchMedia('print'); printChanged()
  motionQuery.addEventListener('change', motionChanged); printQuery.addEventListener('change', printChanged)
  document.addEventListener('visibilitychange', visibilityChanged)
  window.addEventListener('beforeprint', beforePrint); window.addEventListener('afterprint', afterPrint)
})
onBeforeUnmount(() => {
  pause(); motionQuery?.removeEventListener('change', motionChanged); printQuery?.removeEventListener('change', printChanged)
  document.removeEventListener('visibilitychange', visibilityChanged)
  window.removeEventListener('beforeprint', beforePrint); window.removeEventListener('afterprint', afterPrint)
})
defineExpose({ seek, play, pause, reset, state, durationMs, renderAt })
</script>

<template>
  <div class="tqv-chart" data-tool-quality-version :data-phase="state.phase" :data-state="state.state" :data-elapsed="clock"
    :data-day="state.day" :data-rolling="state.rolling" :data-cursor-time="state.cursorTime"
    :data-viewport-start="state.viewportStart" :data-viewport-end="state.viewportEnd" :data-capture="capture" :data-print="printMode">
    <section ref="stage" class="tqv-stage" aria-label="Continuous seven-day tool quality timeline">
      <svg viewBox="0 0 1600 900" role="img" :aria-labelledby="`${uid}-title ${uid}-desc`">
        <title :id="`${uid}-title`">{{ title }} (Claude Code vs. Others)</title>
        <desc :id="`${uid}-desc`">Purple: Claude Code error rate. Slate: other clients. Two time-aligned version ribbons emit dashed lines from exact observed starts. The amber panel retains the latest verified tool-definition change, not a causal explanation. Daily noon UTC samples; connecting geometry is illustrative. Private aggregates: review before external sharing.</desc>
        <defs>
          <clipPath :id="`${uid}-plot`"><rect x="80" y="300" :width="clamp(x(t) - 80, 0, 1110)" height="540" /></clipPath>
          <clipPath :id="`${uid}-timeline`"><rect x="80" y="160" width="1110" height="680" /></clipPath>
          <clipPath :id="`${uid}-ribbon`"><rect x="80" y="160" width="1110" height="113" /></clipPath>
        </defs>
        <rect width="1600" height="900" fill="white" />
        <text ref="measureNode" visibility="hidden" aria-hidden="true" />
        <rect x="45" y="36" width="9" height="48" rx="2" fill="#6430d8" />
        <text x="80" y="76" style="font-size: 54px" font-weight="800" letter-spacing="-1.5"><slot name="title">{{ title }}</slot></text>
        <text x="80" y="118" style="font-size: 22px" fill="#64748b"><slot name="subtitle">{{ subtitle }}</slot></text>
        <text data-clock x="1510" y="76" text-anchor="end" style="font-size: 31px" font-weight="800">{{ date(Math.min(t, END - 1)).toUpperCase() }} 2026</text>
        <g data-axes :style="{ opacity: effectsOff ? 1 : clamp(clock / INTRO) }">
          <g v-for="v in ymax + 1" :key="v">
            <line x1="80" x2="1190" :y1="y(v - 1)" :y2="y(v - 1)" :stroke="v === 1 ? '#94a3b8' : '#e9ecf1'" stroke-width="1" />
            <text x="64" :y="y(v - 1) + 6" text-anchor="end" style="font-size: 18px" fill="#64748b">{{ v - 1 }}%</text>
          </g>
        </g>
        <g data-version-guides :clip-path="clip('timeline')">
          <g v-for="g in guides" :key="`${g.kind}-${g.version}`" class="tqv-version-marker" :data-kind="g.kind" :data-version="g.version" :data-time="g.time" :data-visible="g.visible" :data-landed="g.landed" :visibility="g.visible ? 'visible' : 'hidden'">
            <title>{{ g.kind === 'claude' ? 'Daily usage leader' : 'First observed MCP build' }} {{ g.version }} · {{ iso(g.time) }}</title>
            <line class="tqv-version-guide" :x1="x(g.time)" :x2="x(g.time)" :y1="g.origin" :y2="g.bottom" :stroke="g.kind === 'claude' ? colors.claude : '#a56a00'" :stroke-width="g.kind === 'server' ? 2.8 : 1.8" stroke-dasharray="9 8" stroke-linecap="round" :stroke-opacity="g.opacity" />
          </g>
        </g>
        <g data-plot :clip-path="clip('plot')">
          <path v-for="trace in traces" :key="trace.key" :data-trace="trace.key" :d="trace.path" fill="none" :stroke="colors[trace.key]" :stroke-width="trace.key === 'claude' ? 7 : 5" stroke-linejoin="round" stroke-linecap="round" />
          <g v-for="trace in traces" :key="trace.key" :data-samples="trace.key">
            <circle v-for="d in trace.samples" :key="d.date" :cx="x(d.time)" :cy="y(d[trace.key].rate)" r="4" :fill="colors[trace.key]" stroke="white" stroke-width="1.5"><title>{{ d.date }} noon UTC: {{ d[trace.key].rate }}%; {{ d[trace.key].failures }} / {{ d[trace.key].calls }} calls</title></circle>
          </g>
          <line data-cursor :x1="x(t)" :x2="x(t)" y1="300" y2="840" stroke="#64748b" stroke-opacity=".35" stroke-width="1" stroke-dasharray="3 6" />
        </g>
        <g data-ticks><text v-for="tick in ticks" :key="tick.time" :x="tick.x" y="858" text-anchor="middle" style="font-size: 18px" fill="#64748b">{{ date(tick.time) }}</text></g>
        <rect x="80" y="208" width="1110" height="27" fill="white" />
        <text x="80" y="150" style="font-size: 18px" font-weight="800" fill="#6430d8">CLAUDE CODE</text>
        <text data-client-version x="1190" y="151" text-anchor="end" style="font-size: 32px" font-weight="800" fill="#6430d8">{{ state.clientVersion ?? '—' }}</text>
        <text x="80" y="225" style="font-size: 18px" font-weight="800" fill="#986200">MCP SERVER</text>
        <text data-server-version x="1190" y="226" text-anchor="end" style="font-size: 32px" font-weight="800" fill="#986200">{{ state.serverVersion }}</text>
        <g :clip-path="clip('ribbon')">
          <g v-for="ribbon in ribbons" :key="ribbon.kind" :data-ribbon="ribbon.kind">
            <g v-for="s in ribbon.segments" :key="s.version" class="tqv-tape-segment" :data-kind="s.kind" :data-version="s.version" :data-start="s.time" :data-end="s.observedEnd" :data-visible-start="s.lo" :data-visible-end="s.hi" :data-initial="!!s.initial" :data-visible="s.visible" :visibility="s.visible ? 'visible' : 'hidden'">
              <title>{{ s.version }} · {{ iso(s.time) }} – {{ iso(s.observedEnd) }}</title>
              <rect :x="x(s.lo)" :y="s.yy" :width="s.w" height="38" :fill="s.fill" />
              <line :x1="x(s.time)" :x2="x(s.time)" :y1="s.yy" :y2="s.yy + 38" stroke="white" stroke-width="2" />
              <text :x="x(s.lo) + s.w / 2" :y="s.yy + 19 + Math.max(10, s.size) * .35" fill="white" text-anchor="middle" :style="{ fontSize: (Math.max(10, s.size)) + 'px' }" font-weight="700">{{ s.size >= 10 ? s.version : '' }}</text>
            </g>
          </g>
        </g>
        <line x1="1215" x2="1215" y1="160" y2="880" stroke="#e7e9ee" />
        <text x="1240" y="188" class="tqv-label" fill="#6430d8">CLAUDE CODE</text>
        <text data-rate="claude" x="1235" y="270" class="tqv-rate" fill="#6430d8">{{ observation ? observation.claude.rate.toFixed(2) + '%' : '—' }}</text>
        <text x="1240" y="320" class="tqv-label" fill="#475569">OTHER CLIENTS</text>
        <text data-rate="others" x="1235" y="402" class="tqv-rate" fill="#475569">{{ observation ? observation.others.rate.toFixed(2) + '%' : '—' }}</text>
        <rect x="1230" y="450" width="320" height="350" rx="6" fill="#fff5df" />
        <rect x="1230" y="450" width="4" height="350" fill="#a56a00" />
        <g data-change-note :data-version="change?.version ?? ''" :data-visible="!!change" :style="{ opacity: noteOpacity }">
          <text x="1248" y="482" style="font-size: 16px" font-weight="700" fill="#986200">{{ change ? `LAST TOOL CHANGE · ${change.version}` : '' }}</text>
          <text data-change-headline x="1248" y="543" :style="{ fontSize: (headline.size) + 'px' }" font-weight="800" fill="#172032"><tspan v-for="(line, i) in headline.lines" :key="i" x="1248" :dy="i ? headline.lineHeight : 0">{{ line }}</tspan></text>
          <text data-change-detail x="1248" y="651" :style="{ fontSize: (detail.size) + 'px' }" fill="#475569"><tspan v-for="(line, i) in detail.lines" :key="i" x="1248" :dy="i ? detail.lineHeight : 0">{{ line }}</tspan></text>
        </g>
        <text v-if="footer || $slots.footer" x="80" y="891" style="font-size: 16px" fill="#64748b"><slot name="footer">{{ footer }}</slot></text>
      </svg>
    </section>
    <div v-if="!capture && !printMode" class="tqv-interface" data-controls>
      <div class="tqv-toolbar" aria-label="Animation controls">
        <button type="button" data-play @click="playing ? pause() : play()">{{ playing ? 'Pause' : clock >= durationMs ? 'Replay' : 'Play' }}</button>
        <button type="button" @click="reset(); play()">Replay</button><button type="button" @click="renderAt(durationMs)">Show end</button>
        <label>Speed <select v-model.number="speed"><option v-for="s in [.5, 1, 1.5, 2]" :key="s" :value="s">{{ s }}×</option></select></label>
        <label><input v-model="loop" type="checkbox">Loop</label><button type="button" @click="fullscreen">Fullscreen</button>
        <span role="status" aria-live="polite">{{ playing ? 'Playing' : clock >= durationMs ? 'Complete' : clock === 0 ? 'Ready' : 'Paused' }}</span>
      </div>
      <label class="tqv-scrubber">Animation timeline <input data-scrubber type="range" min="0" :max="durationMs" step="any" :value="clock" :aria-valuetext="`${(clock / 1000).toFixed(1)} seconds; ${state.phase}; last observation ${state.day ?? 'none'}`" @input="seek(Number(($event.target as HTMLInputElement).value))"></label>
      <p>Daily observations: noon UTC. Linear connections are illustrative, not intraday samples. Rates retain the last revealed observation. Final seven-day window: September 7–13 (end exclusive).</p>
      <details data-method>
        <summary>Interpretation, data and provenance · private aggregates</summary>
        <p><strong>Observe errors → refine tool instructions → measure again.</strong> An illustrative instrumentation workflow, not evidence that logged errors prompted these edits or edits caused rate changes. Attribution does not prove client/server fault.</p>
        <p>The amber panel retains the last verified description/inputSchema change until superseded, not necessarily the current server version. 0.4.14 is unchanged. Guidance first tagged .17 is shown at observed .18; no invented .17 rollout.</p>
        <ol><li v-for="c in changes" :key="c.version" :data-change-version="c.version">{{ c.version }} · {{ iso(c.time) }} — {{ c.title }}. {{ c.detail }} Fields: {{ c.changed_fields.join(', ') }}. {{ c.before }} → {{ c.after }}<template v-if="c.introduced_tag">; first tagged {{ c.introduced_tag }}</template>.</li></ol>
        <p>Last observation: {{ observation ? `${observation.date} · noon UTC` : 'Awaiting noon UTC' }} · Published-cell coverage: {{ observation ? observation.coverage_pct.toFixed(1) + '%' : '—' }}.
          Claude: {{ observation ? `${observation.claude.failures} / ${observation.claude.calls}` : '—' }} calls; others: {{ observation ? `${observation.others.failures} / ${observation.others.calls}` : '—' }} calls.</p>
        <p>Claude versions are changed daily usage leaders, self-reported, not releases or first-ever sightings. MCP versions are first-observed build hours, not verified deployment/edit times. Initial server 0.4.12 has no introduction event. Ribbons stop at the clock; exact six-hour .14/.15 boundaries are neither widened nor merged. An old segment entering the left edge does not create a new guide.</p>
        <p>33.7 seconds: 1.2-second axes intro, 30-second affine clock, 2.5-second final hold; no event dwells. Guides grow downward over 450 logical milliseconds and settle over 1.2 seconds. Notes fade over 350 milliseconds. Pause freezes every effect. Reduced motion defaults to the last frame and disables effects during explicit scrubbing/playback. Print always renders the last frame.</p>
        <p>Exclusive tool_quality failed <strong>calls</strong> / selected published client-cell calls, not all failures or batch operations. Mixed-category failures and successful partial batches are excluded from the numerator. Others sums explicitly non-Claude cells, never all traffic minus Claude. Fixed five excluded; missing hashes retained. Client/version/day publication floor: 20 calls. Omitted/suppressed cells are not zero or assigned to either group. Coverage: published client-cell calls / complete daily hf_fs calls (98.35–99.36%). No deduplication; completeness unknown.</p>
        <p><strong>Private aggregates: review before external sharing.</strong> Compatible verified tool-quality publication ends September 13. Newer local September 14 protocol publications lack compatible quality classifications. No remote scan or reanalysis performed.</p>
        <div class="tqv-table-wrap"><table><caption>20 daily observations (UTC noon); rates in percent</caption><thead><tr><th>Date</th><th>CC calls</th><th>CC failures</th><th>CC %</th><th>Others calls</th><th>Others failures</th><th>Others %</th><th>Coverage %</th><th>CC leader</th></tr></thead><tbody><tr v-for="d in days" :key="d.date"><th scope="row">{{ d.date }}</th><td>{{ d.claude.calls }}</td><td>{{ d.claude.failures }}</td><td>{{ d.claude.rate.toFixed(2) }}</td><td>{{ d.others.calls }}</td><td>{{ d.others.failures }}</td><td>{{ d.others.rate.toFixed(2) }}</td><td>{{ d.coverage_pct.toFixed(2) }}</td><td>{{ d.claude_version }}</td></tr></tbody></table></div>
        <h3>Version observations (not release dates)</h3><p>Initial observed server: {{ data.initial_server_version }} (before window; no introduction event).</p>
        <ul><li v-for="e in events" :key="`${e.kind}-${e.version}`">{{ iso(e.time) }} — {{ e.kind === 'claude' ? 'CC daily usage leader' : 'MCP first observed build' }} {{ e.version }}</li></ul>
        <p class="tqv-links"><a :href="dataUrl" download>data.json</a><a :href="csvUrl" download>daily.csv</a><a :href="changesUrl" download>changes.json</a><a :href="provenanceUrl" download>provenance.json</a><a :href="readmeUrl" download>Source README</a></p>
        <details><summary>Unchanged source metadata</summary><pre>{{ JSON.stringify(data.meta, null, 2) }}</pre><pre>{{ JSON.stringify(provenance, null, 2) }}</pre></details>
      </details>
    </div>
  </div>
</template>

<style scoped>
.tqv-chart { width: 100%; height: 100%; min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; color: #172032; background: white; font-family: Arial, Helvetica, sans-serif; accent-color: #6430d8; }
.tqv-chart * { box-sizing: border-box; }
.tqv-stage { flex: 1 1 0; min-height: 0; min-width: 0; overflow: hidden; }
.tqv-stage svg { display: block; width: 100%; height: 100%; font-family: Arial, Helvetica, sans-serif; fill: #172032; }
.tqv-stage svg text { font-variant-numeric: tabular-nums; }
.tqv-stage:fullscreen { width: 100vw; height: 100vh; background: white; }
.tqv-label { font-size: 18px; font-weight: 800; letter-spacing: 1px; }
.tqv-rate { font-size: 80px; font-weight: 800; letter-spacing: -2px; }
.tqv-interface { flex: 0 1 auto; max-height: 30%; overflow: auto; padding: 4px 10px; font-size: 12px; line-height: 1.5; color: #475569; }
.tqv-toolbar, .tqv-toolbar label, .tqv-links { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.tqv-toolbar button, .tqv-toolbar select { font: 600 12px Arial; padding: 4px 9px; border: 1px solid #ccd1da; border-radius: 6px; background: white; color: #172032; cursor: pointer; }
.tqv-toolbar [data-play] { background: #6430d8; border-color: #6430d8; color: white; min-width: 65px; }
.tqv-toolbar [role=status] { margin-left: auto; }
.tqv-chart :is(button, input, select, summary, a):focus-visible { outline: 3px solid #9063eb; outline-offset: 2px; }
.tqv-scrubber { display: flex; align-items: center; gap: 8px; }
.tqv-scrubber input { flex: 1; min-width: 0; }
.tqv-interface p { margin: 6px 0; }
.tqv-interface summary { cursor: pointer; }
.tqv-interface pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 11px; }
.tqv-table-wrap { overflow: auto; }
.tqv-interface table { width: 100%; border-collapse: collapse; font-size: 11px; }
.tqv-interface th, .tqv-interface td { text-align: right; padding: 4px; border-bottom: 1px solid #ddd; }
.tqv-interface th:first-child { text-align: left; }
.tqv-links a { color: #6430d8; text-decoration: underline; }
@media print { .tqv-interface { display: none; } }
</style>
