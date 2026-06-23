<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { gepaBirchRun } from './gepaBirchRun'

type Candidate = {
  id: string
  iteration: number
  image: string
  score: number
  generation: number
  checker: number
  hygiene: number
  skillLengthScore: number
  skillLines: number
  recipeLines: number
  toolCalls: number
  turns: number
  totalTokens: number
  checkerFailures: number
  checkerWarnings: number
  missingCssArtifacts: number
  scoreCap: number
  hasOutput?: boolean
}

const props = withDefaults(defineProps<{ run?: typeof gepaBirchRun }>(), {
  run: () => gepaBirchRun,
})
const runData = computed(() => props.run)
const items = computed(() => (runData.value.items as readonly Candidate[]).filter(Boolean))
const order = ref<'iteration' | 'score'>('score')
const index = ref(0)
const playing = ref(false)
let timer: ReturnType<typeof window.setInterval> | undefined

const ordered = computed(() => {
  return [...items.value].sort((a, b) => {
    if (order.value === 'score') return a.score - b.score || a.iteration - b.iteration
    return a.iteration - b.iteration
  })
})
const current = computed(() => ordered.value[index.value] ?? ordered.value[0])
const rank = computed(() => ordered.value.findIndex((d) => d.id === current.value?.id) + 1)

const barWidth = (value: number, max = 1) => `${Math.max(3, Math.min(100, (value / max) * 100))}%`
const pct = (value: number) => `${Math.round(value * 100)}%`
const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`

function go(delta: number) {
  index.value = (index.value + delta + ordered.value.length) % ordered.value.length
}

function start() {
  playing.value = true
}

function stop() {
  playing.value = false
}

watch(playing, (isPlaying) => {
  if (timer) window.clearInterval(timer)
  timer = isPlaying ? window.setInterval(() => go(1), 2000) : undefined
}, { immediate: true })

watch(order, () => {
  index.value = 0
})

onBeforeUnmount(() => {
  if (timer) window.clearInterval(timer)
})
</script>

<template>
  <section v-if="current" class="gepa-run-explorer">
    <figure class="gepa-run-explorer__preview">
      <Transition name="gepa-swap" mode="out-in">
        <img v-if="current.hasOutput !== false" :key="current.id" :src="asset(current.image)" :alt="`${current.id} deep screenshot`" />
        <div v-else :key="current.id" class="gepa-run-explorer__empty">
          <strong>no output</strong>
          <span>artifact was not generated</span>
        </div>
      </Transition>
      <figcaption>
        <span>deep screenshot</span>
        <strong>{{ current.id }}</strong>
      </figcaption>
    </figure>

    <aside class="gepa-run-explorer__metrics">
      <header>
        <div>
          <div class="kicker">candidate {{ rank }} / {{ ordered.length }}</div>
          <h2>{{ pct(current.score) }}</h2>
          <p>GEPA score</p>
        </div>
        <div class="gepa-run-explorer__head-actions">
          <div class="gepa-run-explorer__iter">#{{ current.iteration }}</div>
          <div class="gepa-run-explorer__controls gepa-run-explorer__controls--top">
            <button type="button" @click="go(-1)">←</button>
            <button type="button" class="primary" @click="playing ? stop() : start()">
              {{ playing ? 'pause' : 'start' }}
            </button>
            <button type="button" @click="go(1)">→</button>
          </div>
        </div>
      </header>

      <div class="gepa-run-explorer__bars" aria-label="score components">
        <div class="gepa-run-explorer__bar gepa-run-explorer__bar--score">
          <span>GEPA</span>
          <div><i :style="{ width: barWidth(current.score) }" /></div>
          <strong>{{ current.score.toFixed(3) }}</strong>
        </div>
        <div class="gepa-run-explorer__bar">
          <span>checker</span>
          <div><i :style="{ width: barWidth(current.checker) }" /></div>
          <strong>{{ pct(current.checker) }}</strong>
        </div>
        <div class="gepa-run-explorer__bar">
          <span>hygiene</span>
          <div><i :style="{ width: barWidth(current.hygiene) }" /></div>
          <strong>{{ pct(current.hygiene) }}</strong>
        </div>
      </div>

      <div class="gepa-run-explorer__stats">
        <div>
          <span>SKILL.md</span>
          <strong>{{ current.skillLines }}</strong>
          <small>lines</small>
        </div>
        <div>
          <span>numeric-data.md</span>
          <strong>{{ current.recipeLines }}</strong>
          <small>lines</small>
        </div>
        <div>
          <span>tool calls</span>
          <strong>{{ current.toolCalls }}</strong>
          <small>{{ current.turns }} turns</small>
        </div>
        <div>
          <span>checker errors</span>
          <strong>{{ current.checkerFailures }}</strong>
          <small>{{ current.checkerWarnings }} warnings</small>
        </div>
      </div>

      <svg class="gepa-run-explorer__chart" viewBox="0 0 360 132" role="img" aria-label="GEPA score by candidate iteration">
        <line x1="0" y1="118" x2="360" y2="118" />
        <g v-for="point in items" :key="point.id" :transform="`translate(${((point.iteration - 1) / Math.max(items.length - 1, 1)) * 340 + 10} ${118 - point.score * 102})`">
          <circle :class="{ 'is-current': point.id === current.id }" r="6.2" />
        </g>
        <polyline :points="items.map((point) => `${((point.iteration - 1) / Math.max(items.length - 1, 1)) * 340 + 10},${118 - point.score * 102}`).join(' ')" />
      </svg>

      <footer class="gepa-run-explorer__controls">
        <span class="gepa-run-explorer__run">{{ runData.run }}</span>
        <label>
          order
          <select v-model="order">
            <option value="score">score</option>
            <option value="iteration">iteration</option>
          </select>
        </label>
      </footer>
    </aside>
  </section>
</template>

<style scoped>
.gepa-run-explorer {
  display: grid;
  grid-template-columns: minmax(0, 1.42fr) minmax(340px, 0.78fr);
  gap: 1.35rem;
  height: 456px;
  margin-top: 0.45rem;
}

.gepa-run-explorer__preview,
.gepa-run-explorer__metrics {
  min-width: 0;
  border: 1px solid rgba(246, 248, 255, 0.18);
  border-radius: 1.2rem;
  background: rgba(255, 255, 255, 0.07);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.24);
  overflow: hidden;
}

.gepa-run-explorer__preview {
  position: relative;
  display: block;
  padding: 0.75rem;
}

.gepa-run-explorer__preview img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center top;
  border-radius: 0.75rem;
  background: white;
}


.gepa-run-explorer__empty {
  display: grid;
  width: 100%;
  height: 100%;
  place-content: center;
  gap: 0.45rem;
  border: 1px dashed rgba(11, 16, 32, 0.18);
  border-radius: 0.75rem;
  background: rgba(255, 255, 255, 0.86);
  color: rgba(11, 16, 32, 0.52);
  text-align: center;
}

.gepa-run-explorer__empty strong {
  color: rgba(11, 16, 32, 0.78);
  font-size: 2.25rem;
  font-weight: 900;
  letter-spacing: -0.05em;
}

.gepa-run-explorer__empty span {
  font-size: 0.72rem;
  font-weight: 850;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.gepa-run-explorer__preview figcaption {
  position: absolute;
  left: 1.25rem;
  top: 1.2rem;
  display: flex;
  gap: 0.55rem;
  align-items: center;
  padding: 0.42rem 0.65rem;
  border: 1px solid rgba(11, 16, 32, 0.14);
  border-radius: 999px;
  background: rgba(11, 16, 32, 0.78);
  color: rgba(246, 248, 255, 0.72);
  font-size: 0.66rem;
  font-weight: 760;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  backdrop-filter: blur(10px);
}

.gepa-run-explorer__preview figcaption strong {
  color: white;
}

.gepa-run-explorer__metrics {
  display: flex;
  flex-direction: column;
  gap: 0.48rem;
  padding: 0.72rem 0.85rem;
}

.gepa-run-explorer__metrics header {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: start;
}

.gepa-run-explorer__metrics .kicker {
  margin-bottom: 0.35rem;
  font-size: 0.58rem;
}

.gepa-run-explorer__metrics h2 {
  margin: 0;
  color: #8ee8ff;
  font-size: 2.45rem;
  font-weight: 900;
  line-height: 0.9;
  letter-spacing: -0.06em;
}

.gepa-run-explorer__metrics p {
  margin: 0.25rem 0 0;
  color: rgba(246, 248, 255, 0.62);
  font-size: 0.82rem;
  font-weight: 760;
}


.gepa-run-explorer__head-actions {
  display: grid;
  gap: 0.42rem;
  justify-items: end;
}

.gepa-run-explorer__controls--top {
  margin: 0;
}

.gepa-run-explorer__controls--top button {
  padding: 0.34rem 0.5rem;
}

.gepa-run-explorer__iter {
  padding: 0.42rem 0.58rem;
  border: 1px solid rgba(255, 184, 107, 0.34);
  border-radius: 0.7rem;
  color: #ffb86b;
  font-size: 1rem;
  font-weight: 850;
}

.gepa-run-explorer__bars {
  display: grid;
  gap: 0.45rem;
}

.gepa-run-explorer__bar {
  display: grid;
  grid-template-columns: 4.4rem minmax(0, 1fr) 3.3rem;
  gap: 0.55rem;
  align-items: center;
  color: rgba(246, 248, 255, 0.62);
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.gepa-run-explorer__bar div {
  height: 0.62rem;
  border-radius: 999px;
  background: rgba(246, 248, 255, 0.1);
  overflow: hidden;
}

.gepa-run-explorer__bar i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #8ee8ff, #c4b5fd);
  transition: width 420ms ease;
}

.gepa-run-explorer__bar--score i {
  background: linear-gradient(90deg, #ffb86b, #8ee8ff);
}

.gepa-run-explorer__bar strong {
  color: white;
  text-align: right;
}

.gepa-run-explorer__stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.52rem;
}

.gepa-run-explorer__stats div {
  padding: 0.5rem 0.56rem;
  border: 1px solid rgba(246, 248, 255, 0.14);
  border-radius: 0.9rem;
  background: rgba(0, 0, 0, 0.14);
}

.gepa-run-explorer__stats span,
.gepa-run-explorer__stats small {
  display: block;
  color: rgba(246, 248, 255, 0.58);
  font-size: 0.54rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.gepa-run-explorer__stats strong {
  display: block;
  margin: 0.18rem 0 0.04rem;
  color: white;
  font-size: 1.32rem;
  font-weight: 900;
  line-height: 1;
}

.gepa-run-explorer__chart {
  width: 100%;
  height: 88px;
  padding: 0.15rem 0;
  overflow: visible;
}

.gepa-run-explorer__chart line {
  stroke: rgba(246, 248, 255, 0.16);
  stroke-width: 2;
}

.gepa-run-explorer__chart polyline {
  fill: none;
  stroke: rgba(142, 232, 255, 0.52);
  stroke-width: 3.6;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.gepa-run-explorer__chart circle {
  fill: rgba(246, 248, 255, 0.56);
  stroke: rgba(11, 16, 32, 0.9);
  stroke-width: 2;
  transition: r 180ms ease, fill 180ms ease;
}

.gepa-run-explorer__chart circle.is-current {
  r: 10.5px;
  fill: #ffb86b;
}

.gepa-run-explorer__mini {
  display: grid;
  grid-template-columns: 3.2rem minmax(0, 1fr);
  gap: 0.42rem 0.62rem;
  align-items: center;
  color: rgba(246, 248, 255, 0.54);
  font-size: 0.62rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.gepa-run-explorer__mini i {
  height: 0.42rem;
  border-radius: 999px;
  background: rgba(255, 184, 107, 0.78);
  transition: width 420ms ease;
}

.gepa-run-explorer__controls {
  display: flex;
  gap: 0.42rem;
  align-items: center;
  justify-content: space-between;
  margin-top: 0;
}

.gepa-run-explorer__run {
  min-width: 0;
  overflow: hidden;
  color: rgba(246, 248, 255, 0.5);
  font-size: 0.56rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-overflow: ellipsis;
  text-transform: uppercase;
  white-space: nowrap;
}


.gepa-run-explorer__controls button,
.gepa-run-explorer__controls select {
  border: 1px solid rgba(246, 248, 255, 0.18);
  border-radius: 0.65rem;
  background: rgba(246, 248, 255, 0.08);
  color: white;
  font-size: 0.72rem;
  font-weight: 850;
}

.gepa-run-explorer__controls button {
  min-width: 2.15rem;
  padding: 0.42rem 0.58rem;
  text-transform: uppercase;
}

.gepa-run-explorer__controls button.primary {
  min-width: 4.3rem;
  background: rgba(142, 232, 255, 0.18);
  color: #8ee8ff;
}

.gepa-run-explorer__controls label {
  display: flex;
  gap: 0.42rem;
  align-items: center;
  margin-left: 0;
  color: rgba(246, 248, 255, 0.56);
  font-size: 0.62rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.gepa-run-explorer__controls select {
  padding: 0.42rem 1.6rem 0.42rem 0.5rem;
  color-scheme: dark;
}

.gepa-swap-enter-active,
.gepa-swap-leave-active {
  transition: opacity 280ms ease, transform 280ms ease;
}

.gepa-swap-enter-from {
  opacity: 0;
  transform: translateX(12px) scale(0.985);
}

.gepa-swap-leave-to {
  opacity: 0;
  transform: translateX(-12px) scale(0.985);
}
</style>
