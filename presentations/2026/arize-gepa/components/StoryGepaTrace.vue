<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { storyGepaTraceData } from './storyGepaTraceData'

type TraceStep = (typeof storyGepaTraceData.steps)[number]

const steps = storyGepaTraceData.steps
const candidates = storyGepaTraceData.candidates
const index = ref(0)
const playing = ref(false)
let timer: ReturnType<typeof window.setInterval> | undefined

const current = computed<TraceStep>(() => steps[index.value] ?? steps[0])
const maxScore = computed(() => Math.max(...candidates.map((candidate) => candidate.score), 1))
const scoreText = (value: number) => value.toFixed(4)
const pct = (value: number) => `${Math.round(value * 100)}%`
const barWidth = (value: number) => `${Math.max(4, Math.min(100, value * 100))}%`
const shortPrompt = (prompt: string) => prompt.replace(/\s+/g, ' ')
const currentPointX = computed(() => pointX(current.value.iteration))
const currentPointY = computed(() => pointY(current.value.score))

function pointX(iteration: number) {
  return 18 + ((iteration - 1) / Math.max(1, candidates.length - 1)) * 624
}

function pointY(score: number) {
  return 86 - (score / maxScore.value) * 64
}

function go(delta: number) {
  index.value = (index.value + delta + steps.length) % steps.length
}

watch(playing, (isPlaying) => {
  if (timer) window.clearInterval(timer)
  timer = isPlaying ? window.setInterval(() => go(1), 4200) : undefined
}, { immediate: true })

onBeforeUnmount(() => {
  if (timer) window.clearInterval(timer)
})
</script>

<template>
  <section class="story-gepa-trace">
    <div class="story-gepa-trace__rail" aria-label="GEPA reflection steps">
      <button
        v-for="(step, stepIndex) in steps"
        :key="step.id"
        type="button"
        :class="{ 'is-active': stepIndex === index }"
        @click="index = stepIndex"
      >
        <span>{{ stepIndex + 1 }}</span>
        <strong>{{ scoreText(step.score) }}</strong>
      </button>
    </div>

    <div class="story-gepa-trace__grid">
      <article class="story-gepa-trace__panel story-gepa-trace__panel--prompt">
        <header>
          <span>{{ current.candidateId }}</span>
          <strong>candidate under test</strong>
        </header>
        <blockquote>{{ shortPrompt(current.currentPrompt) }}</blockquote>
        <div class="story-gepa-trace__score-row">
          <div>
            <span>GEPA score</span>
            <strong>{{ scoreText(current.score) }}</strong>
          </div>
          <div>
            <span>prompt</span>
            <strong>{{ current.promptWords }}w</strong>
          </div>
          <div>
            <span>story</span>
            <strong>{{ current.storyWords }}w</strong>
          </div>
        </div>
      </article>

      <article class="story-gepa-trace__panel story-gepa-trace__panel--asi">
        <header>
          <span>ASI packet</span>
          <strong>to reflection model</strong>
        </header>
        <div class="story-gepa-trace__bars">
          <div>
            <span>required</span>
            <i><b :style="{ width: barWidth(current.scores.requiredItems) }" /></i>
            <strong>{{ pct(current.scores.requiredItems) }}</strong>
          </div>
          <div>
            <span>story length</span>
            <i><b :style="{ width: barWidth(current.scores.storyLength) }" /></i>
            <strong>{{ pct(current.scores.storyLength) }}</strong>
          </div>
          <div>
            <span>prompt length</span>
            <i><b :style="{ width: barWidth(current.scores.promptLength) }" /></i>
            <strong>{{ pct(current.scores.promptLength) }}</strong>
          </div>
        </div>
        <ul>
          <li v-for="item in current.asi" :key="item">{{ item }}</li>
        </ul>
      </article>

      <article class="story-gepa-trace__panel story-gepa-trace__panel--proposal">
        <header>
          <span>{{ current.label }}</span>
          <strong>proposed replacement</strong>
        </header>
        <blockquote>{{ shortPrompt(current.proposedPrompt) }}</blockquote>
        <div class="story-gepa-trace__delta">
          <span>process move</span>
          <strong v-if="index === 0">infer hidden rubric</strong>
          <strong v-else-if="index < steps.length - 1">compress prompt</strong>
          <strong v-else>best observed prompt</strong>
        </div>
      </article>
    </div>

    <footer class="story-gepa-trace__footer">
      <svg viewBox="0 0 660 96" role="img" aria-label="GEPA score by candidate over the run">
        <line x1="18" y1="86" x2="642" y2="86" />
        <polyline :points="candidates.map((candidate) => `${pointX(candidate.iteration)},${pointY(candidate.score)}`).join(' ')" />
        <g v-for="candidate in candidates" :key="candidate.id" :transform="`translate(${pointX(candidate.iteration)} ${pointY(candidate.score)})`">
          <circle :class="{ 'is-current': candidate.iteration === current.iteration }" r="4.8" />
        </g>
        <g class="story-gepa-trace__current" :transform="`translate(${currentPointX} ${currentPointY})`">
          <circle r="9.5" />
          <text y="-14">{{ current.candidateId.replace('candidate-', '#') }}</text>
        </g>
      </svg>

      <div class="story-gepa-trace__controls">
        <button type="button" @click="go(-1)">←</button>
        <button type="button" class="primary" @click="playing = !playing">{{ playing ? 'pause' : 'start' }}</button>
        <button type="button" @click="go(1)">→</button>
      </div>
    </footer>
  </section>
</template>

<style scoped>
.story-gepa-trace {
  display: grid;
  grid-template-rows: auto 248px 66px;
  gap: 0.52rem;
  height: 420px;
  margin-top: 0.45rem;
}

.story-gepa-trace__rail {
  display: flex;
  gap: 0.55rem;
  align-items: center;
}

.story-gepa-trace__rail button,
.story-gepa-trace__controls button {
  border: 1px solid rgba(246, 248, 255, 0.18);
  border-radius: 0.75rem;
  background: rgba(255, 255, 255, 0.07);
  color: rgba(246, 248, 255, 0.78);
  font-weight: 850;
}

.story-gepa-trace__rail button {
  display: flex;
  gap: 0.45rem;
  align-items: center;
  padding: 0.46rem 0.72rem;
  font-size: 0.72rem;
}

.story-gepa-trace__rail button span {
  display: grid;
  place-items: center;
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 999px;
  background: rgba(142, 232, 255, 0.14);
  color: #8ee8ff;
}

.story-gepa-trace__rail button.is-active {
  border-color: rgba(255, 184, 107, 0.78);
  background: rgba(255, 184, 107, 0.2);
  color: white;
}

.story-gepa-trace__grid {
  display: grid;
  grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.25fr) minmax(0, 0.95fr);
  gap: 0.9rem;
  min-height: 0;
}

.story-gepa-trace__panel {
  min-width: 0;
  min-height: 0;
  padding: 0.76rem;
  border: 1px solid rgba(246, 248, 255, 0.18);
  border-radius: 1.05rem;
  background: rgba(255, 255, 255, 0.07);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.22);
  overflow: hidden;
}

.story-gepa-trace__panel--asi {
  border-color: rgba(142, 232, 255, 0.35);
  background: rgba(142, 232, 255, 0.08);
}

.story-gepa-trace__panel--proposal {
  border-color: rgba(255, 184, 107, 0.42);
}

.story-gepa-trace__panel--prompt,
.story-gepa-trace__panel--proposal {
  display: flex;
  flex-direction: column;
}

.story-gepa-trace__panel--prompt .story-gepa-trace__score-row,
.story-gepa-trace__panel--proposal .story-gepa-trace__delta {
  margin-top: auto;
}

.story-gepa-trace__panel header {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  align-items: baseline;
  margin-bottom: 0.52rem;
  color: #8ee8ff;
  font-size: 0.66rem;
  font-weight: 860;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.story-gepa-trace__panel header strong {
  color: rgba(246, 248, 255, 0.62);
  font-size: 0.58rem;
}

.story-gepa-trace blockquote {
  margin: 0;
  padding: 0.72rem 0.82rem;
  border: 1px solid rgba(142, 232, 255, 0.22);
  border-radius: 0.74rem;
  background: rgba(11, 16, 32, 0.48);
  color: rgba(246, 248, 255, 0.94);
  font-size: 0.98rem;
  font-weight: 760;
  line-height: 1.22;
  letter-spacing: -0.03em;
}

.story-gepa-trace__score-row {
  display: grid;
  grid-template-columns: 1.25fr 0.8fr 0.8fr;
  gap: 0.45rem;
  margin-top: 0.58rem;
}

.story-gepa-trace__score-row div,
.story-gepa-trace__delta {
  padding: 0.55rem;
  border: 1px solid rgba(246, 248, 255, 0.13);
  border-radius: 0.78rem;
  background: rgba(0, 0, 0, 0.14);
}

.story-gepa-trace__score-row span,
.story-gepa-trace__delta span {
  display: block;
  color: rgba(246, 248, 255, 0.58);
  font-size: 0.52rem;
  font-weight: 850;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.story-gepa-trace__score-row strong,
.story-gepa-trace__delta strong {
  display: block;
  margin-top: 0.2rem;
  color: #ffb86b;
  font-size: 1rem;
  font-weight: 900;
}

.story-gepa-trace__bars {
  display: grid;
  gap: 0.48rem;
  margin-bottom: 0.75rem;
}

.story-gepa-trace__bars div {
  display: grid;
  grid-template-columns: 5.3rem minmax(0, 1fr) 2.7rem;
  gap: 0.48rem;
  align-items: center;
  color: rgba(246, 248, 255, 0.64);
  font-size: 0.58rem;
  font-weight: 850;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.story-gepa-trace__bars i {
  height: 0.48rem;
  border-radius: 999px;
  background: rgba(246, 248, 255, 0.12);
  overflow: hidden;
}

.story-gepa-trace__bars b {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #ffb86b, #8ee8ff);
  transition: width 360ms ease;
}

.story-gepa-trace__bars strong {
  color: white;
  text-align: right;
}

.story-gepa-trace ul {
  display: grid;
  gap: 0.38rem;
  margin: 0;
  padding-left: 1rem;
}

.story-gepa-trace li {
  color: rgba(246, 248, 255, 0.78);
  font-size: 0.82rem;
  line-height: 1.16;
}

.story-gepa-trace__delta {
  margin-top: 0.58rem;
}

.story-gepa-trace__footer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.9rem;
  align-items: center;
  padding: 0.38rem 0.72rem;
  border: 1px solid rgba(246, 248, 255, 0.14);
  border-radius: 1rem;
  background: rgba(255, 255, 255, 0.055);
}

.story-gepa-trace__footer svg {
  width: 100%;
  height: 52px;
  overflow: visible;
}

.story-gepa-trace__footer line {
  stroke: rgba(246, 248, 255, 0.16);
  stroke-width: 2;
}

.story-gepa-trace__footer polyline {
  fill: none;
  stroke: rgba(142, 232, 255, 0.62);
  stroke-width: 3.2;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.story-gepa-trace__footer circle {
  fill: rgba(246, 248, 255, 0.55);
  stroke: rgba(11, 16, 32, 0.9);
  stroke-width: 2;
}

.story-gepa-trace__footer circle.is-current,
.story-gepa-trace__current circle {
  fill: #ffb86b;
}

.story-gepa-trace__current text {
  fill: #ffb86b;
  font-size: 15px;
  font-weight: 900;
  text-anchor: middle;
}

.story-gepa-trace__controls {
  display: flex;
  gap: 0.45rem;
}

.story-gepa-trace__controls button {
  min-width: 2.2rem;
  padding: 0.46rem 0.62rem;
  font-size: 0.72rem;
  text-transform: uppercase;
}

.story-gepa-trace__controls button.primary {
  min-width: 4.5rem;
  border-color: rgba(142, 232, 255, 0.34);
  background: rgba(142, 232, 255, 0.16);
  color: #8ee8ff;
}
</style>
