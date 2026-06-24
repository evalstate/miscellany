<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { storyGepaTraceData } from './storyGepaTraceData'

type TraceStep = (typeof storyGepaTraceData.steps)[number]

const steps = storyGepaTraceData.steps
const candidates = storyGepaTraceData.candidates
const index = ref(0)
const playing = ref(false)
const demoStepDelayMs = 500
let timer: ReturnType<typeof window.setInterval> | undefined

const current = computed<TraceStep>(() => steps[index.value] ?? steps[0])
const maxScore = computed(() => Math.max(...candidates.map((candidate) => candidate.score), 1))
const scoreText = (value: number) => value.toFixed(4)
const pct = (value: number) => `${Math.round(value * 100)}%`
const rawScore = (value: number) => value.toFixed(3)
const barWidth = (value: number) => `${Math.max(4, Math.min(100, value * 100))}%`
const shortPrompt = (prompt: string) => prompt.replace(/\s+/g, ' ')
const currentPointX = computed(() => pointX(current.value.iteration))
const currentPointY = computed(() => pointY(current.value.score))
const processMove = computed(() => current.value.processMove)

function pointX(iteration: number) {
  return 20 + ((iteration - 1) / Math.max(1, candidates.length - 1)) * 700
}

function pointY(score: number) {
  return 80 - (score / maxScore.value) * 58
}

function go(delta: number) {
  index.value = (index.value + delta + steps.length) % steps.length
}

watch(playing, (isPlaying) => {
  if (timer) window.clearInterval(timer)
  timer = isPlaying ? window.setInterval(() => go(1), demoStepDelayMs) : undefined
}, { immediate: true })

onBeforeUnmount(() => {
  if (timer) window.clearInterval(timer)
})
</script>

<template>
  <section class="story-gepa-trace">
    <div class="story-gepa-trace__top">
      <nav class="story-gepa-trace__rail" aria-label="GEPA reflection steps">
      <button
        v-for="(step, stepIndex) in steps"
        :key="step.id"
        type="button"
        :class="{ 'is-active': stepIndex === index }"
        @click="index = stepIndex"
      >
        <span>{{ stepIndex + 1 }}</span>
        <strong>{{ scoreText(step.score) }}</strong>
        <small>{{ step.candidateId.replace('candidate-', '#') }}</small>
      </button>
      </nav>

      <main class="story-gepa-trace__main">
      <section class="story-gepa-trace__cards">
        <article class="story-gepa-trace__panel story-gepa-trace__panel--prompt">
          <header>
            <span>candidate {{ current.candidateId.replace('candidate-', '#') }}</span>
            <strong>under test</strong>
          </header>
          <blockquote>{{ shortPrompt(current.currentPrompt) }}</blockquote>
          <div class="story-gepa-trace__delta">
            <span>process move</span>
            <strong>{{ processMove }}</strong>
          </div>
        </article>

        <article class="story-gepa-trace__panel story-gepa-trace__panel--asi">
          <header>
            <span>ASI packet</span>
            <strong>to reflection model</strong>
          </header>
          <div class="story-gepa-trace__score-row story-gepa-trace__score-row--asi">
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
          <div class="story-gepa-trace__bars">
            <div>
              <span>required</span>
              <i><b :style="{ width: barWidth(current.scores.requiredItems) }" /></i>
              <strong><b>{{ rawScore(current.scores.requiredItems) }}</b><em>{{ pct(current.scores.requiredItems) }}</em></strong>
            </div>
            <div>
              <span>story length</span>
              <i><b :style="{ width: barWidth(current.scores.storyLength) }" /></i>
              <strong><b>{{ rawScore(current.scores.storyLength) }}</b><em>{{ pct(current.scores.storyLength) }}</em></strong>
            </div>
            <div>
              <span>prompt length</span>
              <i><b :style="{ width: barWidth(current.scores.promptLength) }" /></i>
              <strong><b>{{ rawScore(current.scores.promptLength) }}</b><em>{{ pct(current.scores.promptLength) }}</em></strong>
            </div>
          </div>
          <ul>
            <li v-for="item in current.asi" :key="item">{{ item }}</li>
          </ul>
        </article>
      </section>

      <footer class="story-gepa-trace__footer">
        <svg viewBox="0 0 740 92" role="img" aria-label="GEPA score by candidate over the run">
          <line x1="20" y1="80" x2="720" y2="80" />
          <polyline :points="candidates.map((candidate) => `${pointX(candidate.iteration)},${pointY(candidate.score)}`).join(' ')" />
          <g v-for="candidate in candidates" :key="candidate.id" :transform="`translate(${pointX(candidate.iteration)} ${pointY(candidate.score)})`">
            <circle :class="{ 'is-current': candidate.iteration === current.iteration }" r="5" />
          </g>
          <g class="story-gepa-trace__current" :transform="`translate(${currentPointX} ${currentPointY})`">
            <circle r="10" />
            <text y="-14">{{ current.candidateId.replace('candidate-', '#') }}</text>
          </g>
        </svg>

        <div class="story-gepa-trace__controls">
          <button type="button" @click="go(-1)">←</button>
          <button type="button" class="primary" @click="playing = !playing">{{ playing ? 'pause' : 'start' }}</button>
          <button type="button" @click="go(1)">→</button>
        </div>
      </footer>
      </main>
    </div>
  </section>
</template>

<style scoped>
.story-gepa-trace {
  display: grid;
  grid-template-rows: minmax(0, 1fr);
  height: 442px;
  margin-top: 0.55rem;
}

.story-gepa-trace__top {
  display: grid;
  grid-template-columns: 9.7rem minmax(0, 1fr);
  gap: 0.9rem;
  min-height: 0;
}

.story-gepa-trace__rail {
  display: grid;
  align-content: start;
  gap: 0.42rem;
  padding-top: 0.05rem;
}

.story-gepa-trace__rail button,
.story-gepa-trace__controls button {
  border: 1px solid rgba(246, 248, 255, 0.18);
  border-radius: 0.85rem;
  background: rgba(255, 255, 255, 0.07);
  color: rgba(246, 248, 255, 0.78);
  font-weight: 850;
}

.story-gepa-trace__rail button {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.1rem 0.55rem;
  align-items: center;
  padding: 0.46rem 0.62rem;
  text-align: left;
}

.story-gepa-trace__rail button span {
  grid-row: span 2;
  display: grid;
  place-items: center;
  width: 1.45rem;
  height: 1.45rem;
  border-radius: 999px;
  background: rgba(142, 232, 255, 0.16);
  color: #8ee8ff;
  font-size: 0.78rem;
}

.story-gepa-trace__rail button strong {
  color: white;
  font-size: 0.9rem;
}

.story-gepa-trace__rail button small {
  color: rgba(246, 248, 255, 0.54);
  font-size: 0.6rem;
  font-weight: 850;
  letter-spacing: 0.08em;
}

.story-gepa-trace__rail button.is-active {
  border-color: rgba(255, 184, 107, 0.78);
  background: rgba(255, 184, 107, 0.2);
  color: white;
}

.story-gepa-trace__rail button.is-active span {
  background: rgba(255, 184, 107, 0.24);
  color: #ffb86b;
}

.story-gepa-trace__main {
  display: grid;
  grid-template-rows: minmax(0, 1fr) 5.8rem;
  gap: 0.62rem;
  min-width: 0;
  min-height: 0;
}

.story-gepa-trace__cards {
  display: grid;
  grid-template-columns: minmax(0, 0.78fr) minmax(0, 1.22fr);
  gap: 0.9rem;
  min-height: 0;
}

.story-gepa-trace__panel {
  min-width: 0;
  min-height: 0;
  padding: 0.92rem;
  border: 1px solid rgba(246, 248, 255, 0.18);
  border-radius: 1.08rem;
  background: rgba(255, 255, 255, 0.07);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.22);
  overflow: hidden;
}

.story-gepa-trace__panel--prompt {
  display: flex;
  flex-direction: column;
}

.story-gepa-trace__panel--asi {
  border-color: rgba(142, 232, 255, 0.35);
  background: rgba(142, 232, 255, 0.08);
}

.story-gepa-trace__panel header {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  align-items: baseline;
  margin-bottom: 0.75rem;
  color: #8ee8ff;
  font-size: 0.72rem;
  font-weight: 860;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.story-gepa-trace__panel header strong {
  color: rgba(246, 248, 255, 0.64);
  font-size: 0.62rem;
}

.story-gepa-trace blockquote {
  margin: 0;
  padding: 0.86rem 0.9rem;
  border: 1px solid rgba(142, 232, 255, 0.24);
  border-radius: 0.82rem;
  background: rgba(11, 16, 32, 0.48);
  color: rgba(246, 248, 255, 0.95);
  font-size: 1.18rem;
  font-weight: 780;
  line-height: 1.2;
  letter-spacing: -0.035em;
}

.story-gepa-trace__score-row {
  display: grid;
  grid-template-columns: 1.25fr 0.8fr 0.8fr;
  gap: 0.55rem;
  margin-top: 0.85rem;
}

.story-gepa-trace__score-row--asi {
  margin-top: 0;
  margin-bottom: 0.82rem;
}

.story-gepa-trace__score-row div,
.story-gepa-trace__delta {
  padding: 0.72rem;
  border: 1px solid rgba(246, 248, 255, 0.13);
  border-radius: 0.82rem;
  background: rgba(0, 0, 0, 0.14);
}

.story-gepa-trace__delta {
  margin-top: auto;
}

.story-gepa-trace__score-row span,
.story-gepa-trace__delta span {
  display: block;
  color: rgba(246, 248, 255, 0.6);
  font-size: 0.56rem;
  font-weight: 850;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.story-gepa-trace__score-row strong,
.story-gepa-trace__delta strong {
  display: block;
  margin-top: 0.22rem;
  color: #ffb86b;
  font-size: 1.44rem;
  font-weight: 900;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}

.story-gepa-trace__bars {
  display: grid;
  gap: 0.58rem;
  margin-bottom: 0.82rem;
}

.story-gepa-trace__bars div {
  display: grid;
  grid-template-columns: 6.25rem minmax(0, 1fr) 4.5rem;
  gap: 0.58rem;
  align-items: center;
  color: rgba(246, 248, 255, 0.7);
  font-size: 0.68rem;
  font-weight: 850;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.story-gepa-trace__bars i {
  height: 0.6rem;
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
  display: grid;
  gap: 0.05rem;
  color: white;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.story-gepa-trace__bars strong b {
  display: block;
  height: auto;
  background: transparent;
  color: white;
  font-size: 0.86rem;
  line-height: 1;
}

.story-gepa-trace__bars strong em {
  color: rgba(246, 248, 255, 0.58);
  font-size: 0.58rem;
  font-style: normal;
  line-height: 1;
}

.story-gepa-trace ul {
  display: grid;
  gap: 0.62rem;
  margin: 0;
  padding-left: 1.1rem;
}

.story-gepa-trace li {
  color: rgba(246, 248, 255, 0.82);
  font-size: 0.94rem;
  line-height: 1.14;
}

.story-gepa-trace__footer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.9rem;
  align-items: center;
  padding: 0.36rem 0.68rem;
  border: 1px solid rgba(246, 248, 255, 0.14);
  border-radius: 1rem;
  background: rgba(255, 255, 255, 0.055);
}

.story-gepa-trace__footer svg {
  width: 100%;
  height: 66px;
  overflow: visible;
}

.story-gepa-trace__footer line {
  stroke: rgba(246, 248, 255, 0.16);
  stroke-width: 2;
}

.story-gepa-trace__footer polyline {
  fill: none;
  stroke: rgba(142, 232, 255, 0.66);
  stroke-width: 3.5;
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
  min-width: 2.25rem;
  padding: 0.5rem 0.66rem;
  font-size: 0.74rem;
  text-transform: uppercase;
}

.story-gepa-trace__controls button.primary {
  min-width: 4.6rem;
  border-color: rgba(142, 232, 255, 0.34);
  background: rgba(142, 232, 255, 0.16);
  color: #8ee8ff;
}
</style>
