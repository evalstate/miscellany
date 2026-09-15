<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useSlideContext, useNav } from '@slidev/client';
import { useRoute } from 'vue-router';
import { useTimedStoryboard } from '../composables/useTimedStoryboard';
import provenance from '../data/legacy-message-ratio.provenance.json';

const props = defineProps<{ capture?: boolean }>();
const captureElapsed = ref<number | null>(null);

type Kind = 'tool' | 'initialize' | 'listing' | 'other';
type Phase = 'ready' | 'tool' | 'initialize' | 'listing' | 'other' | 'complete';
type Frame = { phase: Phase; revealed: number; flash?: Kind; duration: number };
const categories: { kind: Kind; title: string }[] = [
  { kind: 'tool', title: 'Tool call' }, { kind: 'initialize', title: 'Initialization' },
  { kind: 'listing', title: 'Listing' }, { kind: 'other', title: 'Other' },
];
const cells = categories.flatMap(({ kind, title }) => Array.from(
  { length: provenance.square_counts[kind] }, (_, index) => ({ kind, title: `${title} ${index + 1}` }),
));
const loop = ref(false);
const speed = ref(1);
const reduced = ref(false);
const showingAll = ref(false);
const chart = ref<HTMLElement>();
const { $page, $nav, $renderContext } = useSlideContext();
const route = useRoute();
const { isPrintMode } = useNav();
const printMode = computed(() => isPrintMode.value || route.path === '/print' || ['print', 'overview'].includes($renderContext.value));
// One immutable source timeline, shared by live frames and absolute capture seeks.
const events = (() => {
  const events: { at: number; kind: Kind; revealed: number }[] = [];
  let at = 450;
  for (const { kind } of categories) {
    if (kind === 'initialize') at = 1650;
    else if (kind !== 'tool') at += 440;
    for (let n = 0; n < provenance.square_counts[kind]; n++) {
      events.push({ at, kind, revealed: events.length + 1 });
      at += kind === 'other' ? 145 : 105;
    }
  }
  return events;
})();
const durationMs = events[events.length - 1].at + 1250;
const frames = computed<Frame[]>(() => [
    { phase: 'ready', revealed: 0, duration: 450 },
    ...events.map((event, index): Frame => ({
      phase: event.kind, flash: event.kind, revealed: event.revealed,
      duration: (events[index + 1]?.at ?? durationMs) - event.at,
    })),
    { phase: 'complete', revealed: cells.length, duration: loop.value ? 2200 : 0 },
  ]);
const { active, animationKey, isRunning, isPaused, isLooping, play, pause, resume, stop, setPlaybackRate } =
  useTimedStoryboard(frames, { endDelay: 0 });
const captureMode = computed(() => props.capture || route.query.capture === '1');
const capturing = computed(() => captureElapsed.value !== null);
const displayFrame = computed<Frame | undefined>(() => {
  if (captureElapsed.value === null) return active.value;
  if (captureElapsed.value >= durationMs) return { phase: 'complete', revealed: cells.length, duration: 0 };
  const event = events.findLast(event => event.at <= captureElapsed.value!);
  return event
    ? { phase: event.kind, flash: event.kind, revealed: event.revealed, duration: 0 }
    : { phase: 'ready', revealed: 0, duration: 450 };
});
const staticDisplay = computed(() => printMode.value || (!capturing.value && (showingAll.value || reduced.value)));
const revealed = computed(() => staticDisplay.value ? cells.length : displayFrame.value?.revealed ?? 0);
const counts = computed(() => Object.fromEntries(categories.map(({ kind }) => [kind, cells.slice(0, revealed.value).filter(cell => cell.kind === kind).length])) as Record<Kind, number>);
const otherCount = computed(() => counts.value.initialize + counts.value.listing + counts.value.other);
const mode = computed(() => staticDisplay.value ? 'complete' : capturing.value
  ? captureElapsed.value! >= durationMs ? 'complete' : captureElapsed.value === 0 ? 'idle' : 'playing'
  : isPaused.value ? 'paused' : isRunning.value ? 'playing' : active.value?.phase === 'complete' ? 'complete' : 'idle');
const state = computed(() => Object.freeze({
  mode: mode.value, phase: displayFrame.value?.phase ?? 'ready',
  elapsedMs: captureElapsed.value, durationMs, revealed: revealed.value,
  counts: Object.freeze({ ...counts.value }), otherShown: otherCount.value,
}));
const status = computed(() => mode.value === 'complete' ? 'Complete' : mode.value === 'paused' ? 'Paused' : mode.value === 'playing' ? `${categories.find(c => c.kind === displayFrame.value?.phase)?.title ?? 'Ready'}…` : 'Ready');
// Paused CSS animations sample source-clock age, including their natural resting
// state after the flash. Inline importance makes explicit capture independent of
// the OS reduced-motion preference; print and ordinary playback still respect it.
function captureStyle(at: number | undefined, animation: string) {
  if (!capturing.value || staticDisplay.value || at === undefined || captureElapsed.value! < at) return undefined;
  return `animation: ${animation} -${captureElapsed.value! - at}ms paused !important;`;
}
function legendEvent(kind: Kind) {
  return events.findLast(event => event.kind === kind && event.at <= (captureElapsed.value ?? -1));
}
function legendActive(kind: Kind) {
  if (staticDisplay.value) return false;
  return capturing.value ? !!legendEvent(kind) : active.value?.flash === kind && isRunning.value;
}
async function renderAt(ms: number) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) throw new TypeError('renderAt requires finite milliseconds');
  stop();
  wasPlaying = false;
  showingAll.value = false;
  captureElapsed.value = Math.max(0, Math.min(durationMs, ms));
  await nextTick();
}
defineExpose({ durationMs, renderAt, state });
let media: MediaQueryList | undefined;
let wasPlaying = false;
function cancelMotion() { chart.value?.getAnimations({ subtree: true }).forEach(animation => animation.cancel()); }
function replay() { stop(); cancelMotion(); captureElapsed.value = null; wasPlaying = false; showingAll.value = false; if (!reduced.value) play(loop.value); }
function showAll() { stop(); cancelMotion(); captureElapsed.value = null; wasPlaying = false; showingAll.value = true; }
function toggle() { if (isRunning.value) pause(); else if (isPaused.value) resume(); else replay(); }
function syncMotion() {
  if (capturing.value) return;
  chart.value?.getAnimations({ subtree: true }).forEach(animation => {
    if (animation.playState === 'finished' || animation.playState === 'idle') return;
    animation.updatePlaybackRate(speed.value);
    if (isPaused.value) animation.pause(); else animation.play();
  });
}
watch(isPaused, syncMotion);
watch(speed, () => { setPlaybackRate(speed.value); syncMotion(); });
watch(loop, value => {
  isLooping.value = value && (isRunning.value || isPaused.value);
  setPlaybackRate(speed.value); // Reschedule the optional final hold without restarting.
});
watch(revealed, async () => { await nextTick(); if (!reduced.value) syncMotion(); });
watch(() => $nav.value.currentSlideNo, page => {
  if (page !== $page.value) { stop(); cancelMotion(); captureElapsed.value = null; showingAll.value = false; wasPlaying = false; }
});
function visibilityChanged() {
  if (document.hidden && isRunning.value) { wasPlaying = true; pause(); }
  else if (!document.hidden && wasPlaying) { wasPlaying = false; resume(); }
}
function motionChanged() { reduced.value = media?.matches ?? false; if (reduced.value && !capturing.value) showAll(); }
onMounted(() => {
  media = window.matchMedia('(prefers-reduced-motion: reduce)'); motionChanged();
  media.addEventListener('change', motionChanged);
  document.addEventListener('visibilitychange', visibilityChanged);
});
onBeforeUnmount(() => { media?.removeEventListener('change', motionChanged); document.removeEventListener('visibilitychange', visibilityChanged); cancelMotion(); });
</script>

<template>
  <div ref="chart" class="message-ratio-chart" :data-state="mode" :data-phase="displayFrame?.phase ?? 'ready'" :data-other-shown="otherCount" :data-reduced-motion="reduced">
    <header class="message-ratio-heading"><slot name="heading" :other-count="otherCount" /></header>
    <div class="message-ratio-grid" role="img" :aria-label="`${counts.tool} tool call, ${counts.initialize} initialization, ${counts.listing} listing and ${counts.other} other message squares shown. Rounded aggregate ratio, not a session trace.`">
      <div v-for="(cell, index) in cells" :key="`${capturing ? 'capture' : 'live'}-${index}`" class="message-ratio-cell" :data-reveal-at="events[index].at" :style="captureStyle(events[index].at, 'ratio-cell-arrival 1100ms cubic-bezier(.16,.7,.25,1) both')" :class="{ 'is-visible': index < revealed, 'is-animated': index < revealed && !staticDisplay }" :data-kind="cell.kind" :title="cell.title" aria-hidden="true"></div>
    </div>
    <div class="message-ratio-legend" aria-label="Messages shown by category">
      <div v-for="category in categories" :key="category.kind" class="message-ratio-category" :data-kind="category.kind">
        <span :key="capturing ? 'capture' : animationKey" class="message-ratio-legend-content" :class="{ 'is-active': legendActive(category.kind) }" :style="captureStyle(legendEvent(category.kind)?.at, 'ratio-label-arrival 600ms ease-out')">
          <i :style="captureStyle(legendEvent(category.kind)?.at, 'ratio-swatch-arrival 650ms ease-out')"></i><span>{{ category.title }}</span><strong class="message-ratio-count">{{ counts[category.kind] }}</strong>
        </span>
      </div>
    </div>
    <footer class="message-ratio-footnote"><slot name="footnote" /></footer>
    <div v-if="!printMode && !captureMode" class="message-ratio-controls" @click.stop @keydown.enter.stop @keydown.space.stop>
      <button type="button" class="is-primary" @click="toggle">{{ isRunning ? 'Pause' : isPaused ? 'Resume' : mode === 'complete' ? 'Replay' : 'Play' }}</button>
      <button type="button" @click="replay">Replay</button>
      <button type="button" @click="showAll">Show all</button>
      <label>Speed <select v-model.number="speed" aria-label="Animation speed"><option :value="0.75">0.75×</option><option :value="1">1×</option><option :value="1.5">1.5×</option><option :value="2">2×</option></select></label>
      <label><input v-model="loop" type="checkbox" /> Loop</label>
      <span role="status">{{ status }}</span>
    </div>
  </div>
</template>
