<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useSlideContext, useNav } from '@slidev/client';
import { useRoute } from 'vue-router';
import { useTimedStoryboard } from '../composables/useTimedStoryboard';
import provenance from '../data/legacy-message-ratio.provenance.json';

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
const frames = computed<Frame[]>(() => {
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
  const finishAt = events[events.length - 1].at + 1250;
  return [
    { phase: 'ready', revealed: 0, duration: 450 },
    ...events.map((event, index): Frame => ({
      phase: event.kind, flash: event.kind, revealed: event.revealed,
      duration: (events[index + 1]?.at ?? finishAt) - event.at,
    })),
    { phase: 'complete', revealed: cells.length, duration: loop.value ? 2200 : 0 },
  ];
});
const { active, animationKey, isRunning, isPaused, isLooping, play, pause, resume, stop, setPlaybackRate } =
  useTimedStoryboard(frames, { endDelay: 0 });
const revealed = computed(() => showingAll.value || reduced.value || printMode.value ? cells.length : active.value?.revealed ?? 0);
const counts = computed(() => Object.fromEntries(categories.map(({ kind }) => [kind, cells.slice(0, revealed.value).filter(cell => cell.kind === kind).length])) as Record<Kind, number>);
const otherCount = computed(() => counts.value.initialize + counts.value.listing + counts.value.other);
const state = computed(() => showingAll.value || reduced.value || printMode.value ? 'complete' : isPaused.value ? 'paused' : isRunning.value ? 'playing' : active.value?.phase === 'complete' ? 'complete' : 'idle');
const status = computed(() => state.value === 'complete' ? 'Complete' : state.value === 'paused' ? 'Paused' : isRunning.value ? `${categories.find(c => c.kind === active.value?.phase)?.title ?? 'Ready'}…` : 'Ready');
let media: MediaQueryList | undefined;
let wasPlaying = false;
function cancelMotion() { chart.value?.getAnimations({ subtree: true }).forEach(animation => animation.cancel()); }
function replay() { stop(); cancelMotion(); showingAll.value = false; if (!reduced.value) play(loop.value); }
function showAll() { stop(); cancelMotion(); showingAll.value = true; }
function toggle() { if (isRunning.value) pause(); else if (isPaused.value) resume(); else replay(); }
function syncMotion() {
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
  if (page !== $page.value) { stop(); cancelMotion(); showingAll.value = false; wasPlaying = false; }
});
function visibilityChanged() {
  if (document.hidden && isRunning.value) { wasPlaying = true; pause(); }
  else if (!document.hidden && wasPlaying) { wasPlaying = false; resume(); }
}
function motionChanged() { reduced.value = media?.matches ?? false; if (reduced.value) showAll(); }
onMounted(() => {
  media = window.matchMedia('(prefers-reduced-motion: reduce)'); motionChanged();
  media.addEventListener('change', motionChanged);
  document.addEventListener('visibilitychange', visibilityChanged);
});
onBeforeUnmount(() => { media?.removeEventListener('change', motionChanged); document.removeEventListener('visibilitychange', visibilityChanged); cancelMotion(); });
</script>

<template>
  <div ref="chart" class="message-ratio-chart" :data-state="state" :data-phase="active?.phase ?? 'ready'" :data-other-shown="otherCount" :data-reduced-motion="reduced">
    <header class="message-ratio-heading"><slot name="heading" :other-count="otherCount" /></header>
    <div class="message-ratio-grid" role="img" :aria-label="`${counts.tool} tool call, ${counts.initialize} initialization, ${counts.listing} listing and ${counts.other} other message squares shown. Rounded aggregate ratio, not a session trace.`">
      <div v-for="(cell, index) in cells" :key="index" class="message-ratio-cell" :class="{ 'is-visible': index < revealed, 'is-animated': index < revealed && !showingAll && !reduced && !printMode }" :data-kind="cell.kind" :title="cell.title" aria-hidden="true"></div>
    </div>
    <div class="message-ratio-legend" aria-label="Messages shown by category">
      <div v-for="category in categories" :key="category.kind" class="message-ratio-category" :data-kind="category.kind">
        <span :key="animationKey" class="message-ratio-legend-content" :class="{ 'is-active': active?.flash === category.kind && isRunning && !showingAll && !reduced }">
          <i></i><span>{{ category.title }}</span><strong class="message-ratio-count">{{ counts[category.kind] }}</strong>
        </span>
      </div>
    </div>
    <footer class="message-ratio-footnote"><slot name="footnote" /></footer>
    <div v-if="!printMode" class="message-ratio-controls" @click.stop @keydown.enter.stop @keydown.space.stop>
      <button type="button" class="is-primary" @click="toggle">{{ isRunning ? 'Pause' : isPaused ? 'Resume' : state === 'complete' ? 'Replay' : 'Play' }}</button>
      <button type="button" @click="replay">Replay</button>
      <button type="button" @click="showAll">Show all</button>
      <label>Speed <select v-model.number="speed" aria-label="Animation speed"><option :value="0.75">0.75×</option><option :value="1">1×</option><option :value="1.5">1.5×</option><option :value="2">2×</option></select></label>
      <label><input v-model="loop" type="checkbox" /> Loop</label>
      <span role="status">{{ status }}</span>
    </div>
  </div>
</template>
