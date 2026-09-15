<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch, type CSSProperties } from 'vue';
import { useSlideContext, useNav } from '@slidev/client';
import { useRoute } from 'vue-router';
import { useTimedStoryboard } from '../composables/useTimedStoryboard';
import CapabilityIcon from './CapabilityIcon.vue';

const props = withDefaults(defineProps<{ title?: string; capture?: boolean; loop?: boolean; deprecated?: boolean }>(), {
  title: 'Model Context Protocol', capture: false, loop: false, deprecated: false,
});
type Direction = 'right' | 'left';
type Capability = 'tools' | 'sampling' | 'resources' | 'elicitation' | 'prompts' | 'roots';
type Frame = { phase: 'message' | 'hold' | 'end'; direction: Direction; flash: Capability; at: number; duration: number; final: boolean };
const travelMs = 1275;
const holdMs = 5000;
const durationMs = 6 * (travelMs + holdMs);
const order: { flash: Capability; direction: Direction; final?: boolean }[] = [
  { flash: 'tools', direction: 'right' }, { flash: 'sampling', direction: 'left' },
  { flash: 'resources', direction: 'right' }, { flash: 'elicitation', direction: 'left' },
  { flash: 'prompts', direction: 'right' }, { flash: 'roots', direction: 'left', final: true },
];
const timeline: readonly Frame[] = Object.freeze(order.flatMap(({ flash, direction, final }, i): Frame[] => {
  const at = i * (travelMs + holdMs);
  return [
    { phase: 'message', direction, flash, at, duration: travelMs, final: false },
    { phase: 'hold', direction, flash, at: at + travelMs, duration: holdMs, final: !!final },
  ];
}).concat([{ phase: 'end', direction: 'left', flash: 'roots', at: durationMs, duration: 0, final: true }]));
const frames = computed(() => timeline);
const { active, animationKey, isRunning, isPaused, isLooping, play, pause, resume, stop } = useTimedStoryboard(frames, { endDelay: 0 });
const loopEnabled = ref(props.loop);
watch(() => props.loop, value => { loopEnabled.value = value; });
watch(loopEnabled, value => { isLooping.value = value && (isRunning.value || isPaused.value); });
const { $page, $nav, $renderContext } = useSlideContext();
const { isPrintMode } = useNav();
const route = useRoute();
const printMode = computed(() => isPrintMode.value || route.path === '/print' || ['print', 'overview'].includes($renderContext.value));
const reduced = ref(false);
const seekMs = ref<number | null>(null);
const showPoster = ref(false);
const liveMs = ref(0);
const root = ref<HTMLElement>();
const staticMode = computed(() => printMode.value || (seekMs.value === null && (reduced.value || showPoster.value)));
const frame = computed(() => {
  if (staticMode.value) return timeline[timeline.length - 1];
  if (seekMs.value !== null) return timeline.findLast(f => f.at <= seekMs.value!)!;
  return active.value ?? timeline[0];
});
const elapsedMs = computed(() => staticMode.value ? durationMs : seekMs.value ?? liveMs.value);
const age = computed(() => Math.max(0, Math.min(frame.value.duration, elapsedMs.value - frame.value.at)));
const idle = computed(() => !staticMode.value && seekMs.value === null && !active.value);
const state = computed(() => Object.freeze({
  phase: frame.value.phase, direction: frame.value.direction, flash: frame.value.flash,
  elapsedMs: elapsedMs.value, progress: elapsedMs.value / durationMs,
  frameElapsedMs: age.value, frameProgress: frame.value.duration ? age.value / frame.value.duration : 1,
  durationMs, receiver: frame.value.direction === 'right' ? 'server' : 'client',
  mode: idle.value ? 'ready' : staticMode.value || frame.value.phase === 'end' ? 'end' : seekMs.value !== null ? 'capture' : isPaused.value ? 'paused' : 'playing',
}));
async function renderAt(ms: number) {
  if (!Number.isFinite(ms)) throw new TypeError('A finite number is required');
  stop();
  seekMs.value = Math.max(0, Math.min(durationMs, ms));
  await nextTick();
  // Flush styles so callers can immediately rasterize any arbitrary, backwards seek.
  root.value?.getBoundingClientRect();
  return state.value;
}
defineExpose({ renderAt, durationMs, state });
let raf = 0;
let lastTick = 0;
function tick(now: number) {
  liveMs.value = Math.min(durationMs, liveMs.value + now - lastTick);
  lastTick = now;
  if (isRunning.value) raf = requestAnimationFrame(tick);
}
watch(isRunning, running => {
  cancelAnimationFrame(raf);
  if (running) { lastTick = performance.now(); raf = requestAnimationFrame(tick); }
}, { flush: 'sync' });
watch(active, f => { if (f) liveMs.value = f.at; });
function replay() {
  stop(); seekMs.value = null; showPoster.value = false; liveMs.value = 0;
  if (!reduced.value && !printMode.value) play(loopEnabled.value);
}
function toggle() { if (isRunning.value) pause(); else if (isPaused.value) resume(); else replay(); }
function showEnd() { stop(); seekMs.value = null; showPoster.value = true; }
function visibilityChanged() { if (document.hidden) pause(); }
watch(() => $nav.value.currentSlideNo, page => {
  if (page !== $page.value) { stop(); seekMs.value = null; showPoster.value = false; liveMs.value = 0; }
});
let media: MediaQueryList | undefined;
function motionChanged() { reduced.value = media?.matches ?? false; if (reduced.value && seekMs.value === null) showEnd(); }
onMounted(() => {
  media = matchMedia('(prefers-reduced-motion: reduce)'); motionChanged();
  media.addEventListener('change', motionChanged);
  document.addEventListener('visibilitychange', visibilityChanged);
});
onBeforeUnmount(() => {
  cancelAnimationFrame(raf);
  media?.removeEventListener('change', motionChanged);
  document.removeEventListener('visibilitychange', visibilityChanged);
});
function animationStyle(kind: 'pulse' | 'highlight'): CSSProperties {
  const duration = kind === 'pulse' ? travelMs : holdMs;
  const name = kind === 'pulse' ? `lpv-travel-${frame.value.direction}` : frame.value.final ? 'lpv-final-highlight' : 'lpv-highlight';
  if (staticMode.value || frame.value.phase === 'end') return { opacity: 1 };
  if (seekMs.value !== null) return { animation: `${name} ${duration}ms linear -${age.value}ms 1 both paused !important` };
  return { animation: `${name} ${duration}ms linear 1 both`, animationPlayState: isPaused.value ? 'paused' : 'running' };
}
const purple = '#6430d8';
const slate = '#475569';
const color = computed(() => frame.value.direction === 'right' ? purple : slate);
const highlighting = computed(() => !props.deprecated && !idle.value && frame.value.phase !== 'message');
const cards: { id: Capability; label: string; x: number; y: number }[] = [
  { id: 'roots', label: 'Roots', x: 72, y: 198 },
  { id: 'sampling', label: 'Sampling', x: 72, y: 410 },
  { id: 'elicitation', label: 'Elicitation', x: 72, y: 622 },
  { id: 'tools', label: 'Tools', x: 1244, y: 198 },
  { id: 'resources', label: 'Resources', x: 1244, y: 410 },
  { id: 'prompts', label: 'Prompts', x: 1244, y: 622 },
];
</script>

<template>
  <div ref="root" class="lpv-root" :data-phase="deprecated ? 'static' : state.phase" :data-direction="deprecated ? 'both' : state.direction" :data-flash="deprecated ? 'none' : state.flash" :data-mode="deprecated ? 'static' : state.mode" :data-capture="seekMs !== null">
    <svg class="lpv-svg" viewBox="0 0 1600 900" xmlns="http://www.w3.org/2000/svg" role="img" :aria-label="deprecated ? `${title}. Roots and Sampling crossed out; one double-headed communication arrow.` : `${title}. Client and Server capabilities. ${state.mode === 'ready' ? 'Ready to play.' : `${state.flash}, ${state.direction === 'right' ? 'Client to Server' : 'Server to Client'}.`}`">
      <rect width="1600" height="900" fill="#ffffff" />
      <text x="80" y="77" class="lpv-title" fill="#151515">{{ title }}</text>
      <rect x="45" y="36" width="9" height="48" rx="2" :fill="purple" />
      <g v-for="card in cards" :key="card.id" :data-capability="card.id" :data-deprecated="deprecated && ['roots', 'sampling'].includes(card.id)">
        <line :x1="card.x < 800 ? 356 : 1215" :x2="card.x < 800 ? 385 : 1244" :y1="card.y + 86.5" :y2="card.y + 86.5" stroke="#cbd5e1" stroke-width="3" />
        <rect :x="card.x" :y="card.y" width="284" height="173" rx="20" fill="#fff" stroke="#cbd5e1" stroke-width="2" />
        <rect v-if="highlighting && frame.flash === card.id" :key="`card-${animationKey}`" class="lpv-highlight" :x="card.x" :y="card.y" width="284" height="173" rx="20" fill="#ede9fe" :stroke="color" stroke-width="5" :style="animationStyle('highlight')" />
        <CapabilityIcon :name="card.id" :size="52" :x="card.x + 116" :y="card.y + 33" width="52" height="52" style="width: 52px; height: 52px; color: #475569" />
        <text :x="card.x + 142" :y="card.y + 133" text-anchor="middle" class="lpv-card-label" fill="#334155">{{ card.label }}</text>
        <g v-if="deprecated && ['roots', 'sampling'].includes(card.id)" data-role="deprecated-cross" aria-label="Deprecated">
          <title>{{ card.label }} — deprecated</title>
          <path :d="`M${card.x + 30} ${card.y + 22}L${card.x + 254} ${card.y + 151}M${card.x + 254} ${card.y + 22}L${card.x + 30} ${card.y + 151}`" fill="none" stroke="#dc2626" stroke-width="7" stroke-linecap="round" />
        </g>
      </g>
      <g v-for="actor in [{ id: 'client', x: 385, label: 'Client', color: purple }, { id: 'server', x: 1002, label: 'Server', color: slate }]" :key="actor.id" :data-actor="actor.id">
        <rect :x="actor.x" y="198" width="213" height="597" rx="20" :fill="actor.id === 'client' ? '#f5f3ff' : '#f8fafc'" :stroke="actor.color" stroke-width="3" />
        <rect v-if="highlighting && state.receiver === actor.id" :key="`actor-${animationKey}`" class="lpv-highlight" :x="actor.x" y="198" width="213" height="597" rx="20" fill="#ede9fe" :stroke="color" stroke-width="6" :style="animationStyle('highlight')" />
        <text :x="actor.x + 106.5" y="514" text-anchor="middle" class="lpv-actor-label" :fill="actor.color">{{ actor.label }}</text>
      </g>
      <!-- Each complete arrow is one filled polygon: no marker/shaft seam. -->
      <path v-if="deprecated" class="lpv-arrow" data-arrow="both" d="M627 497L653 479V492H947V479L973 497L947 515V502H653V515Z" :fill="purple" />
      <path v-if="!deprecated" class="lpv-arrow" data-arrow="right" d="M627 443H947V430L973 448L947 466V453H627Z" :fill="purple" />
      <path v-if="!deprecated" class="lpv-arrow" data-arrow="left" d="M973 541H653V528L627 546L653 564V551H973Z" :fill="slate" />
      <circle v-if="!deprecated && !idle && frame.phase === 'message'" :key="`pulse-${seekMs !== null ? 'capture' : animationKey}`" class="lpv-pulse" data-role="pulse" cx="0" cy="0" r="12" :fill="color" stroke="#ffffff" stroke-width="3" :style="animationStyle('pulse')" />
    </svg>
    <div v-if="!deprecated && !capture && !printMode" class="lpv-controls" @click.stop @keydown.enter.stop @keydown.space.stop>
      <button type="button" :disabled="reduced" @click="toggle">{{ isRunning ? 'Pause' : isPaused ? 'Resume' : 'Play' }}</button>
      <button type="button" :disabled="reduced" @click="replay">Replay</button>
      <button type="button" @click="showEnd">Show end</button>
      <label><input v-model="loopEnabled" type="checkbox" :disabled="reduced" /> Loop</label>
      <span role="status">{{ reduced ? 'Reduced motion · static' : state.mode === 'ready' ? 'Ready' : state.mode === 'end' ? 'Complete' : isPaused ? 'Paused' : frame.flash }}</span>
    </div>
  </div>
</template>
