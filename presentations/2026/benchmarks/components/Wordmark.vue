<script setup lang="ts">
// Stacked "fast-" / "agent" wordmark on the amber score burst, as on the
// homepage splash. The burst swings in, then the tagline types out behind a
// block cursor that blinks briefly and fades. Animations replay whenever the
// slide is shown (Slidev toggles display on inactive slides).
withDefaults(
  defineProps<{ tagline?: string; size?: number; dark?: boolean; sparkles?: boolean }>(),
  { size: 340, dark: false, sparkles: true },
);
</script>

<template>
  <div class="fa-wordmark" :class="{ 'is-dark': dark }" :style="{ '--wm-size': `${size}px` }">
    <div class="fa-wordmark-mark" role="img" aria-label="fast-agent">
      <span aria-hidden="true">fast-<br />agent</span>
    </div>
    <p v-if="tagline" class="fa-wordmark-tagline">
      <span class="fa-terminal-line"
        ><span class="fa-terminal-text">{{ tagline }}</span><i aria-hidden="true"></i
      ></span>
    </p>
    <template v-if="sparkles">
      <span class="fa-wordmark-spark fa-wordmark-spark--a" aria-hidden="true"></span>
      <span class="fa-wordmark-spark fa-wordmark-spark--b" aria-hidden="true"></span>
    </template>
  </div>
</template>

<style scoped>
.fa-wordmark {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  width: calc(var(--wm-size) * 1.3);
}
.fa-wordmark-mark {
  position: relative;
  isolation: isolate;
  display: grid;
  place-items: center;
  width: var(--wm-size);
  aspect-ratio: 1;
  font-family: var(--font-voice);
  font-variation-settings: var(--voice-settings);
  font-weight: 900;
  letter-spacing: var(--track-voice);
  font-size: calc(var(--wm-size) * 0.3);
  line-height: 0.86;
  text-align: center;
  color: var(--petrol);
  text-shadow: 3px 4px 0 var(--ivory);
}
.fa-wordmark-mark > span {
  display: block;
  transform: rotate(-7deg) scale(1.5);
}
.fa-wordmark-mark::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  background: var(--amber);
  mask: url("/fa/burst-score.svg") center / contain no-repeat;
  transform: rotate(28deg);
  animation: fa-score-swing 1.35s ease backwards;
}
.is-dark .fa-wordmark-mark {
  color: var(--ivory);
  text-shadow:
    -2px -2px 0 var(--petrol),
    2px -2px 0 var(--petrol),
    -2px 2px 0 var(--petrol),
    2px 2px 0 var(--petrol),
    5px 6px 0 var(--petrol);
}

.fa-wordmark-tagline {
  margin: calc(var(--wm-size) * 0.1) 0 0;
  font: 600 calc(var(--wm-size) * 0.078) / 1.15 var(--font-read);
  letter-spacing: -0.02em;
  white-space: nowrap;
}
.fa-terminal-line {
  display: block;
  position: relative;
  width: fit-content;
  margin-inline: auto;
  padding-right: 0.5em;
}
.fa-terminal-text {
  display: block;
  animation: fa-tagline-reveal 2.6s linear 1.5s both;
}
.fa-terminal-line i {
  position: absolute;
  top: 0.1em;
  width: 0.4em;
  height: 0.95em;
  background: var(--orange);
  animation:
    fa-cursor-travel 2.6s linear 1.5s both,
    fa-cursor-blink 0.6s step-end 4.1s 3,
    fa-cursor-fade 0.7s ease 5.9s forwards;
}

.fa-wordmark-spark {
  position: absolute;
  width: 30px;
  height: 30px;
  background: var(--teal);
  mask: url("/fa/sparkle-atomic.svg") center / contain no-repeat;
  opacity: 0.65;
  pointer-events: none;
}
.is-dark .fa-wordmark-spark {
  background: var(--amber);
}
.fa-wordmark-spark--a {
  left: 0;
  top: 10%;
}
.fa-wordmark-spark--b {
  right: 2%;
  bottom: 30%;
}

@keyframes fa-score-swing {
  0% { opacity: 0; transform: scale(0.4) rotate(-18deg); }
  48% { opacity: 1; transform: scale(1.05) rotate(-18deg); }
  82% { transform: rotate(35deg); }
  100% { transform: rotate(28deg); }
}
@keyframes fa-tagline-reveal {
  from { clip-path: inset(0 100% 0 0); }
  to { clip-path: inset(0); }
}
@keyframes fa-cursor-travel {
  0% { left: 0; opacity: 0; }
  1% { opacity: 1; }
  100% { left: calc(100% - 0.45em); opacity: 1; }
}
@keyframes fa-cursor-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
@keyframes fa-cursor-fade {
  to { opacity: 0; }
}
@media (prefers-reduced-motion: reduce) {
  .fa-wordmark-mark::before,
  .fa-terminal-text { animation: none; }
  .fa-terminal-line i { display: none; }
}
</style>
