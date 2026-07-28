<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

const CELL_HEIGHT = 170;
const REEL_HEIGHT = 520;
const CYCLE_HEIGHT = CELL_HEIGHT * 4;
const START_Y = (REEL_HEIGHT - CELL_HEIGHT) / 2;
const BOUNCE_DURATION = 440;
const AAIF_HOLD_DURATION = 5_000;
const MCP_HOLD_DURATION = 6_000;
const FINAL_PULSE_DURATION = 600;
const REVEAL_DURATION = 550;

const symbols = [
  { id: "aaif", src: "/brand/aaif-symbol-black.svg", label: "AAIF" },
  { id: "mcp", src: "/brand/mcp-symbol-black.svg", label: "MCP" },
  { id: "heart", src: "/brand/heart.svg", label: "loves" },
  { id: "huggy", src: "/brand/hugging-face.svg", label: "Hugging Face" },
] as const;

const repeatedSymbols = [...symbols, ...symbols, ...symbols];
const winnerIndices = { aaif: 0, mcp: 1, heart: 2, huggy: 3 } as const;

type Mode = "spinning" | "decelerating" | "bouncing" | "stopped";

type ReelState = {
  phase: number;
  velocity: number;
  mode: Mode;
  startTime: number;
  startPhase: number;
  travel: number;
  targetPhase: number;
  duration: number;
};

const phases = ref([0, 110, 235]);
const bounceOffsets = ref([0, 0, 0]);
const settled = ref([false, false, false]);
const heartPulse = ref(false);
const isRevealing = ref(false);
const cycle = ref(0);

const reels: ReelState[] = [
  makeReel(phases.value[0], 1680),
  makeReel(phases.value[1], 1740),
  makeReel(phases.value[2], 1800),
];

let frame = 0;
let previousTime = 0;
const timers: ReturnType<typeof setTimeout>[] = [];

function makeReel(phase: number, velocity: number): ReelState {
  return {
    phase,
    velocity,
    mode: "spinning",
    startTime: 0,
    startPhase: phase,
    travel: 0,
    targetPhase: 0,
    duration: 0,
  };
}

function schedule(callback: () => void, delay: number) {
  timers.push(setTimeout(callback, delay));
}

function clearTimers() {
  while (timers.length) clearTimeout(timers.pop());
}

function wrap(value: number) {
  return ((value % CYCLE_HEIGHT) + CYCLE_HEIGHT) % CYCLE_HEIGHT;
}

function setSettled(index: number, value: boolean) {
  const next = [...settled.value];
  next[index] = value;
  settled.value = next;
}

function startSpin(index: number, velocity: number) {
  const reel = reels[index];
  reel.mode = "spinning";
  reel.velocity = velocity;
  bounceOffsets.value[index] = 0;
  setSettled(index, false);
}

function startDeceleration(
  index: number,
  winner: keyof typeof winnerIndices,
  duration: number,
) {
  const reel = reels[index];
  const targetPhase = winnerIndices[winner] * CELL_HEIGHT;
  const delta = wrap(reel.phase - targetPhase);

  reel.mode = "decelerating";
  reel.startTime = performance.now();
  reel.startPhase = reel.phase;
  reel.targetPhase = targetPhase;
  reel.duration = duration;
  // Match the start of the ease-out to the incoming reel speed, then land
  // after a whole number of additional revolutions.
  const idealTravel = (reel.velocity * duration) / 2000;
  const extraCycles = Math.max(
    1,
    Math.round((idealTravel - delta) / CYCLE_HEIGHT),
  );
  reel.travel = delta + extraCycles * CYCLE_HEIGHT;
}

function bounceAt(progress: number) {
  if (progress < 0.28) {
    const p = progress / 0.28;
    return -20 * (1 - (1 - p) ** 2);
  }
  if (progress < 0.56) {
    const p = (progress - 0.28) / 0.28;
    return -20 + 32 * (1 - (1 - p) ** 2);
  }
  if (progress < 0.8) {
    const p = (progress - 0.56) / 0.24;
    return 12 - 18 * (1 - (1 - p) ** 2);
  }
  const p = (progress - 0.8) / 0.2;
  return -6 + 6 * (1 - (1 - p) ** 2);
}

function animate(time: number) {
  const elapsed = previousTime ? Math.min(48, time - previousTime) : 16;
  previousTime = time;

  reels.forEach((reel, index) => {
    if (reel.mode === "spinning") {
      reel.phase = wrap(reel.phase - (reel.velocity * elapsed) / 1000);
    } else if (reel.mode === "decelerating") {
      const progress = Math.min(1, (time - reel.startTime) / reel.duration);
      const eased = 1 - (1 - progress) ** 2;
      reel.phase = wrap(reel.startPhase - reel.travel * eased);

      if (progress >= 1) {
        reel.phase = reel.targetPhase;
        reel.mode = "bouncing";
        reel.startTime = time;
      }
    } else if (reel.mode === "bouncing") {
      const progress = Math.min(
        1,
        (time - reel.startTime) / BOUNCE_DURATION,
      );
      bounceOffsets.value[index] = bounceAt(progress);

      if (progress >= 1) {
        bounceOffsets.value[index] = 0;
        reel.mode = "stopped";
        setSettled(index, true);
      }
    }

    phases.value[index] = reel.phase;
  });

  // Trigger Vue updates for array element mutations.
  phases.value = [...phases.value];
  bounceOffsets.value = [...bounceOffsets.value];
  frame = requestAnimationFrame(animate);
}

function replay() {
  clearTimers();
  cancelAnimationFrame(frame);

  heartPulse.value = false;
  isRevealing.value = false;
  phases.value = [0, 110, 235];
  bounceOffsets.value = [0, 0, 0];
  settled.value = [false, false, false];
  cycle.value += 1;

  Object.assign(reels[0], makeReel(phases.value[0], 1680));
  Object.assign(reels[1], makeReel(phases.value[1], 1740));
  Object.assign(reels[2], makeReel(phases.value[2], 1800));

  previousTime = 0;
  frame = requestAnimationFrame(animate);

  const aaifDecelerationStart = 3450;
  const aaifDecelerationDuration = 1450;
  const aaifSettledAt =
    aaifDecelerationStart + aaifDecelerationDuration + BOUNCE_DURATION;
  const mcpSpinStart = aaifSettledAt + AAIF_HOLD_DURATION;
  const mcpDecelerationStart = mcpSpinStart + 1300;
  const mcpDecelerationDuration = 1350;
  const mcpSettledAt =
    mcpDecelerationStart + mcpDecelerationDuration + BOUNCE_DURATION;
  const finalPulseStart = mcpSettledAt + MCP_HOLD_DURATION;
  const revealStart = finalPulseStart + FINAL_PULSE_DURATION;
  const replayAt = revealStart + REVEAL_DURATION;

  schedule(() => startDeceleration(0, "huggy", 1450), 2550);
  schedule(() => startDeceleration(1, "heart", 1450), 3000);
  schedule(
    () =>
      startDeceleration(2, "aaif", aaifDecelerationDuration),
    aaifDecelerationStart,
  );

  schedule(() => startSpin(2, 1900), mcpSpinStart);
  schedule(
    () => startDeceleration(2, "mcp", mcpDecelerationDuration),
    mcpDecelerationStart,
  );

  schedule(() => {
    heartPulse.value = true;
  }, finalPulseStart);
  schedule(() => {
    heartPulse.value = false;
    isRevealing.value = true;
    settled.value = [false, false, false];
  }, revealStart);
  schedule(replay, replayAt);
}

const trackStyles = computed(() =>
  phases.value.map((phase, index) => ({
    transform: `translateY(${START_Y - CYCLE_HEIGHT - phase + bounceOffsets.value[index]}px)`,
  })),
);

onMounted(replay);
onBeforeUnmount(() => {
  clearTimers();
  cancelAnimationFrame(frame);
});
</script>

<template>
  <div
    :key="cycle"
    class="icon-machine"
    :class="{
      'is-heart-pulsing': heartPulse,
      'is-revealing': isRevealing,
    }"
    role="img"
    aria-label="Hugging Face loves AAIF, then MCP"
    title="Click to replay"
    @click.stop="replay"
  >
    <div
      v-for="(_, reelIndex) in reels"
      :key="reelIndex"
      class="icon-reel"
      :class="{ 'is-settled': settled[reelIndex] }"
    >
      <div class="icon-track" :style="trackStyles[reelIndex]">
        <div
          v-for="(symbol, symbolIndex) in repeatedSymbols"
          :key="`${reelIndex}-${symbolIndex}-${symbol.id}`"
          class="icon-cell"
          :class="[
            `symbol-${symbol.id}`,
            {
              'is-winner':
                settled[reelIndex] &&
                symbolIndex >= 4 &&
                symbolIndex < 8 &&
                Math.abs(
                  symbolIndex * CELL_HEIGHT +
                    START_Y -
                    CYCLE_HEIGHT -
                    phases[reelIndex] -
                    REEL_HEIGHT / 2 +
                    CELL_HEIGHT / 2,
                ) < 2,
            },
          ]"
        >
          <img :src="symbol.src" :alt="symbolIndex >= 4 ? symbol.label : ''" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.icon-machine {
  display: grid;
  width: min(600px, 88%);
  height: 520px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.3rem;
  align-items: center;
  margin-inline: auto;
  cursor: pointer;
}

.icon-reel {
  position: relative;
  height: 520px;
  overflow: hidden;
  mask-image: linear-gradient(
    transparent 0%,
    rgba(0, 0, 0, 0.76) 10%,
    #000 25%,
    #000 75%,
    rgba(0, 0, 0, 0.76) 90%,
    transparent 100%
  );
}

.icon-track {
  display: grid;
  grid-auto-rows: 170px;
  will-change: transform;
}

.icon-cell {
  display: grid;
  place-items: center;
  padding: 10px;
  opacity: 1;
  transition: opacity 320ms ease-out;
}

.icon-cell img {
  display: block;
  width: 132px;
  height: 132px;
  object-fit: contain;
}

.symbol-huggy img {
  width: 144px;
  height: 144px;
}

.symbol-aaif img,
.symbol-mcp img {
  width: 126px;
  height: 126px;
}

.symbol-heart img {
  width: 136px;
  height: 126px;
}

.icon-reel.is-settled .icon-cell:not(.is-winner) {
  opacity: 0;
}

.icon-machine.is-heart-pulsing
  .icon-reel:nth-child(2)
  .symbol-heart.is-winner
  img {
  animation: heart-loop-cue 600ms cubic-bezier(0.2, 0.78, 0.24, 1) both;
}

.icon-machine.is-revealing .icon-cell {
  transition-duration: 440ms;
}

@keyframes heart-loop-cue {
  0%,
  100% {
    filter: none;
    transform: scale(1);
  }
  24% {
    filter: drop-shadow(0 0 14px rgba(245, 55, 71, 0.24));
    transform: scale(1.12);
  }
  48% {
    filter: none;
    transform: scale(1);
  }
  72% {
    filter: drop-shadow(0 0 12px rgba(245, 55, 71, 0.2));
    transform: scale(1.08);
  }
}

@media (prefers-reduced-motion: reduce) {
  .icon-cell {
    transition-duration: 1ms;
  }
}
</style>
