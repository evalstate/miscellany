<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

const CELL_HEIGHT = 170;
const REEL_HEIGHT = 520;
const CYCLE_HEIGHT = CELL_HEIGHT * 4;
const START_Y = (REEL_HEIGHT - CELL_HEIGHT) / 2;

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
const cycle = ref(0);

const reels: ReelState[] = [
  makeReel(phases.value[0], 1520),
  makeReel(phases.value[1], 1580),
  makeReel(phases.value[2], 1640),
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
  const delta = wrap(targetPhase - reel.phase);

  reel.mode = "decelerating";
  reel.startTime = performance.now();
  reel.startPhase = reel.phase;
  reel.targetPhase = targetPhase;
  reel.duration = duration;
  // One complete extra revolution keeps the reel moving forward while it slows.
  reel.travel = delta + CYCLE_HEIGHT;
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
      reel.phase = wrap(reel.phase + (reel.velocity * elapsed) / 1000);
    } else if (reel.mode === "decelerating") {
      const progress = Math.min(1, (time - reel.startTime) / reel.duration);
      const eased = 1 - (1 - progress) ** 2;
      reel.phase = wrap(reel.startPhase + reel.travel * eased);

      if (progress >= 1) {
        reel.phase = reel.targetPhase;
        reel.mode = "bouncing";
        reel.startTime = time;
      }
    } else if (reel.mode === "bouncing") {
      const progress = Math.min(1, (time - reel.startTime) / 440);
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

  phases.value = [0, 110, 235];
  bounceOffsets.value = [0, 0, 0];
  settled.value = [false, false, false];
  cycle.value += 1;

  Object.assign(reels[0], makeReel(phases.value[0], 1520));
  Object.assign(reels[1], makeReel(phases.value[1], 1580));
  Object.assign(reels[2], makeReel(phases.value[2], 1640));

  previousTime = 0;
  frame = requestAnimationFrame(animate);

  schedule(() => startDeceleration(0, "huggy", 1450), 2550);
  schedule(() => startDeceleration(1, "heart", 1450), 3000);
  schedule(() => startDeceleration(2, "aaif", 1450), 3450);

  schedule(() => startSpin(2, 1780), 6600);
  schedule(() => startDeceleration(2, "mcp", 1350), 7900);
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
  width: min(1080px, 96%);
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

@media (prefers-reduced-motion: reduce) {
  .icon-cell {
    transition-duration: 1ms;
  }
}
</style>
