<script setup lang="ts">
// One Harbor trial, animated: agent setup, the time-limited task, then the
// verifier, which emits a reward of 0 or 1. A coordinator (the Harbor job)
// runs many of these and records each reward.
//
// mode="shared":   one container for all three phases (TB 2.1, TB-lite).
// mode="separate": the agent's container is stopped and what it produced is
//                  handed to a second container for verification (DeepSWE, TB 4).
// `network` draws the DeepSWE isolation: a no-network wall round the agent's
// container, the internet stopped at it, and only the model endpoint let
// through (allowlisted by the runner, Pier).
// `total` is the number of trials in a full run, for the coordinator's counter.
//
// Flow checked against harbor src/harbor/trial/single_step.py: run agent,
// collect artifacts, (separate: stop the agent environment), run verifier,
// (shared: stop the environment). Phase widths are illustrative, not measured.
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useIsSlideActive } from "@slidev/client";

const props = withDefaults(
  defineProps<{ mode?: "shared" | "separate"; artifact?: string; verifierLabel?: string; verifierNote?: string; network?: boolean; total?: number }>(),
  { mode: "shared", artifact: "artifacts", verifierLabel: "Verifier container", verifierNote: "", network: false, total: 445 },
);

const separate = computed(() => props.mode === "separate");
// seconds: [start, end] of each beat within one trial
const T = computed(() => {
  const setup = [0.2, 1.0];
  const task = [1.0, 3.2];
  const hand = separate.value ? [3.2, 4.0] : [3.2, 3.2];
  const verify = [hand[1], hand[1] + 1.2];
  const reward = verify[1] + 0.1;
  return { setup, task, hand, verify, reward, total: reward + 1.5 };
});

const QUEUE = 30;
const RESULTS = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0]; // illustrative
const active = useIsSlideActive();
const t = ref(999); // past the end: the still frame shows a finished trial
const trial = ref(4);
const count = computed(() => trial.value + (done.value ? 1 : 0));
let raf = 0;
let last = 0;

const span = (range: number[]) => Math.min(1, Math.max(0, (t.value - range[0]) / Math.max(0.001, range[1] - range[0])));
const setup = computed(() => span(T.value.setup));
const task = computed(() => span(T.value.task));
const hand = computed(() => span(T.value.hand));
const verify = computed(() => span(T.value.verify));
const done = computed(() => t.value >= T.value.reward);
const reward = computed(() => RESULTS[trial.value % QUEUE]);
const agentStopped = computed(() => separate.value && hand.value >= 1);

function tick(now: number) {
  raf = 0;
  if (!active.value) return;
  t.value += Math.min(0.1, (now - last) / 1000);
  last = now;
  if (t.value >= T.value.total) {
    t.value = 0;
    trial.value = (trial.value + 1) % QUEUE;
  }
  raf = requestAnimationFrame(tick);
}
function run() {
  if (raf) return;
  if (active.value) {
    t.value = 0;
    last = performance.now();
    raf = requestAnimationFrame(tick);
  } else {
    t.value = 999;
  }
}
onMounted(run);
watch(active, run);
onBeforeUnmount(() => cancelAnimationFrame(raf));
</script>

<template>
  <div class="hb" :class="{ 'hb--separate': separate }">
    <div class="hb-coord">
      <div class="hb-coord-copy">
        <strong>Coordinator</strong>
        <span>queues the trials, starts containers, enforces limits, records rewards</span>
      </div>
      <div class="hb-tally"><b>{{ count }}</b> of {{ total }} trials</div>
      <div class="hb-queue" aria-hidden="true">
        <i v-for="n in QUEUE" :key="n" :class="{ won: n - 1 < trial && RESULTS[n - 1], lost: n - 1 < trial && !RESULTS[n - 1], now: n - 1 === trial, 'now-won': n - 1 === trial && done && reward, 'now-lost': n - 1 === trial && done && !reward }" />
      </div>
    </div>

    <div v-if="network" class="hb-net">
      <span class="hb-net-host hb-net-host--ok">Model endpoint</span>
      <span class="hb-net-host hb-net-host--no">Internet</span>
      <span class="hb-net-note">No network. Only the model endpoint gets through the wall.</span>
    </div>

    <div class="hb-stage">
      <div class="hb-boxes">
        <div class="hb-box hb-box--agent" :class="{ stopped: agentStopped, walled: network }">
          <span class="hb-box-label">{{ separate ? "Agent container" : "Task container" }}<em v-if="agentStopped"> · stopped</em></span>
          <div class="hb-seg hb-seg--setup">
            <b :style="{ width: `${setup * 100}%` }"><span class="hb-lab"><strong>Agent setup</strong><small>Start the image and the harness.</small></span></b>
            <span class="hb-lab"><strong>Agent setup</strong><small>Start the image and the harness.</small></span>
          </div>
          <div class="hb-seg hb-seg--task">
            <b :style="{ width: `${task * 100}%` }"><span class="hb-lab"><strong>Perform task</strong><small>The agent works against a clock. Out of time means stop.</small></span></b>
            <span class="hb-lab"><strong>Perform task</strong><small>The agent works against a clock. Out of time means stop.</small></span>
            <svg class="hb-clock" viewBox="0 0 48 48" aria-label="time limited">
              <circle cx="24" cy="24" r="20" />
              <path d="M24 24 L24 9" :transform="`rotate(${task * 360} 24 24)`" />
              <path d="M24 4 v5 M44 24 h-5 M24 44 v-5 M4 24 h5" class="hb-clock-ticks" />
            </svg>
          </div>
          <div v-if="!separate" class="hb-seg hb-seg--verify">
            <b :style="{ width: `${verify * 100}%` }"><span class="hb-lab"><strong>Verifier</strong><small>Tests are copied in and run where the agent just worked.</small></span></b>
            <span class="hb-lab"><strong>Verifier</strong><small>Tests are copied in and run where the agent just worked.</small></span>
          </div>
        </div>

        <div v-if="separate" class="hb-hand" aria-hidden="true">
          <code :style="{ left: `${hand * 100}%`, opacity: hand > 0 && verify < 0.25 ? 1 : 0 }">{{ artifact }}</code>
        </div>

        <div v-if="separate" class="hb-box hb-box--verifier" :class="{ waiting: hand < 1 }">
          <span class="hb-box-label">{{ verifierLabel }}</span>
          <div class="hb-seg hb-seg--verify">
            <b :style="{ width: `${verify * 100}%` }"><span class="hb-lab"><strong>Verifier</strong><small>{{ verifierNote || "A fresh container the agent never touched." }}</small></span></b>
            <span class="hb-lab"><strong>Verifier</strong><small>{{ verifierNote || "A fresh container the agent never touched." }}</small></span>
          </div>
        </div>
      </div>

      <span class="hb-out" aria-hidden="true">❯</span>
      <div class="hb-reward" :class="{ on: done }">
        <span>reward</span>
        <strong>{{ done ? reward : "?" }}</strong>
      </div>
    </div>
  </div>
</template>

<style scoped>
.hb {
  display: flex;
  flex-direction: column;
  gap: 18px;
  font-family: var(--font-read);
}
.hb-coord {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 14px 22px;
  border: var(--border);
  border-radius: var(--radius-lg);
  background: var(--paper);
}
.hb-coord-copy { display: flex; align-items: center; gap: 16px; min-width: 0; }
.hb-coord strong { font: 900 20px/1 var(--font-voice); letter-spacing: var(--track-voice); }
.hb-coord span { font-size: 14px; line-height: 1.3; color: var(--text-muted); }
.hb-tally { flex: none; font-size: 14px; color: var(--text-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.hb-tally b { font: 900 22px/1 var(--font-voice); color: var(--petrol); margin-right: 2px; }
.hb-queue { display: grid; grid-template-columns: repeat(15, 14px); gap: 4px; flex: none; }
.hb-queue i {
  width: 14px;
  height: 14px;
  border: 2px solid var(--line);
  border-radius: 3px;
}
.hb-queue i.now-won, .hb-queue i.now-lost { animation: hb-pop var(--dur-pop) var(--ease-pop) 1; }
.hb-queue i.won, .hb-queue i.now-won { background: var(--petrol); border-color: var(--petrol); }
.hb-queue i.lost, .hb-queue i.now-lost { border-color: var(--petrol); }
.hb-queue i.now { border-color: var(--orange); }

.hb-net {
  position: relative;
  display: flex;
  align-items: center;
  gap: 14px;
  margin: -2px 0 -4px 22%;
  padding-bottom: 20px;
  font-size: 14px;
  color: var(--text-muted);
}
.hb-net-host {
  position: relative;
  padding: 4px 10px;
  border: var(--border);
  border-radius: var(--radius-sm);
  background: var(--paper);
  font: 500 14px/1.2 var(--font-code);
  color: var(--petrol);
}
/* connectors: the model endpoint's line runs down through the wall into the
   container; the internet's stops short at a bar */
.hb-net-host::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 100%;
  height: 34px;
  border-left: 2px solid var(--petrol);
}
.hb-net-host--no { color: var(--text-muted); text-decoration: line-through; text-decoration-color: var(--orange); text-decoration-thickness: 2px; }
.hb-net-host--no::after { height: 14px; border-left-color: var(--orange); }
.hb-net-host--no::before {
  content: "";
  position: absolute;
  left: calc(50% - 9px);
  top: calc(100% + 12px);
  width: 20px;
  border-top: 4px solid var(--orange);
}
.hb-box.walled { outline: 4px solid var(--orange); outline-offset: 8px; }
.hb-box.walled::after {
  content: "No network";
  position: absolute;
  right: 18px;
  bottom: -24px;
  padding: 4px 10px;
  border-radius: var(--radius-sm);
  background: var(--orange);
  color: var(--ivory);
  font: 700 13px/1 var(--font-read);
  letter-spacing: 0.02em;
}

.hb-stage { display: flex; align-items: stretch; gap: 12px; }
.hb-boxes { flex: 1; display: flex; align-items: stretch; min-width: 0; }
.hb-box {
  position: relative;
  display: flex;
  gap: 6px;
  flex: 1;
  height: 270px;
  padding: 28px 8px 8px;
  border: var(--border);
  border-radius: var(--radius-lg);
  background: var(--paper);
  transition: opacity var(--dur-ui) var(--ease-ui);
}
.hb-box--agent { flex: 7; }
.hb-box--verifier { flex: 3; border-style: dashed; }
.hb-box.stopped { opacity: 0.45; }
.hb-box.waiting { opacity: 0.5; }
.hb-box-label {
  position: absolute;
  top: 7px;
  left: 12px;
  font: 600 13px/1 var(--font-read);
  color: var(--text-muted);
}
.hb-box-label em { font-style: normal; }

.hb-seg {
  position: relative;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 20px 16px;
  border-radius: var(--radius-sm);
  background: var(--ivory-deep);
  overflow: hidden;
  min-width: 0;
}
.hb-seg { container-type: inline-size; }
.hb-seg b { position: absolute; inset: 0 auto 0 0; z-index: 1; overflow: hidden; }
.hb-lab {
  position: relative;
  width: calc(100cqw - 32px);
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
  color: var(--petrol);
}
.hb-lab strong {
  font: 900 30px/1.02 var(--font-voice);
  letter-spacing: var(--track-voice);
}
.hb-lab small { font: 500 15px/1.3 var(--font-read); max-width: 30ch; }
/* The same label again inside the fill, in ivory. The fill clips it, so the
   text changes colour exactly where the colour behind it changes. */
.hb-seg b .hb-lab {
  position: absolute;
  left: 16px;
  top: 20px;
  width: calc(100cqw - 32px);
  color: var(--ivory);
}
.hb-seg--setup { flex: 2.4; }
.hb-seg--task { flex: 5; }
.hb-seg--verify { flex: 3; }
.hb-seg--setup b { background: var(--teal); }
.hb-seg--task b { background: var(--petrol); }
.hb-seg--verify b { background: var(--orange); }

.hb-clock { position: absolute; z-index: 2; bottom: 14px; right: 14px; width: 56px; height: 56px; }
.hb-clock circle { fill: var(--ivory); stroke: var(--petrol); stroke-width: 3; }
.hb-clock path { stroke: var(--orange); stroke-width: 4; stroke-linecap: round; fill: none; }
.hb-clock .hb-clock-ticks { stroke: var(--petrol); stroke-width: 2; }

.hb-hand { position: relative; z-index: 2; width: 124px; flex: none; }
.hb-hand::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; border-top: var(--border-dash); }
.hb-hand code {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  padding: 4px 8px;
  border: var(--border);
  border-radius: var(--radius-sm);
  background: var(--ivory);
  font: 500 13px/1 var(--font-code);
  white-space: nowrap;
  transition: opacity var(--dur-ui) var(--ease-ui);
}

.hb-out { align-self: center; font-size: 26px; color: var(--petrol); }
.hb-reward {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 104px;
  flex: none;
  border: var(--border);
  border-radius: var(--radius-lg);
  background: var(--paper);
}
.hb-reward span { font: 600 13px/1 var(--font-read); color: var(--text-muted); }
.hb-reward strong { font: 900 88px/1 var(--font-voice); color: var(--line-strong); opacity: 0.25; }
.hb-reward.on strong { opacity: 1; animation: hb-pop var(--dur-pop) var(--ease-pop) 1; }
@keyframes hb-pop {
  from { transform: scale(0.55); }
  to { transform: scale(1); }
}

.hb--separate .hb-box { height: 220px; }
.hb--separate .hb-stage { padding-bottom: 26px; }
.hb--separate .hb-lab strong { font-size: 26px; }
.hb--separate .hb-lab small { font-size: 14px; }
/* The shared slide explains setup; here the narrow setup block keeps its title only. */
.hb--separate .hb-seg--setup small { display: none; }
.hb--separate .hb-seg { padding-top: 16px; }
.hb--separate .hb-seg b .hb-lab { top: 16px; }
</style>
