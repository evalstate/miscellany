<script setup lang="ts">
// Verifier designs across three phases: setup → agent time → verifier.
// `phase` 0 shows the overview; 1–3 spotlight a phase (drive it from $clicks).
// Boxes are environments (containers); each box spans the phases it lives
// through, and each cell caption says what happens in that phase.
withDefaults(defineProps<{ phase?: number }>(), { phase: 0 });

type Cell = { text: string; artifact?: string };
type Box = { label: string; from: number; to: number; tone?: "paper" | "deep"; cells: Cell[] };
type Lane = { name: string; note: string; boxes: Box[] };

const phases = ["Setup", "Agent time", "Verifier"];

// Sources (checked 29 Sep 2026): harbor trial/single_step.py (shared: agent →
// verifier in the same env → stop; separate: agent → collect artifacts → stop
// agent env → verifier in its own container); terminal-bench-2-1 @7131e43
// (no environment_mode → shared); deep-swe @0b9fabb README + task.toml
// ([[verifier.collect]] writes model.patch, verifier network "no-network");
// terminal-bench v4.0.0 CONTRIBUTING.md + task template (environment_mode
// "separate" required; verifier reads artifacts, its own image, sidecars).
const lanes: Lane[] = [
  {
    name: "TB 2.1",
    note: "Shared verifier",
    boxes: [
      {
        label: "Task container",
        from: 1,
        to: 3,
        cells: [
          { text: "Build task image" },
          { text: "Agent works (internet on)" },
          { text: "Tests copied in, run where the agent worked" },
        ],
      },
    ],
  },
  {
    name: "DeepSWE 1.1",
    note: "Separate · patch in",
    boxes: [
      {
        label: "Agent container",
        from: 1,
        to: 2,
        cells: [
          { text: "Repo at base commit, no network" },
          { text: "Agent commits its work", artifact: "model.patch" },
        ],
      },
      {
        label: "Pristine container",
        from: 3,
        to: 3,
        tone: "deep",
        cells: [{ text: "Apply patch, run held-out tests" }],
      },
    ],
  },
  {
    name: "TB 4",
    note: "Separate · artifacts in",
    boxes: [
      {
        label: "Agent container",
        from: 1,
        to: 2,
        cells: [
          { text: "Build task image (+ sidecars)" },
          { text: "Agent works (internet on)", artifact: "artifacts" },
        ],
      },
      {
        label: "Verifier image",
        from: 3,
        to: 3,
        tone: "deep",
        cells: [{ text: "Own image reads artifacts and live sidecars" }],
      },
    ],
  },
];
</script>

<template>
  <div class="vd" :class="{ 'is-focused': phase > 0 }">
    <!-- Phase bands: a single flat tint marks the active phase. -->
    <div
      v-for="(p, i) in phases"
      :key="`band-${p}`"
      class="vd-band"
      :class="{ on: phase === i + 1 }"
      :style="{ gridColumn: i + 2 }"
    ></div>

    <div class="vd-corner"></div>
    <div
      v-for="(p, i) in phases"
      :key="`head-${p}`"
      class="vd-head"
      :class="{ on: phase === i + 1, dim: phase > 0 && phase !== i + 1 }"
      :style="{ gridColumn: i + 2 }"
    >
      <span class="vd-step">{{ i + 1 }}</span>{{ p }}
    </div>

    <template v-for="(lane, r) in lanes" :key="lane.name">
      <div class="vd-lane" :style="{ gridRow: r + 2 }">
        <strong>{{ lane.name }}</strong>
        <span>{{ lane.note }}</span>
      </div>
      <div
        v-for="box in lane.boxes"
        :key="box.label"
        class="vd-box"
        :class="`vd-box--${box.tone ?? 'paper'}`"
        :style="{ gridRow: r + 2, gridColumn: `${box.from + 1} / ${box.to + 2}`, '--cols': box.to - box.from + 1 }"
      >
        <span class="vd-box-label">{{ box.label }}</span>
        <div
          v-for="(cell, c) in box.cells"
          :key="c"
          class="vd-cell"
          :class="{ on: phase === box.from + c, dim: phase > 0 && phase !== box.from + c }"
        >
          {{ cell.text }}
          <code v-if="cell.artifact" class="vd-artifact">{{ cell.artifact }} ❯</code>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.vd {
  position: relative;
  display: grid;
  grid-template-columns: 150px repeat(3, 1fr);
  grid-template-rows: auto repeat(3, 1fr);
  column-gap: 0;
  row-gap: 10px;
  height: 100%;
  min-height: 300px;
}
.vd-band {
  grid-row: 1 / -1;
  margin: -6px 0;
  border-radius: var(--radius-lg);
  transition: background var(--dur-ui) var(--ease-ui);
}
.vd-band.on {
  background: var(--ivory-deep);
}
.vd-corner {
  grid-column: 1;
  grid-row: 1;
}
.vd-head {
  grid-row: 1;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px 8px;
  font: 700 13px/1.2 var(--font-read);
  letter-spacing: var(--track-label);
  text-transform: uppercase;
  border-bottom: var(--border);
  transition: opacity var(--dur-ui) var(--ease-ui);
}
.vd-head.on {
  color: var(--teal);
  border-bottom-color: var(--teal);
}
.vd-step {
  display: inline-grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border-radius: var(--radius-sm);
  background: var(--petrol);
  color: var(--ivory);
  font: 800 12px/1 var(--font-read);
  letter-spacing: 0;
}
.vd-head.on .vd-step {
  background: var(--teal);
}
.vd-lane {
  grid-column: 1;
  z-index: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding-right: 12px;
}
.vd-lane strong {
  font: 900 24px/1.05 var(--font-voice);
  font-variation-settings: var(--voice-settings);
  letter-spacing: var(--track-voice);
}
.vd-lane span {
  font-size: 14px;
  color: var(--text-muted);
}
.vd-box {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(var(--cols), 1fr);
  margin: 12px 8px 0;
  border: var(--border);
  border-radius: var(--radius-md);
}
.vd-box--paper {
  background: var(--paper);
}
.vd-box--deep {
  background: var(--paper);
  border-style: dashed;
}
.vd-box-label {
  position: absolute;
  top: -11px;
  left: 12px;
  padding: 2px 8px;
  background: var(--petrol);
  color: var(--ivory);
  border-radius: var(--radius-sm);
  font: 700 11px/1.3 var(--font-read);
  letter-spacing: var(--track-label);
  text-transform: uppercase;
}
.vd-cell {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
  padding: 14px 14px 10px;
  font-size: 15px;
  line-height: 1.3;
  font-weight: 600;
  transition: opacity var(--dur-ui) var(--ease-ui);
}
.vd-cell + .vd-cell {
  border-left: 2px solid var(--line);
}
.dim {
  opacity: 0.3;
}
.vd-artifact {
  align-self: flex-start;
  padding: 2px 8px;
  border: var(--border);
  border-radius: var(--radius-sm);
  background: var(--ivory);
  font: 500 13px/1.4 var(--font-code);
}
</style>
