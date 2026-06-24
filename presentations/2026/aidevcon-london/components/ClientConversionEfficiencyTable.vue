<script setup lang="ts">
import csv from "../data-viz/client_session_conversion_2026-04-07_to_2026-06-01.csv?raw";

type Row = {
  client_name: string;
  sessions: number;
  conversion_rate_pct: number;
};

const MIN_SESSIONS = 1_000;

function parseCsv(text: string): Row[] {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  const idx = (name: string) => headers.indexOf(name);

  return lines
    .map((line) => {
      const values = line.split(",");
      return {
        client_name: values[idx("client_name")],
        sessions: Number(values[idx("sessions")]),
        conversion_rate_pct: Number(values[idx("conversion_rate_pct")]),
      };
    })
    .filter((row) => row.client_name && row.sessions >= MIN_SESSIONS);
}

const rows = parseCsv(csv).filter((row) => row.client_name !== "(unknown)");
const top = rows
  .toSorted(
    (a, b) =>
      b.conversion_rate_pct - a.conversion_rate_pct || b.sessions - a.sessions,
  )
  .slice(0, 7);
const bottom = rows
  .toSorted(
    (a, b) =>
      a.conversion_rate_pct - b.conversion_rate_pct || b.sessions - a.sessions,
  )
  .slice(0, 7);

const displayRows = [
  ...top.map((row) => ({ ...row, kind: "top" as const })),
  { kind: "gap" as const },
  ...bottom.map((row) => ({ ...row, kind: "bottom" as const })),
];

function formatRate(value: number) {
  if (value === 0) return "0%";
  if (value >= 10) return `${value.toFixed(1)}%`;
  return `${value.toFixed(2)}%`;
}
</script>

<template>
  <section class="conversion-efficiency deck-panel">
    <header>
      <p>Session conversion efficiency</p>
      <span>clients with ≥{{ MIN_SESSIONS.toLocaleString() }} sessions</span>
    </header>

    <div class="conversion-efficiency__rows">
      <template v-for="(row, index) in displayRows" :key="index">
        <div v-if="row.kind === 'gap'" class="conversion-efficiency__gap">…</div>
        <div
          v-else
          class="conversion-efficiency__row"
          :class="`conversion-efficiency__row--${row.kind}`"
        >
          <div
            class="conversion-efficiency__bar"
            :style="{ '--bar-width': `${row.conversion_rate_pct}%` }"
          />
          <span class="conversion-efficiency__name">{{ row.client_name }}</span>
          <strong>{{ formatRate(row.conversion_rate_pct) }}</strong>
        </div>
      </template>
    </div>
  </section>
</template>

<style scoped>
.conversion-efficiency {
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  padding: 0.82rem;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 0.58rem;
  background:
    radial-gradient(circle at 86% 8%, rgba(106, 163, 247, 0.06), transparent 30%),
    rgba(11, 12, 15, 0.72);
}

.conversion-efficiency header {
  min-width: 0;
}

.conversion-efficiency header p {
  margin: 0;
  color: var(--deck-text);
  font-size: clamp(1rem, 2vw, 1.18rem);
  font-weight: 850;
  line-height: 1;
  letter-spacing: -0.055em;
}

.conversion-efficiency header span {
  display: block;
  margin-top: 0.24rem;
  color: var(--deck-dim);
  font-size: 0.58rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.conversion-efficiency__rows {
  min-height: 0;
  display: grid;
  grid-template-rows: repeat(7, minmax(0, 1fr)) 0.48fr repeat(7, minmax(0, 1fr));
  gap: 0.22rem;
}

.conversion-efficiency__row {
  position: relative;
  min-width: 0;
  min-height: 0;
  padding: 0 0.46rem;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.44rem;
  overflow: hidden;
  border: 1px solid rgba(185, 179, 165, 0.14);
  border-radius: var(--deck-radius-sm);
  background: rgba(20, 22, 27, 0.54);
}

.conversion-efficiency__bar {
  position: absolute;
  inset: 0 auto 0 0;
  width: max(2px, var(--bar-width));
  opacity: 0.72;
  background: linear-gradient(90deg, rgba(106, 163, 247, 0.32), rgba(106, 163, 247, 0.08));
}

.conversion-efficiency__row--bottom .conversion-efficiency__bar {
  background: linear-gradient(90deg, rgba(240, 107, 90, 0.28), rgba(240, 107, 90, 0.06));
}

.conversion-efficiency__name,
.conversion-efficiency__row strong {
  position: relative;
  z-index: 1;
}

.conversion-efficiency__name {
  min-width: 0;
  overflow: hidden;
  color: var(--deck-text);
  font-size: clamp(0.58rem, 1.25vw, 0.76rem);
  font-weight: 760;
  line-height: 1;
  letter-spacing: -0.04em;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversion-efficiency__row strong {
  color: var(--deck-accent-hi);
  font-size: clamp(0.58rem, 1.2vw, 0.74rem);
  font-weight: 900;
  line-height: 1;
}

.conversion-efficiency__row--bottom strong {
  color: var(--deck-no);
}

.conversion-efficiency__gap {
  display: grid;
  place-items: center;
  color: var(--deck-dim);
  font-size: 1rem;
  font-weight: 900;
  line-height: 1;
}
</style>
