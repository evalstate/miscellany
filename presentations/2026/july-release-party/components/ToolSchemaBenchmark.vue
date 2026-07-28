<script setup lang="ts">
import benchmark from "../data-viz/tool_schema_benchmark.json";

type Run = {
  tokens: number;
  accepted_rate: number;
  canonical_rate: number;
  argument_f1: number;
};

type Row = {
  model: string;
  calls: number;
  "v0.3.30": Run;
  "v0.3.31": Run;
  token_change_pct: number;
};

const rows = benchmark.rows as Row[];
const summary = benchmark.summary;
const maxTokens = Math.max(
  ...rows.flatMap((row) => [row["v0.3.30"].tokens, row["v0.3.31"].tokens]),
);

function formatTokens(value: number) {
  return `${Math.round(value / 1000)}k`;
}

function formatRate(value: number) {
  return `${value.toFixed(1)}%`;
}
</script>

<template>
  <section class="tool-benchmark">
    <div class="tool-benchmark__stats">
      <article class="tool-benchmark__stat tool-benchmark__stat--hero">
        <strong>{{ Math.abs(summary.token_change_pct).toFixed(1) }}%</strong>
        <span>fewer total tokens</span>
        <small>
          {{ formatTokens(summary["v0.3.30"].tokens) }} →
          {{ formatTokens(summary["v0.3.31"].tokens) }}
        </small>
      </article>
      <article class="tool-benchmark__stat">
        <strong>+{{ summary.accepted_count_change }}</strong>
        <span>accepted result</span>
        <small>
          {{ formatRate(summary["v0.3.30"].accepted_rate) }} →
          {{ formatRate(summary["v0.3.31"].accepted_rate) }}
        </small>
      </article>
      <article class="tool-benchmark__stat">
        <strong>+{{ summary.canonical_count_change }}</strong>
        <span>strict results</span>
        <small>
          {{ formatRate(summary["v0.3.30"].canonical_rate) }} →
          {{ formatRate(summary["v0.3.31"].canonical_rate) }}
        </small>
      </article>
    </div>

    <div class="tool-benchmark__charts">
      <section class="tool-benchmark__tokens deck-panel">
        <header>
          <strong>Tokens by model</strong>
          <div class="tool-benchmark__legend">
            <span class="is-before">v0.3.30</span>
            <span class="is-after">v0.3.31</span>
          </div>
        </header>
        <div class="tool-benchmark__token-rows">
          <div
            v-for="row in rows"
            :key="row.model"
            class="tool-benchmark__token-row"
          >
            <span class="tool-benchmark__model">{{ row.model }}</span>
            <div class="tool-benchmark__bars">
              <i
                class="is-before"
                :style="{
                  width: `${(row['v0.3.30'].tokens / maxTokens) * 100}%`,
                }"
              ></i>
              <i
                class="is-after"
                :style="{
                  width: `${(row['v0.3.31'].tokens / maxTokens) * 100}%`,
                }"
              ></i>
            </div>
            <div class="tool-benchmark__token-change">
              <span>
                {{ formatTokens(row["v0.3.30"].tokens) }} →
                {{ formatTokens(row["v0.3.31"].tokens) }}
              </span>
              <strong>{{ row.token_change_pct.toFixed(0) }}%</strong>
            </div>
          </div>
        </div>
      </section>

      <section class="tool-benchmark__accuracy deck-panel">
        <header>
          <strong>Accuracy held — or improved</strong>
          <span>all 604 matched calls</span>
        </header>
        <div class="tool-benchmark__accuracy-row">
          <div>
            <span>Accepted</span>
            <small>equivalent result</small>
          </div>
          <strong>{{ formatRate(summary["v0.3.31"].accepted_rate) }}</strong>
          <em>+{{ summary.accepted_rate_change_pp.toFixed(1) }} pp</em>
        </div>
        <div class="tool-benchmark__accuracy-row">
          <div>
            <span>Strict</span>
            <small>canonical result</small>
          </div>
          <strong>{{ formatRate(summary["v0.3.31"].canonical_rate) }}</strong>
          <em>+{{ summary.canonical_rate_change_pp.toFixed(1) }} pp</em>
        </div>
        <div class="tool-benchmark__accuracy-row">
          <div>
            <span>Argument F1</span>
            <small>argument quality</small>
          </div>
          <strong>{{ formatRate(summary["v0.3.31"].argument_f1) }}</strong>
          <em>+{{ summary.argument_f1_change_pp.toFixed(1) }} pp</em>
        </div>
      </section>
    </div>
  </section>
</template>

<style scoped>
.tool-benchmark {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 0.75rem;
  height: 100%;
  min-height: 0;
}

.tool-benchmark__stats {
  display: grid;
  grid-template-columns: 1.35fr 1fr 1fr;
  gap: 0.7rem;
}

.tool-benchmark__stat {
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: baseline;
  column-gap: 0.65rem;
  padding: 0.62rem 0.82rem;
  border: 1px solid var(--deck-border);
  border-radius: var(--deck-radius);
  background: rgba(255, 255, 255, 0.72);
}

.tool-benchmark__stat--hero {
  border-color: rgba(106, 209, 156, 0.48);
  background:
    linear-gradient(120deg, rgba(106, 209, 156, 0.14), transparent 74%),
    rgba(255, 255, 255, 0.78);
}

.tool-benchmark__stat strong {
  color: var(--deck-ok);
  font-family: var(--deck-font-sans);
  font-size: 1.72rem;
  line-height: 1;
  letter-spacing: -0.055em;
}

.tool-benchmark__stat span {
  color: var(--deck-text);
  font-size: 0.78rem;
  font-weight: 750;
}

.tool-benchmark__stat small {
  grid-column: 1 / -1;
  margin-top: 0.18rem;
  color: var(--deck-dim);
  font-size: 0.55rem;
}

.tool-benchmark__charts {
  display: grid;
  grid-template-columns: minmax(0, 1.58fr) minmax(0, 0.94fr);
  gap: 0.75rem;
  min-height: 0;
}

.tool-benchmark__tokens,
.tool-benchmark__accuracy {
  min-height: 0;
  padding: 0.72rem 0.82rem;
}

.tool-benchmark__tokens {
  display: flex;
  flex-direction: column;
}

.tool-benchmark__tokens header,
.tool-benchmark__accuracy header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.56rem;
}

.tool-benchmark__tokens header > strong,
.tool-benchmark__accuracy header > strong {
  font-size: 0.7rem;
  letter-spacing: -0.02em;
}

.tool-benchmark__accuracy header > strong {
  font-size: 0.78rem;
}

.tool-benchmark__accuracy header > span {
  color: var(--deck-muted);
  font-size: 0.56rem;
}

.tool-benchmark__legend {
  display: flex;
  gap: 0.75rem;
  color: var(--deck-muted);
  font-size: 0.52rem;
}

.tool-benchmark__legend span::before {
  content: "";
  display: inline-block;
  width: 0.5rem;
  height: 0.5rem;
  margin-right: 0.28rem;
  border-radius: 2px;
  vertical-align: -0.04rem;
}

.tool-benchmark__legend .is-before::before {
  background: var(--deck-border-2);
}

.tool-benchmark__legend .is-after::before {
  background: var(--deck-ok);
}

.tool-benchmark__token-rows {
  display: grid;
  grid-template-rows: repeat(6, minmax(0, 1fr));
  flex: 1;
  min-height: 0;
}

.tool-benchmark__token-row {
  display: grid;
  grid-template-columns: 5.5rem minmax(0, 1fr) 5.4rem;
  align-items: center;
  gap: 0.48rem;
}

.tool-benchmark__model {
  color: var(--deck-muted);
  font-size: 0.56rem;
  text-align: right;
}

.tool-benchmark__bars {
  position: relative;
  height: 1.15rem;
  border-left: 1px solid var(--deck-border-2);
  background: linear-gradient(
    90deg,
    transparent 49.8%,
    rgba(255, 255, 255, 0.045) 50%,
    transparent 50.2%
  );
}

.tool-benchmark__bars i {
  position: absolute;
  left: 0;
  height: 0.43rem;
  border-radius: 0 2px 2px 0;
}

.tool-benchmark__bars .is-before {
  top: 0.06rem;
  background: var(--deck-border-2);
}

.tool-benchmark__bars .is-after {
  bottom: 0.06rem;
  background: linear-gradient(90deg, #4fae80, var(--deck-ok));
}

.tool-benchmark__token-change {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: baseline;
  gap: 0.35rem;
}

.tool-benchmark__token-change span {
  color: var(--deck-muted);
  font-size: 0.45rem;
  white-space: nowrap;
}

.tool-benchmark__token-change strong {
  color: var(--deck-ok);
  font-size: 0.56rem;
  text-align: right;
}

.tool-benchmark__accuracy {
  display: flex;
  flex-direction: column;
}

.tool-benchmark__accuracy-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.25rem 0.55rem;
  padding: 0.62rem 0;
  border-top: 1px solid var(--deck-border);
}

.tool-benchmark__accuracy-row div {
  display: grid;
}

.tool-benchmark__accuracy-row span {
  color: var(--deck-text);
  font-size: 0.68rem;
  font-weight: 750;
}

.tool-benchmark__accuracy-row small {
  color: var(--deck-muted);
  font-size: 0.52rem;
}

.tool-benchmark__accuracy-row > strong {
  font-family: var(--deck-font-sans);
  font-size: 1.36rem;
  letter-spacing: -0.05em;
}

.tool-benchmark__accuracy-row em {
  grid-column: 1 / -1;
  justify-self: end;
  color: var(--deck-ok);
  font-size: 0.59rem;
  font-style: normal;
  font-weight: 750;
}
</style>
