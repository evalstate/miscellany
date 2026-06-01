<script setup lang="ts">
const rows = [
  {
    id: "all",
    label: "All MCP protocol messages",
    value: 10_000_000,
    pct: 100,
    color: "neutral",
    note: "baseline",
  },
  {
    id: "initialize",
    label: "initialize",
    value: 1_216_172,
    pct: 12.16,
    color: "amber",
    note: "setup / handshake",
  },
  {
    id: "tools",
    label: "tools/call:*",
    value: 62_232,
    pct: 0.622,
    color: "blue",
    note: "actual tool invocation",
  },
] as const;

function formatCount(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function barWidth(pct: number) {
  if (pct === 100) return "100%";
  return `${Math.max(pct, 0.9)}%`;
}
</script>

<template>
  <section class="protocol-efficiency">
    <div class="protocol-efficiency__body">
      <div
        class="protocol-efficiency__bars"
        aria-label="Protocol message mix per 10 million MCP requests"
      >
        <article
          v-for="row in rows"
          :key="row.id"
          class="protocol-efficiency__row"
          :class="`protocol-efficiency__row--${row.color}`"
        >
          <div class="protocol-efficiency__row-copy">
            <div>
              <h2>{{ row.label }}</h2>
              <p>{{ row.note }}</p>
            </div>
            <strong>{{ formatCount(row.value) }}</strong>
          </div>

          <div class="protocol-efficiency__track">
            <div
              class="protocol-efficiency__bar"
              :style="{ width: barWidth(row.pct) }"
            />
          </div>

          <div class="protocol-efficiency__meta">
            <span>{{
              row.pct === 100
                ? "100%"
                : `${row.pct.toFixed(row.pct < 1 ? 2 : 2)}%`
            }}</span>
            <em>of all MCP protocol messages</em>
          </div>
        </article>
      </div>

      <aside class="protocol-efficiency__callout">
        <span>Tool calls</span>
        <strong>0.62%</strong>
        <p>Only ~62k of every 10M protocol messages are actual tool calls.</p>
      </aside>
    </div>
  </section>
</template>

<style scoped>
.protocol-efficiency {
  box-sizing: border-box;
  position: relative;
  height: 100%;
  padding: 1.05rem 1.8rem 1rem;
  display: grid;
  background:
    radial-gradient(
      circle at 78% 18%,
      rgba(106, 163, 247, 0.15),
      transparent 28%
    ),
    radial-gradient(
      circle at 16% 68%,
      rgba(245, 164, 0, 0.12),
      transparent 28%
    ),
    rgba(20, 22, 27, 0.76);
  border: 1px solid var(--deck-border);
  border-radius: var(--deck-radius);
  box-shadow: var(--deck-shadow);
  overflow: visible;
}

.protocol-efficiency__body {
  width: 100%;
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 0.28fr);
  gap: 1rem;
  align-items: stretch;
}

.protocol-efficiency__bars {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: repeat(3, minmax(92px, 1fr));
  gap: 0.5rem;
}

.protocol-efficiency__row {
  min-height: 0;
  padding: 0.56rem 0.8rem;
  display: grid;
  grid-template-rows: minmax(32px, auto) 28px 18px;
  gap: 0.34rem;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 5px);
  background: rgba(11, 12, 15, 0.35);
}

.protocol-efficiency__row-copy {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 1rem;
}

.protocol-efficiency__row h2 {
  margin: 0;
  color: var(--deck-text);
  font-size: clamp(1.05rem, 2.35cqw, 1.45rem);
  line-height: 1;
  letter-spacing: -0.05em;
}

.protocol-efficiency__row p {
  margin: 0.28rem 0 0;
  color: var(--deck-dim);
  font-size: 0.58rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.protocol-efficiency__row-copy strong {
  color: var(--deck-text);
  font-size: clamp(1.22rem, 3.1cqw, 1.86rem);
  line-height: 0.92;
  letter-spacing: -0.07em;
  white-space: nowrap;
}

.protocol-efficiency__track {
  position: relative;
  height: 28px;
  border: 1px solid rgba(185, 179, 165, 0.13);
  border-radius: 999px;
  background: rgba(240, 236, 226, 0.055);
  overflow: hidden;
}

.protocol-efficiency__bar {
  height: 100%;
  border-radius: inherit;
  box-shadow: 0 0 28px rgba(255, 198, 73, 0.12);
}

.protocol-efficiency__row--neutral .protocol-efficiency__bar {
  background: linear-gradient(
    90deg,
    rgba(240, 236, 226, 0.58),
    rgba(240, 236, 226, 0.24)
  );
}

.protocol-efficiency__row--amber .protocol-efficiency__bar {
  background: linear-gradient(
    90deg,
    var(--deck-accent-hi),
    rgba(245, 164, 0, 0.58)
  );
}

.protocol-efficiency__row--blue .protocol-efficiency__bar {
  background: linear-gradient(
    90deg,
    var(--deck-info),
    rgba(106, 163, 247, 0.5)
  );
  min-width: 8px;
}

.protocol-efficiency__meta span {
  color: var(--deck-accent-hi);
  font-size: 0.9rem;
  font-weight: 950;
}

.protocol-efficiency__meta {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  min-height: 0;
}

.protocol-efficiency__row--blue .protocol-efficiency__meta span {
  color: var(--deck-info);
}

.protocol-efficiency__meta em {
  color: var(--deck-dim);
  font-size: 0.52rem;
  font-style: normal;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.protocol-efficiency__callout {
  min-width: 0;
  min-height: 0;
  padding: 0.95rem;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  align-content: center;
  justify-items: start;
  border: 1px solid var(--deck-info-line);
  border-radius: calc(var(--deck-radius) + 6px);
  background:
    radial-gradient(
      circle at 50% 18%,
      rgba(106, 163, 247, 0.18),
      transparent 42%
    ),
    rgba(11, 12, 15, 0.42);
}

.protocol-efficiency__callout span {
  display: block;
  color: var(--deck-dim);
  font-size: 0.62rem;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.protocol-efficiency__callout strong {
  display: block;
  margin: 0.2rem 0 0.6rem;
  color: var(--deck-info);
  font-size: clamp(3rem, 7.4cqw, 4.65rem);
  line-height: 0.92;
  letter-spacing: -0.08em;
}

.protocol-efficiency__callout p {
  max-width: 100%;
  margin: 0;
  color: var(--deck-muted);
  font-size: 0.84rem;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
</style>
