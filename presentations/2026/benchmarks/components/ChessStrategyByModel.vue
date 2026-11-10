<script setup lang="ts">
// How each model family went about finding the move, as a share of its
// fast-agent chess-best-move trials: ran Stockfish, used python-chess only,
// or used no chess library. Colours match ChessSankey. The right-hand
// columns give how many trials printed the board as ASCII and how many were
// rewarded. Data: data/chess-flows.json (analysis/chess-intro/flows.py).
import flows from "../data/chess-flows.json";

type Row = { model: string; n: number; stockfish: number; pychess: number; none: number; rewarded: number; ascii: number };
const parts = [
  { key: "stockfish", label: "Runs Stockfish" },
  { key: "pychess", label: "python-chess only" },
  { key: "none", label: "No chess library" },
] as const;
const rows = (flows.strategy as Row[]).map((r) => ({
  ...r,
  segments: parts.map((p) => ({ key: p.key, width: `${(r[p.key] / r.n) * 100}%`, share: Math.floor((r[p.key] / r.n) * 100) })).filter((s) => s.share > 0 || s.width !== "0%"),
  rate: `${Math.floor((r.rewarded / r.n) * 100)}%`,
}));
</script>

<template>
  <div class="fa-strat">
    <div class="fa-strat-legend">
      <span v-for="p in parts" :key="p.key"><i :class="`is-${p.key}`" />{{ p.label }}</span>
      <em>Prints ASCII</em>
      <em>Rewarded</em>
    </div>
    <div v-for="r in rows" :key="r.model" class="fa-strat-row">
      <strong>{{ r.model }}</strong>
      <div class="fa-strat-bar" :aria-label="`${r.model}: ${r.stockfish} Stockfish, ${r.pychess} python-chess only, ${r.none} no chess library, of ${r.n}`">
        <span v-for="s in r.segments" :key="s.key" :class="`is-${s.key}`" :style="{ width: s.width }">
          <b v-if="s.share >= 9">{{ s.share }}%</b>
        </span>
      </div>
      <em :class="{ on: r.ascii > 0 }">{{ r.ascii > 0 ? `${r.ascii} of ${r.n}` : "" }}</em>
      <em class="on">{{ r.rate }}</em>
    </div>
  </div>
</template>

<style scoped>
.fa-strat { display: grid; gap: 4px; font-family: var(--font-read); }
.fa-strat-legend, .fa-strat-row { display: grid; grid-template-columns: 158px 1fr 96px 76px; column-gap: 14px; align-items: center; }
.fa-strat-legend { grid-template-columns: 158px 1fr 96px 76px; margin-bottom: 4px; font-size: 13px; font-weight: 600; color: var(--text-muted); }
/* the three legend entries share the bar column */
.fa-strat-legend span { grid-row: 1; display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.fa-strat-legend span:nth-child(1) { grid-column: 2; justify-self: start; }
.fa-strat-legend span:nth-child(2) { grid-column: 2; justify-self: center; }
.fa-strat-legend span:nth-child(3) { grid-column: 2; justify-self: end; }
.fa-strat-legend i { width: 14px; height: 14px; border-radius: 3px; }
.fa-strat-legend em { grid-row: 1; font-style: normal; text-align: right; }
.fa-strat-legend em:nth-of-type(1) { grid-column: 3; }
.fa-strat-legend em:nth-of-type(2) { grid-column: 4; }
.fa-strat-row strong { font-size: 15px; font-weight: 700; color: var(--petrol); text-align: right; white-space: nowrap; }
.fa-strat-bar { display: flex; gap: 2px; height: 21px; }
.fa-strat-bar span { display: flex; align-items: center; min-width: 0; border-radius: 4px; overflow: hidden; }
.fa-strat-bar b { padding-left: 7px; font-size: 13px; font-weight: 800; color: var(--ivory); font-variant-numeric: tabular-nums; }
.is-stockfish { background: var(--teal); }
.is-pychess { background: var(--petrol); }
.is-none { background: var(--orange); }
.fa-strat-bar .is-none b { color: var(--petrol); }
.fa-strat-row em { font-style: normal; text-align: right; font-size: 14px; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.fa-strat-row em.on { font-weight: 800; color: var(--petrol); }
</style>
