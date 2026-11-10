<script setup lang="ts">
// The routes agents take when a request is blocked, as counts of trajectories
// whose commands have that shape, with one reviewed example of how each ended.
// Counts: data/evasion.json (analysis/evasion/extract.py, corpus scan v2).
// Examples: talk-hunt-2026-11-08 net-evasion findings (NET01-NET07).
import data from "../data/evasion.json";

const c = data.checks as Record<string, { n: number; families: [string, number][] }>;
const rows = [
  { key: "archive", name: "Ask a web archive", how: "web.archive.org/web/…",
    end: "Fetched the official POV-Ray archives when the live site stalled.", ok: true },
  { key: "reader", name: "Ask a reader service", how: "curl r.jina.ai/https://…",
    end: "Five direct requests got 403; the reader returned the manual.", ok: true },
  { key: "pinned", name: "Skip DNS", how: "curl --resolve host:443:IP",
    end: "In no-network DeepSWE: upstream source, and 146 packages.", ok: true },
  { key: "noproxy", name: "Go round the proxy", how: "curl --noproxy '*'",
    end: "Proxy: ERR_ACCESS_DENIED. Direct: connect timeout.", ok: false },
  { key: "model", name: "Ask another model", how: "POST /v1/chat/completions",
    end: `${c.modelAnswered.n} of the ${c.model.n} got an answer back.`, ok: true },
].map((r) => ({ ...r, n: c[r.key].n, top: c[r.key].families[0] }));
const max = Math.max(...rows.map((r) => r.n));
</script>

<template>
  <div class="evs">
    <div class="evs-head"><span>Route</span><span>Looks like</span><span>Trajectories</span><span>One reviewed case</span></div>
    <div v-for="r in rows" :key="r.key" class="evs-row">
      <strong>{{ r.name }}<em>most: {{ r.top[0] }}, {{ r.top[1] }}</em></strong>
      <code>{{ r.how }}</code>
      <div class="evs-bar"><span :style="{ width: `${(r.n / max) * 100}%` }" /><b>{{ r.n }}</b></div>
      <p :class="{ held: !r.ok }"><i>{{ r.ok ? "got through" : "held" }}</i>{{ r.end }}</p>
    </div>
  </div>
</template>

<style scoped>
.evs { font-family: var(--font-read); }
.evs-head, .evs-row { display: grid; grid-template-columns: 178px 196px 150px 1fr; column-gap: 16px; align-items: center; }
.evs-head { padding-bottom: 6px; border-bottom: var(--border); font-size: 12px; font-weight: 700; color: var(--text-muted); }
.evs-row { padding: 9px 0; border-bottom: 2px solid var(--line); }
.evs-row strong { font-size: 16px; line-height: 1.2; font-weight: 800; color: var(--petrol); }
.evs-row em { display: block; margin-top: 2px; font-style: normal; font-size: 12px; font-weight: 500; color: var(--text-muted); }
.evs-row code { justify-self: start; font-size: 12px; white-space: nowrap; }
.evs-bar { display: flex; align-items: center; height: 20px; }
.evs-bar span { height: 100%; min-width: 3px; border-radius: 3px; background: var(--petrol); }
.evs-bar b { margin-left: 6px; font-size: 16px; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.evs .evs-row p { margin: 0; font-size: 14px; line-height: 1.3; color: var(--petrol); }
.evs-row p i { display: inline-block; margin-right: 8px; padding: 1px 8px; border-radius: var(--radius-sm); background: var(--orange); color: var(--petrol); font-style: normal; font-size: 12px; font-weight: 800; }
.evs-row p.held i { background: var(--petrol); color: var(--ivory); }
</style>
