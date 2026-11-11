<script setup lang="ts">
// Safety stops per run, with where the run comes from: the main Terminal-Bench
// 2.1 leaderboard, a pull-request submission, or our own fast-agent runs. The
// last two columns are the run's pass rate, and its pass rate over the trials
// that were not stopped. Data: data/safety-stops.json.
import data from "../data/safety-stops.json";

const SITE: Record<string, string> = { leaderboard: "leaderboard", submissions: "submission", "fast-agent": "our run" };
const ORDER = ["leaderboard", "submissions", "fast-agent"];
const pct = (a: number, b: number) => (Math.floor((a / b) * 1000) / 10).toFixed(1);
const all = data.rows.filter((r) => r.site === "leaderboard" || r.stops >= 12);
const max = Math.max(...all.map((r) => r.stops));
const rows = all
  .sort((a, b) => ORDER.indexOf(a.site) - ORDER.indexOf(b.site) || b.stops - a.stops)
  .map((r) => ({
    ...r, source: SITE[r.site], width: `${(r.stops / max) * 100}%`,
    pass: pct(r.rewarded, r.trials), clear: pct(r.rewarded - r.stopsRewarded, r.trials - r.stops),
  }));
const board = data.rows.filter((r) => r.site === "leaderboard");
const boardStops = board.reduce((s, r) => s + r.stops, 0);
</script>

<template>
  <div class="sfr">
    <div class="sfr-head"><span>Source</span><span>Run</span><span>Stopped, of {{ rows[0].trials }}</span><span>Pass rate</span><span>Not stopped</span></div>
    <div v-for="r in rows" :key="r.site + r.agent + r.model + r.effort" class="sfr-row" :class="`is-${r.site}`">
      <i>{{ r.source }}</i>
      <strong>{{ r.model }} <small>{{ r.effort }}</small><em>{{ r.agent }}</em></strong>
      <div class="sfr-bar"><span :style="{ width: r.width }" /><b>{{ r.stops }}</b></div>
      <span class="sfr-n">{{ r.pass }}%</span>
      <span class="sfr-n sfr-clear">{{ r.clear }}%</span>
    </div>
    <p class="lk-note"><b>{{ board.length }} main-leaderboard rows have stops, {{ boardStops }} trials in all.</b> They are the five GPT-6 Astra rows on Codex.</p>
  </div>
</template>

<style scoped>
.sfr { font-family: var(--font-read); }
.sfr-head, .sfr-row { display: grid; grid-template-columns: 92px 300px 1fr 78px 92px; column-gap: 14px; align-items: center; }
.sfr-head { padding-bottom: 4px; border-bottom: var(--border); font-size: 12px; font-weight: 700; color: var(--text-muted); }
.sfr-head span:nth-child(n + 4) { text-align: right; }
.sfr-row { height: 23px; border-bottom: 2px solid var(--line); }
.sfr-row i { justify-self: start; padding: 0 7px; border: 2px solid var(--petrol); border-radius: var(--radius-sm); font-style: normal; font-size: 10.5px; line-height: 14px; font-weight: 800; color: var(--petrol); }
.sfr-row.is-leaderboard i { background: var(--petrol); color: var(--ivory); }
.sfr-row strong { font-size: 13px; line-height: 1; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.sfr-row small { font-size: 12px; font-weight: 600; color: var(--text-muted); }
.sfr-row em { margin-left: 8px; font-style: normal; font-size: 12px; font-weight: 500; color: var(--text-muted); }
.sfr-bar { display: flex; align-items: center; height: 12px; }
.sfr-bar span { height: 100%; border-radius: 3px; background: var(--orange); }
.sfr-bar b { margin-left: 6px; font-size: 13px; line-height: 1; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.sfr-n { text-align: right; font-size: 13px; line-height: 1; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.sfr-clear { font-weight: 800; color: var(--petrol); }
</style>
