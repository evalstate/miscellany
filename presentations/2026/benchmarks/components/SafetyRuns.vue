<script setup lang="ts">
// Trials the named model did not finish, per run: stopped by the provider's
// classifier (harnesses with no fallback, fast-agent among them), or handed
// to a fallback model (Claude Code, Terminus 2), which then passes or fails.
// The last two columns are the run's pass rate as scored, and the named
// model's pass rate over only the trials it finished itself.
// Data: data/safety-stops.json (analysis/safety-stops/extract.py).
import data from "../data/safety-stops.json";

const SITE: Record<string, string> = { leaderboard: "leaderboard", submissions: "submission", "fast-agent": "our run" };
const ORDER = ["leaderboard", "submissions", "fast-agent"];
const pct = (a: number, b: number) => (Math.floor((a / b) * 1000) / 10).toFixed(1);
const all = data.rows.filter((r) => r.site === "leaderboard" || r.stops >= 12);
const lost = (r: (typeof all)[number]) => r.stops + r.handed;
const max = Math.max(...all.map(lost));
const w = (n: number) => `${(n / max) * 100}%`;
const rows = all
  .sort((a, b) => ORDER.indexOf(a.site) - ORDER.indexOf(b.site) || lost(b) - lost(a))
  .map((r) => ({
    ...r, source: SITE[r.site], lost: lost(r),
    pass: pct(r.rewarded, r.trials), own: pct(r.rewarded - r.stopsRewarded - r.handedRewarded, r.trials - lost(r)),
  }));
</script>

<template>
  <div class="sfr">
    <div class="sfr-head">
      <span>Source</span><span>Run</span>
      <span><i class="sfr-stop" />Stopped<i class="sfr-pass" />Handed on, passed<i class="sfr-fail" />Handed on, failed</span>
      <span>Pass rate</span><span>Own trials</span>
    </div>
    <div v-for="r in rows" :key="r.site + r.agent + r.model + r.effort" class="sfr-row" :class="`is-${r.site}`">
      <i>{{ r.source }}</i>
      <strong>{{ r.model }} <small>{{ r.effort }}</small><em>{{ r.agent }}</em></strong>
      <div class="sfr-bar">
        <span v-if="r.stops" class="sfr-stop" :style="{ width: w(r.stops) }" />
        <span v-if="r.handed" class="sfr-pass" :style="{ width: w(r.handedRewarded) }" />
        <span v-if="r.handed" class="sfr-fail" :style="{ width: w(r.handed - r.handedRewarded) }" />
        <b>{{ r.lost }}</b><small v-if="r.handed">{{ r.handedRewarded }} passed</small>
      </div>
      <span class="sfr-n">{{ r.pass }}%</span>
      <span class="sfr-n sfr-clear">{{ r.own }}%</span>
    </div>
    <p class="lk-note"><b>A stop and a hand-off are the same event.</b> fast-agent has no fallback model; Claude Code and Terminus 2 pass the trial to Opus 4.8.</p>
  </div>
</template>

<style scoped>
.sfr { font-family: var(--font-read); }
.sfr-head, .sfr-row { display: grid; grid-template-columns: 92px 290px 1fr 70px 82px; column-gap: 14px; align-items: center; }
.sfr-head { padding-bottom: 4px; border-bottom: var(--border); font-size: 12px; font-weight: 700; color: var(--text-muted); white-space: nowrap; }
.sfr-head span:nth-child(n + 4) { text-align: right; }
.sfr-head i { display: inline-block; width: 11px; height: 11px; margin: 0 4px 0 12px; border-radius: 3px; vertical-align: -1px; }
.sfr-head i:first-child { margin-left: 0; }
.sfr-row { height: 20px; border-bottom: 2px solid var(--line); }
.sfr-row > i { justify-self: start; padding: 0 7px; border: 2px solid var(--petrol); border-radius: var(--radius-sm); font-style: normal; font-size: 10.5px; line-height: 13px; font-weight: 800; color: var(--petrol); }
.sfr-row.is-leaderboard > i { background: var(--petrol); color: var(--ivory); }
.sfr-row strong { font-size: 13px; line-height: 1; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.sfr-row strong small { font-size: 12px; font-weight: 600; color: var(--text-muted); }
.sfr-row em { margin-left: 8px; font-style: normal; font-size: 12px; font-weight: 500; color: var(--text-muted); }
.sfr-bar { display: flex; align-items: center; gap: 2px; height: 12px; }
.sfr-bar span { height: 100%; border-radius: 3px; }
.sfr-stop { background: var(--orange); }
.sfr-pass { background: var(--teal); }
.sfr-fail { background: var(--petrol-tint); }
.sfr-bar b { margin-left: 5px; font-size: 13px; line-height: 1; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.sfr-bar small { margin-left: 6px; font-size: 11.5px; color: var(--text-muted); white-space: nowrap; }
.sfr-n { text-align: right; font-size: 13px; line-height: 1; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; }
.sfr-clear { font-weight: 800; color: var(--petrol); }
.sfr .lk-note { margin-top: 6px; font-size: 13.5px; white-space: nowrap; }
</style>
