<script setup lang="ts">
// What the two mirror sites' public git history says about how the pages were
// made: sixteen commit subjects, verbatim, with the words that give the game
// away picked out, and commits per day. Data: data/lure-commits.json
// (analysis/lure/extract_commits.py). No hashes or account names.
import log from "../data/lure-commits.json";

const HOT = /(Search Console|search terms|SEO|search snippets|page snippets|curl-first|curl runs|curl setup|sudo|no-install|questionnaire|search study|Gemini calibration|gpt55andopus48|Grok|Terminal-Bench patch pages|controlled research pages offline)/;
const fmt = (iso: string) => `${Number(iso.slice(8, 10))} ${["Jun", "Jul", "Aug"][Number(iso.slice(5, 7)) - 6]}`;
const lines = log.picks.map((p) => ({ date: fmt(p.date).padStart(6, " "), parts: p.subject.split(HOT).map((text, i) => ({ text, hot: i % 2 === 1 })) }));

const W = 380, H = 62, T0 = Date.UTC(2026, 5, 1), T1 = Date.UTC(2026, 8, 1);
const x = (iso: string) => ((Date.parse(iso + "T00:00:00Z") - T0) / (T1 - T0)) * W;
const peak = Math.max(...log.days.map((d) => d.n));
const bars = log.days.map((d) => ({ x: x(d.date), h: (d.n / peak) * (H - 16), n: d.n }));
const ticks = [["Jun", "2026-06-01"], ["Jul", "2026-07-01"], ["Aug", "2026-08-01"]].map(([label, iso]) => ({ label, x: x(iso) }));
</script>

<template>
  <div class="lc">
    <div class="lc-term">
      <div class="lc-head"><span>git log · two mirror sites · {{ log.total }} commits</span><span aria-hidden="true">❯_</span></div>
      <pre><code><span v-for="(l, i) in lines" :key="i"><i>{{ l.date }}</i>  <template v-for="(p, j) in l.parts" :key="j"><b v-if="p.hot">{{ p.text }}</b><template v-else>{{ p.text }}</template></template>
</span></code></pre>
    </div>
    <div class="lc-side">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Commits per day, June to August 2026">
        <line x1="0" :x2="W" :y1="H - 14" :y2="H - 14" />
        <rect v-for="b in bars" :key="b.x" :x="b.x" :y="H - 14 - b.h" width="3.4" :height="b.h" />
        <text v-for="t in ticks" :key="t.label" :x="t.x + 2" :y="H - 2">{{ t.label }}</text>
      </svg>
      <p class="lc-cap">Commits per day. The busiest day has {{ peak }}.</p>
      <p><b>Built to be found.</b> Search Console, SEO, and the command moved into the search snippet.</p>
      <p><b>Built to be run.</b> “curl-first”, “no-install routes”, a “sudo” bundle.</p>
      <p><b>Tuned per model.</b> Gemini, Grok, GPT and Opus appear by name.</p>
      <p><b>People too.</b> A questionnaire asks what a person would search first.</p>
      <p class="lc-limit">The Terminal-Bench pages were only edited 8–17 Jun. The July and August tuning is on pages for other benchmarks.</p>
    </div>
  </div>
</template>

<style scoped>
.lc { display: grid; grid-template-columns: 1.22fr 1fr; gap: 26px; align-items: start; font-family: var(--font-read); }
.lc-term { border: var(--border); border-radius: var(--radius-md); background: var(--petrol); color: var(--ivory); overflow: hidden; }
.lc-head { display: flex; justify-content: space-between; padding: 7px 14px; border-bottom: 2px solid var(--petrol-2); font: 500 12px var(--font-code); color: rgba(255, 247, 232, 0.78); }
.lc-term pre { margin: 0; padding: 10px 14px 12px; border: 0; border-radius: 0; background: transparent !important; font: 500 11.5px/1.62 var(--font-code); white-space: pre; overflow: hidden; }
.lc-term span { color: rgba(255, 247, 232, 0.86); }
.lc-term i { font-style: normal; color: rgba(255, 247, 232, 0.55); }
.lc-term b { font-weight: 700; color: var(--orange); }
.lc-side svg { display: block; width: 100%; height: auto; }
.lc-side svg line { stroke: var(--petrol); stroke-width: 2; }
.lc-side svg rect { fill: var(--petrol); }
.lc-side svg text { font: 600 10px var(--font-read); fill: var(--text-muted); }
.lc .lc-side p { margin: 0 0 8px; font-size: 15px; line-height: 1.35; color: var(--petrol); }
.lc .lc-side p b { font-weight: 800; }
.lc .lc-side p.lc-cap { margin: 4px 0 12px; font-size: 12px; color: var(--text-muted); }
.lc .lc-side p.lc-limit { margin-top: 12px; padding-top: 8px; border-top: 2px solid var(--line); font-size: 13px; color: var(--text-muted); }
</style>
