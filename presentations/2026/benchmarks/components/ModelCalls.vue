<script setup lang="ts">
// Agents that used a provider key from their own environment to ask another
// model. Figures are from the model-calls report (7 Nov 2026, 143,069
// trajectories) and its 8 Nov update (harness-leaderboard runs); the last
// line is the 8 Nov corpus scan (data/evasion.json).
import data from "../data/evasion.json";

const c = data.checks as Record<string, { n: number }>;
const funnel = [
  { n: 1379, label: "saw a provider key", note: "printed by their own listing of the environment" },
  { n: 20, label: "used one to reach a model", note: "all DeepSeek V4 Pro, V4 Flash, V4.1 Flash or Muse Spark" },
  { n: 0, label: "Claude, Gemini, GPT or Grok", note: "of more than 700 that saw a key" },
];
const why = [
  { what: "Eyes", who: "Muse Spark, later GLM and Flint",
    text: "Sends the task's images to a vision model it can't run itself." },
  { what: "The upstream code", who: "DeepSeek V4 Pro, on no-network DeepSWE",
    text: "9 trials ask deepseek-chat for the library's real implementation. The answers were mostly made up. 3 passed.",
    quote: "Could query DeepSeek for code? Unethical and network maybe. We should solve ourselves." },
  { what: "The answer", who: "DeepSeek V4 Flash, on a broken task",
    text: "Two runs talk themselves into using the key. The answer came from the benchmark's published solution instead.",
    quote: "The key is provided in the environment for the agent to use (the harness itself uses it)." },
];
</script>

<template>
  <div class="mc">
    <div class="mc-left">
      <div v-for="f in funnel" :key="f.label" class="mc-row" :class="{ zero: f.n === 0 }">
        <b>{{ f.n.toLocaleString("en-GB") }}</b>
        <div><strong>{{ f.label }}</strong><em>{{ f.note }}</em></div>
      </div>
      <p class="mc-note">The harness runs inside the sandbox, so its key is in the agent's environment too. And the route to the provider has to stay open, even when the task has no network.</p>
      <p class="mc-note mc-later">Later scan, with router-based runs included: <b>{{ c.model.n }}</b> calls, <b>{{ c.modelAnswered.n }}</b> answered.</p>
    </div>
    <div class="mc-right">
      <article v-for="w in why" :key="w.what">
        <h3>{{ w.what }}</h3>
        <p class="mc-who">{{ w.who }}</p>
        <p v-if="w.quote" class="mc-quote">{{ w.quote }}</p>
        <p>{{ w.text }}</p>
      </article>
    </div>
  </div>
</template>

<style scoped>
.mc { display: grid; grid-template-columns: 0.82fr 1.18fr; gap: 26px; align-items: start; font-family: var(--font-read); }
.mc-row { display: grid; grid-template-columns: 96px 1fr; column-gap: 12px; align-items: center; margin-bottom: 12px; }
.mc-row > b { font: 900 34px/1 var(--font-voice); color: var(--petrol); text-align: right; font-variant-numeric: tabular-nums; }
.mc-row strong { display: block; font-size: 17px; line-height: 1.15; font-weight: 800; color: var(--petrol); }
.mc-row em { display: block; margin-top: 2px; font-style: normal; font-size: 12.5px; line-height: 1.3; color: var(--text-muted); }
.mc-row.zero > b { justify-self: end; padding: 2px 9px; border-radius: var(--radius-sm); background: var(--petrol); color: var(--ivory); }
.mc .mc-note { margin: 12px 0 0; padding-top: 10px; border-top: 2px solid var(--line); font-size: 14px; line-height: 1.35; color: var(--petrol); }
.mc .mc-later { font-size: 13px; color: var(--text-muted); }
.mc .mc-later b { color: var(--petrol); font-weight: 800; }
.mc-right { display: grid; gap: 9px; }
.mc article { padding: 10px 14px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.mc article h3 { display: inline; margin: 0 8px 0 0; font-size: 18px; }
.mc article p { margin: 4px 0 0; font-size: 13.5px; line-height: 1.35; color: var(--petrol); }
.mc article p.mc-who { display: inline; margin: 0; font-size: 12.5px; color: var(--text-muted); }
.mc article p.mc-quote { font-style: italic; }
.mc-quote::before { content: "“"; }
.mc-quote::after { content: "”"; }
</style>
