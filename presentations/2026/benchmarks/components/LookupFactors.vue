<script setup lang="ts">
// Three like-for-like comparisons of answer-lookup flags per 445-trial run:
// change only the model, only the effort, or only the harness.
// Every figure is looked up in data/answer-lookup.json (analysis/answer-lookup/extract.py).
import data from "../data/answer-lookup.json";

const find = (agent: string, model: string, effort: string) => {
  const row = data.rows.find((r) => r.agent.startsWith(agent) && r.model === model && r.effort === effort);
  if (!row) throw new Error(`no row for ${agent} / ${model} / ${effort}`);
  return row;
};
const n = (agent: string, model: string, effort: string) => find(agent, model, effort).lookup;
const MAX = 58;
const w = (v: number) => `${(v / MAX) * 100}%`;

const models = ["DeepSeek V4.1 Flash", "DeepSeek V4 Flash", "DeepSeek V4 Flash Vision (exp)", "GLM-5.3-Flash", "GPT-5.6 Luna", "GPT-6 Luna"]
  .map((m) => ({ label: m.replace(" Flash Vision (exp)", " Vision"), v: n("fast-agent", m, "max") }));
const claude = data.rows.filter((r) => r.agent.startsWith("fast-agent") && /^(Opus|Sonnet|Haiku)/.test(r.model));
const claudeLine = { runs: claude.length, flags: claude.reduce((s, r) => s + r.lookup, 0), trials: claude.reduce((s, r) => s + r.trials, 0) };

const efforts = ["low", "medium", "high", "xhigh", "max"];
const ladders = [
  { label: "GPT-6 Luna on fast-agent", values: efforts.map((e) => n("fast-agent", "GPT-6 Luna", e)) },
  { label: "GPT-6 Astra on Codex", values: efforts.map((e) => n("Codex", "GPT-6 Astra", e)) },
];
const ladderMax = Math.max(...ladders.flatMap((l) => l.values));

const pairs = [
  { model: "GPT-5.6 Luna", effort: "max", agents: ["Codex", "fast-agent"] },
  { model: "GPT-5.6 Sol", effort: "xhigh", agents: ["Dext", "IndusAGI"] },
  { model: "DeepSeek V4 Flash", effort: "max", agents: ["fast-agent", "Mobile Coder"] },
  { model: "GPT-5.5", effort: "xhigh", agents: ["Codex", "Linghun", "Terminus 2"] },
  { model: "Opus 4.7", effort: "max", agents: ["Claude Code", "Terminus 2"] },
  { model: "Gemini 3 Pro", effort: "high", agents: ["Gemini CLI", "Terminus 2"] },
  { model: "Gemini 3.1 Pro", effort: "high", agents: ["Gemini CLI", "Terminus 2"] },
].map((p) => ({ ...p, cells: p.agents.map((a) => ({ agent: a, v: n(a, p.model, p.effort) })) }));

const hidden = efforts.map((e) => find("Codex", "GPT-6 Astra", e).webHidden);
</script>

<template>
  <div class="lkf">
    <section>
      <h3>Change the model</h3>
      <p class="lkf-sub">fast-agent, max effort</p>
      <div v-for="m in models" :key="m.label" class="lkf-row">
        <strong>{{ m.label }}</strong>
        <div class="lkf-bar"><span :style="{ width: w(m.v) }" /><b>{{ m.v }}</b></div>
      </div>
      <p class="lkf-foot">Claude on fast-agent: <b>{{ claudeLine.flags }}</b> flag in {{ claudeLine.runs }} runs, {{ claudeLine.trials.toLocaleString("en-GB") }} trials.</p>
    </section>

    <section>
      <h3>Change the effort</h3>
      <p class="lkf-sub">low to max, five steps</p>
      <div v-for="l in ladders" :key="l.label" class="lkf-ladder">
        <strong>{{ l.label }}</strong>
        <div class="lkf-steps">
          <span v-for="(v, i) in l.values" :key="i">
            <i :style="{ height: `${Math.max(2, (v / ladderMax) * 32)}px` }" />
            <b>{{ v }}</b>
          </span>
        </div>
      </div>
    </section>

    <section>
      <h3>Change the harness</h3>
      <p class="lkf-sub">same model, same effort</p>
      <div v-for="p in pairs" :key="p.model" class="lkf-pair">
        <strong>{{ p.model }}</strong>
        <span v-for="c in p.cells" :key="c.agent" :class="{ zero: c.v === 0 }">{{ c.agent }} <b>{{ c.v }}</b></span>
      </div>
      <p class="lkf-foot">Terminus 2: <b>0</b> in all five of its runs.</p>
    </section>
  </div>
  <p class="lk-note lkf-note"><b>We only count what the files record.</b> GPT-6 Astra on Codex: {{ Math.min(...hidden) }} to {{ Math.max(...hidden) }} trials per run have hosted search results missing.</p>
</template>

<style scoped>
.lkf { display: grid; grid-template-columns: 0.95fr 0.75fr 1.5fr; gap: 14px; align-items: stretch; font-family: var(--font-read); }
.lkf section { min-width: 0; padding: 12px 16px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.lkf h3 { margin: 0; font-size: 18px; white-space: nowrap; }
.lkf-sub { margin: 2px 0 10px; font-size: 13px; font-weight: 600; color: var(--text-muted); }
.lkf-row { display: grid; grid-template-columns: 126px 1fr; column-gap: 8px; align-items: center; margin-bottom: 4px; }
.lkf-row strong, .lkf-pair strong, .lkf-ladder strong { font-size: 13px; font-weight: 700; color: var(--petrol); line-height: 1.15; }
.lkf-bar { display: flex; align-items: center; height: 17px; }
.lkf-bar span { height: 100%; min-width: 2px; border-radius: 3px; background: var(--petrol); }
.lkf-bar b { margin-left: 6px; font-size: 14px; font-weight: 900; font-variant-numeric: tabular-nums; }
.lkf-note { font-size: 15px; margin-top: 8px; }
.lkf-foot { margin: 10px 0 0; font-size: 13px; line-height: 1.3; color: var(--text-muted); }
.lkf-foot b { font-weight: 900; color: var(--petrol); }
.lkf-ladder { margin-bottom: 8px; }
.lkf-steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; align-items: end; height: 54px; margin-top: 2px; }
.lkf-steps span { display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; }
.lkf-steps i { width: 100%; border-radius: 3px 3px 0 0; background: var(--petrol); }
.lkf-steps b { font-size: 14px; font-weight: 900; font-variant-numeric: tabular-nums; }
.lkf-pair { display: flex; align-items: baseline; gap: 2px 9px; padding: 2px 0; border-bottom: 2px solid var(--line); }
.lkf-pair strong { flex: 0 0 122px; white-space: nowrap; }
.lkf-pair span { font-size: 13px; color: var(--text-muted); white-space: nowrap; }
.lkf-pair span b { font-size: 15px; font-weight: 900; color: var(--petrol); font-variant-numeric: tabular-nums; }
.lkf-pair span.zero b { color: var(--teal); }
</style>
