<script setup lang="ts">
// The like-for-like cut of the flag channels: one benchmark (DeepSWE 1.1), one
// harness (mini-swe-agent), one effort setting (max), 448 to 452 trials a run.
// Rows are sorted by how often a tool call is flagged; the second bar is how
// often the reasoning is. Data: data/awareness-channels.json.
import data from "../data/awareness-channels.json";

const NAMES: Record<string, string> = {
  claude_fable_5: "Fable 5", claude_opus_4_8: "Opus 4.8", claude_opus_5: "Opus 5", claude_sonnet_5: "Sonnet 5",
  deepseek_v4_flash: "DeepSeek V4 Flash", deepseek_v4_pro: "DeepSeek V4 Pro", glm_5_2: "GLM-5.2", glm_5_3_flash: "GLM-5.3 Flash", glm_5_3: "GLM-5.3",
  gpt_5_6_luna: "GPT-5.6 Luna", gpt_5_6_sol: "GPT-5.6 Sol", gpt_5_6_terra: "GPT-5.6 Terra", kimi_k3: "Kimi K3",
};
const pct = (a: number, b: number) => Math.floor((a / b) * 1000) / 10;
const rows = data.deepswe
  .filter((r) => r.run.endsWith("_max"))
  .map((r) => ({
    name: NAMES[r.run.replace(/_max$/, "")] ?? r.run, trials: r.trials, text: r.recordedTrials > 0,
    tool: pct(r.tool, r.trials), reasoning: pct(r.reasoning, r.trials),
  }))
  .sort((a, b) => b.tool - a.tool);
const by = (n: string) => rows.find((r) => r.name === n)!;
const gpt = ["GPT-5.6 Terra", "GPT-5.6 Sol", "GPT-5.6 Luna"].map(by);
const range = (xs: number[]) => `${Math.min(...xs).toFixed(1)} to ${Math.max(...xs).toFixed(1)}%`;
const flash = by("DeepSeek V4 Flash"), glm = by("GLM-5.3");
const blind = rows.filter((r) => !r.text);
</script>

<template>
  <div class="fam">
    <div class="fam-chart">
      <div class="fam-row fam-head"><span /><b><i class="fam-t" />Flagged in a tool call</b><b><i class="fam-r" />Flagged in reasoning</b></div>
      <div v-for="r in rows" :key="r.name" class="fam-row">
        <strong>{{ r.name }}</strong>
        <div class="fam-bar"><span class="fam-t" :style="{ width: `${r.tool * 0.8}%` }" /><em>{{ r.tool.toFixed(1) }}%</em></div>
        <div v-if="r.text" class="fam-bar"><span class="fam-r" :style="{ width: `${r.reasoning * 0.8}%` }" /><em>{{ r.reasoning.toFixed(1) }}%</em></div>
        <div v-else class="fam-bar fam-blank">nothing to read</div>
      </div>
    </div>
    <div class="fam-facts">
      <section>
        <h3>The tool calls look alike</h3>
        <p>GPT-5.6 at max: <b>{{ range(gpt.map((r) => r.tool)) }}</b> of trials have a flagged tool call. DeepSeek V4 Flash: <b>{{ flash.tool.toFixed(1) }}%</b>. GLM-5.3: {{ glm.tool.toFixed(1) }}%.</p>
      </section>
      <section>
        <h3>The reasoning doesn't</h3>
        <p>GPT-5.6: <b>{{ range(gpt.map((r) => r.reasoning)) }}</b> flagged in reasoning. DeepSeek V4 Flash: <b>{{ flash.reasoning.toFixed(1) }}%</b>. GLM-5.3: {{ glm.reasoning.toFixed(1) }}%.</p>
      </section>
      <section>
        <h3>Or there is none</h3>
        <p>{{ blind.map((r) => r.name).join(", ") }}: <b>{{ range(blind.map((r) => r.tool)) }}</b> flagged tool calls, and no reasoning text in the trace.</p>
      </section>
    </div>
  </div>
</template>

<style scoped>
.fam { display: grid; grid-template-columns: 1.5fr 1fr; gap: 22px; align-items: start; font-family: var(--font-read); }
.fam-row { display: grid; grid-template-columns: 138px 1.3fr 1fr; column-gap: 14px; align-items: center; height: 24px; border-bottom: 2px solid var(--line); }
.fam-head { height: 24px; border-bottom: var(--border); }
.fam-head b { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.fam-head i { width: 13px; height: 13px; border-radius: 3px; }
.fam-row strong { font-size: 14px; font-weight: 700; color: var(--petrol); white-space: nowrap; }
.fam-bar { display: flex; align-items: center; height: 14px; }
.fam-bar span { height: 100%; min-width: 2px; border-radius: 3px; }
.fam-r { background: var(--teal); }
.fam-t { background: var(--petrol); }
.fam-bar em { margin-left: 6px; font-style: normal; font-size: 13px; line-height: 1; font-weight: 700; color: var(--petrol); font-variant-numeric: tabular-nums; }
.fam-blank { padding-left: 8px; border-radius: 3px; background: var(--ivory-deep); font-size: 12px; line-height: 14px; color: var(--text-muted); }
.fam-facts { display: grid; gap: 10px; }
.fam-facts section { padding: 10px 16px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.fam-facts h3 { margin: 0 0 3px; font-size: 19px; }
.fam-facts p { margin: 0; font-size: 14px; line-height: 1.35; color: var(--petrol); }
.fam-facts p b { font-weight: 800; }
</style>
