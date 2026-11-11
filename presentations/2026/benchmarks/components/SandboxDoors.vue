<script setup lang="ts">
// The three things an agent can reach beyond its sandbox, and whether any
// trajectory got there. Counts: data/evasion.json (corpus scan v2, 181,732
// trajectories); the reviewed outcomes are from the talk-hunt reports.
import data from "../data/evasion.json";

const c = data.checks as Record<string, { n: number }>;
const doors = [
  { n: 1, name: "The internet", who: "when the task says no network",
    tried: `${c.pinned.n} skipped DNS, ${c.noproxy.n} went round the proxy`,
    out: "Yes", detail: "Where only DNS was blocked, a pinned address got through. Two DeepSWE runs installed packages that way and were rewarded.", ok: true },
  { n: 2, name: "Another model", who: "with a provider key found in the environment",
    tried: `${c.model.n} sent a request`,
    out: "Yes", detail: `${c.modelAnswered.n} got an answer back: vision they lacked, upstream code, or a guess at the answer.`, ok: true },
  { n: 3, name: "The machine underneath", who: "the cluster API, the host disk, other processes",
    tried: `${c.k8sRequest.n} asked the Kubernetes API`,
    out: "No", detail: "No request was answered and no host file was read in the traces reviewed.", ok: false },
];
</script>

<template>
  <div class="sbx">
    <div class="sbx-box"><span>sandbox</span><b>agent</b><em>task files · a shell · the harness and its key</em></div>
    <div class="sbx-doors">
      <article v-for="d in doors" :key="d.n" :class="{ shut: !d.ok }">
        <i>{{ d.n }}</i>
        <div>
          <h3>{{ d.name }}</h3>
          <p class="sbx-who">{{ d.who }}</p>
          <p>{{ d.tried }}.</p>
        </div>
        <div class="sbx-out"><b>{{ d.out }}</b><p>{{ d.detail }}</p></div>
      </article>
    </div>
  </div>
</template>

<style scoped>
.sbx { display: grid; grid-template-columns: 170px 1fr; gap: 0; align-items: stretch; font-family: var(--font-read); }
.sbx-box { display: flex; flex-direction: column; justify-content: center; gap: 6px; padding: 16px; border: var(--border); border-right-width: 6px; border-radius: var(--radius-lg) 0 0 var(--radius-lg); background: var(--petrol); color: var(--ivory); }
.sbx-box span { font: 500 12px var(--font-code); color: rgba(255, 247, 232, 0.7); }
.sbx-box b { font: 900 30px/1 var(--font-voice); }
.sbx-box em { font-style: normal; font-size: 12.5px; line-height: 1.35; color: rgba(255, 247, 232, 0.82); }
.sbx-doors { display: grid; gap: 10px; padding-left: 26px; }
.sbx article { position: relative; display: grid; grid-template-columns: 40px 1fr 1.15fr; column-gap: 14px; align-items: center; padding: 11px 16px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.sbx article::before { content: ""; position: absolute; left: -28px; top: 50%; width: 26px; border-top: 3px solid var(--orange); }
.sbx article.shut::before { border-top: 3px dashed var(--petrol); }
.sbx article > i { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 50%; background: var(--petrol); color: var(--ivory); font: 900 18px var(--font-read); font-style: normal; }
.sbx h3 { margin: 0; font-size: 20px; }
.sbx article p { margin: 2px 0 0; font-size: 13.5px; line-height: 1.3; color: var(--petrol); }
.sbx article p.sbx-who { color: var(--text-muted); font-size: 12.5px; }
.sbx-out { display: grid; grid-template-columns: 58px 1fr; column-gap: 12px; align-items: center; padding-left: 14px; border-left: 2px solid var(--line); }
.sbx-out b { justify-self: start; padding: 4px 10px; border-radius: var(--radius-sm); background: var(--orange); color: var(--petrol); font-size: 16px; font-weight: 900; }
.sbx article.shut .sbx-out b { background: var(--petrol); color: var(--ivory); }
</style>
