<script setup lang="ts">
// "Broken task, perfect score": what the task shipped (no .git), then the reward
// by model, and the two answers side by side.
// Figures: talk-hunt-2026-11-08 defective-task report. No trial IDs here.
const groups = [
  { model: "DeepSeek", note: "found the answer key", reward: "1.0", n: 2, kind: "full" },
  { model: "DeepSeek", note: "searched, found no key", reward: "0", n: 1, kind: "zero" },
  { model: "GPT-6 Luna", note: "said the repo was missing", reward: "0", n: 6, kind: "zero" },
  { model: "Qwen3.8-27B", note: "guessed", reward: "0–0.35", n: 26, kind: "part" },
];
</script>

<template>
  <div class="bt">
    <div class="bt-left">
      <div class="bt-dir" aria-label="What the task shipped">
        <div class="bt-dir-head"><span>git-repo-forensics</span><span aria-hidden="true">❯_</span></div>
        <div class="bt-files">
          <div><span class="bt-prompt">❯ </span>ls -a /workdir/data</div>
          <div class="bt-out">config.json&nbsp;&nbsp;database.sql</div>
          <div class="bt-ghost"><span>.git/</span><em>never shipped</em></div>
          <div><span class="bt-prompt">❯ </span>git log</div>
          <div class="bt-err">git: command not found</div>
        </div>
      </div>

      <div class="bt-trials" aria-label="Reward by model, 35 trials">
        <div v-for="g in groups" :key="g.model + g.note" class="bt-row">
          <span class="bt-reward" :class="{ 'bt-reward--full': g.kind === 'full' }">{{ g.reward }}</span>
          <span class="bt-label"><b>{{ g.model }}</b> {{ g.note }}</span>
          <span class="bt-n">{{ g.n }} {{ g.n === 1 ? "trial" : "trials" }}</span>
        </div>
      </div>
    </div>

    <div class="bt-right">
      <figure class="bt-say">
        <blockquote>The <code>.git</code> directory was stripped when the task was packaged … which is why the repo appears ‘missing.’ The task's own reference solution includes fallback values computed from the original fixture.</blockquote>
        <figcaption>
          <span><b>DeepSeek V4.1 Flash</b> · step 111</span>
          <strong class="bt-score bt-score--full">reward 1.0</strong>
        </figcaption>
      </figure>

      <figure class="bt-say bt-say--right">
        <blockquote>The Git repository metadata and objects are absent, so commit counts and recovery details are marked <code>null</code> rather than guessed.</blockquote>
        <figcaption>
          <span><b>GPT-6 Luna</b> · step 11</span>
          <strong class="bt-score">reward 0</strong>
        </figcaption>
      </figure>
    </div>
  </div>
</template>

<style scoped>
.bt {
  display: grid;
  grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
  gap: 32px;
  align-items: start;
}

/* the directory listing */
.bt-dir {
  border: var(--border);
  border-radius: var(--radius-md);
  background: var(--petrol);
  color: var(--ivory);
  overflow: hidden;
}
.bt-dir-head {
  display: flex;
  justify-content: space-between;
  padding: 8px 16px;
  border-bottom: 2px solid var(--petrol-2);
  font: 500 14px var(--font-code);
  color: rgba(255, 247, 232, 0.78);
}
.bt-files {
  padding: 6px 16px 10px;
  font: 500 17px/1.4 var(--font-code);
}
.bt-prompt { color: var(--amber); }
.bt-out { color: rgba(255, 247, 232, 0.78); }
.bt-err { color: var(--danger); font-weight: 700; }
.bt-ghost {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 6px 0 8px;
  padding: 4px 12px;
  border: 2px dashed var(--danger);
  border-radius: var(--radius-sm);
}
.bt-ghost span { text-decoration: line-through; text-decoration-thickness: 2px; color: var(--danger); }
.bt-ghost em {
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  background: var(--danger);
  color: var(--petrol);
  font: 800 15px/1.2 var(--font-read);
  font-style: normal;
}

/* outcomes */
.bt-trials { margin-top: 10px; display: grid; }
.bt-row {
  display: grid;
  grid-template-columns: 78px minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
  padding: 3px 0;
  border-bottom: 2px solid var(--line);
}
.bt-row:last-child { border-bottom: 0; }
.bt-reward {
  justify-self: start;
  padding: 3px 9px;
  border: 2px solid transparent;
  border-radius: var(--radius-sm);
  font: 800 17px/1.1 var(--font-read);
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
  color: var(--petrol);
}
.bt-reward--full { background: var(--petrol); color: var(--ivory); }
.bt-label { white-space: nowrap; font: 500 15px/1.25 var(--font-read); color: var(--text-muted); }
.bt-label b { font-weight: 700; color: var(--petrol); margin-right: 4px; }
.bt-n { font: 500 14px var(--font-read); color: var(--text-muted); font-variant-numeric: tabular-nums; }

/* the two answers */
.bt-right { display: grid; gap: 14px; }
.bt-say { margin: 0; }
.bt-say blockquote {
  position: relative;
  margin: 0;
  padding: 12px 18px;
  border: var(--border);
  border-radius: var(--radius-lg);
  background: var(--paper);
  font: italic 500 18px/1.35 var(--font-read);
  color: var(--petrol);
}
.bt-say blockquote::before { content: "“"; }
.bt-say blockquote::after { content: "”"; }
.bt-say code { font-style: normal; font-size: 0.9em; }
.bt-say figcaption {
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 12px 0 0 34px;
  font: 500 14px/1.3 var(--font-read);
  color: var(--text-muted);
}
.bt-say figcaption::before {
  content: "";
  position: absolute;
  left: 24px;
  top: -9px;
  width: 14px;
  height: 14px;
  background: var(--paper);
  border-right: var(--border);
  border-bottom: var(--border);
  transform: rotate(45deg);
}
.bt-say figcaption b { color: var(--petrol); font-weight: 700; }
.bt-score {
  flex: none;
  padding: 5px 12px;
  border: var(--border);
  border-radius: var(--radius-sm);
  background: var(--paper);
  color: var(--petrol);
  font: 800 18px/1 var(--font-read);
  font-variant-numeric: tabular-nums;
}
.bt-score--full { background: var(--petrol); color: var(--ivory); }
</style>
