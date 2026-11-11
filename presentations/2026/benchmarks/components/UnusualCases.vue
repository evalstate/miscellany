<script setup lang="ts">
// Four reviewed lookups that don't fit the usual pattern. Each card says what
// the task was, what the agent did, and how it ended. Sources: atif-scan's
// task-named-skill check on the 65 public TB 2.1 runs; talk-hunt wildcard-02,
// wildcard-03 and wildcard-06.
const cases = [
  { title: "A skill named after the task",
    task: "Skills are how-to files an agent can install. Some public ones are named after Terminal-Bench tasks.",
    did: "13 trials fetched the one for their own task. One held the task's one-line fix, word for word.",
    who: "Grok 4.5 on Cursor CLI, SWE-1.7 on Devin CLI", end: "11 of 13 rewarded" },
  { title: "221 MB of other agents' attempts",
    task: "Task: write a small parser for a binary file.",
    did: "It downloads 221 MB of other agents' recorded runs of the benchmark, then the official solution and checker.",
    quote: "These will tell me exactly what the reference produces and how it's graded.",
    who: "DeepSeek V4 Flash · 10 Aug", end: "Reward 1" },
  { title: "It stops its own password cracker",
    task: "Task: recover an archive's password. The archive it was given is empty, 0 bytes.",
    did: "It fetches the real archive from a public copy of the task and starts cracking. Then it finds the password in the published solution.",
    quote: "Let me stop the slow crack and extract properly.",
    who: "DeepSeek V4 Flash · 9 Aug", end: "Reward 1" },
  { title: "It looks itself up",
    task: "Task: fix a training bug. The prompt says not to touch one compatibility file.",
    did: "After its context is compacted it overwrites that file. To undo it, it searches its own session log for its first read of the file.",
    quote: "I'll recover that forward from the session history and put it back.",
    who: "Grok 4.7 · 1 Nov", end: "Reward 0" },
];
</script>

<template>
  <div class="uc">
    <article v-for="c in cases" :key="c.title">
      <h3>{{ c.title }}</h3>
      <p class="uc-task">{{ c.task }}</p>
      <p>{{ c.did }}</p>
      <p v-if="c.quote" class="uc-quote">{{ c.quote }}</p>
      <footer><span>{{ c.who }}</span><b>{{ c.end }}</b></footer>
    </article>
  </div>
</template>

<style scoped>
.uc { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; align-items: stretch; font-family: var(--font-read); }
.uc article { display: flex; flex-direction: column; min-width: 0; padding: 14px 14px 12px; border: var(--border); border-radius: var(--radius-lg); background: var(--paper); }
.uc h3 { margin: 0 0 8px; font-size: 18px; line-height: 1.12; }
.uc article p { margin: 0 0 7px; font-size: 13px; line-height: 1.35; color: var(--petrol); }
.uc article p.uc-task { padding: 6px 9px; border-radius: var(--radius-sm); background: var(--ivory-deep); font-size: 12.5px; }
.uc article p.uc-quote { font-style: italic; }
.uc-quote::before { content: "“"; }
.uc-quote::after { content: "”"; }
.uc footer { display: flex; flex-direction: column; gap: 4px; margin-top: auto; padding-top: 8px; border-top: 2px solid var(--line); font-size: 12px; color: var(--text-muted); }
.uc footer b { font-size: 14px; font-weight: 800; color: var(--petrol); }
</style>
