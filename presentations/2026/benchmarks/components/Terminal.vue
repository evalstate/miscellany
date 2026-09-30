<script setup lang="ts">
// Petrol terminal panel (homepage "Get started" style). Lines starting with
// "$ " render as commands with the ❯ prompt; other lines are output.
withDefaults(defineProps<{ lines: string[]; title?: string }>(), { title: "terminal" });
</script>

<template>
  <div class="fa-terminal">
    <div class="fa-terminal-heading"><span>{{ title }}</span><span aria-hidden="true">❯_</span></div>
    <pre><code><template v-for="(line, i) in lines" :key="i"><span v-if="line.startsWith('$ ')" class="cmd"><span class="prompt">❯ </span>{{ line.slice(2) }}</span><span v-else class="out">{{ line }}</span>
</template></code></pre>
  </div>
</template>

<style scoped>
.fa-terminal {
  background: var(--petrol);
  color: var(--ivory);
  border: var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
}
.fa-terminal-heading {
  display: flex;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 2px solid var(--petrol-2);
  font: 500 13px var(--font-code);
  color: rgba(255, 247, 232, 0.72);
}
.fa-terminal pre {
  margin: 0;
  padding: 18px 20px;
  border: 0;
  border-radius: 0;
  background: transparent !important;
  font: 500 17px/1.6 var(--font-code);
  white-space: pre-wrap;
}
.fa-terminal .prompt { color: var(--amber); }
.fa-terminal .out { color: rgba(255, 247, 232, 0.72); }
</style>
