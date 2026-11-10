<script setup lang="ts">
type Example = {
  model: string;
  initials: string;
  lead: string;
  quote: string;
  action: string;
  task: string;
  step: number;
  date: string;
  harness: string;
};

defineProps<{ examples: Example[] }>();
</script>

<template>
  <div
    class="reward-chat"
    aria-label="Three models finding benchmark reference material"
  >
    <article
      v-for="example in examples"
      :key="example.model"
      class="reward-chat-thread"
    >
      <header class="reward-chat-person">
        <span class="reward-chat-avatar" aria-hidden="true">{{
          example.initials
        }}</span>
        <h3>{{ example.model }}</h3>
      </header>

      <blockquote
        class="reward-chat-bubble"
        :aria-label="`${example.model}, recorded reasoning`"
      >
        <strong>{{ example.lead }}</strong>
        <span>{{ example.quote }}</span>
        <cite>Reasoning · step {{ example.step }}</cite>
      </blockquote>

      <div class="reward-chat-action">
        <span class="reward-chat-action-label">Observed action</span>
        <p>{{ example.action }}</p>
      </div>

      <p class="reward-chat-source">
        <span>{{ example.task }} · {{ example.date }}</span>
        <span>{{ example.harness }}</span>
      </p>
    </article>
  </div>
</template>

<style scoped>
.reward-chat {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}
.reward-chat-thread {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.reward-chat-person {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}
.reward-chat-avatar {
  display: grid;
  place-items: center;
  flex: 0 0 32px;
  height: 32px;
  border: 2px solid var(--petrol);
  border-radius: 8px;
  background: var(--petrol);
  color: var(--ivory);
  font: 800 14px/1 var(--font-read);
}
.reward-chat-person h3 {
  margin: 0;
  font: 800 19px/1.1 var(--font-read);
  letter-spacing: -0.02em;
}
.reward-chat-bubble {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 160px;
  margin: 0 0 18px;
  padding: 14px 16px 12px;
  border: 2px solid var(--petrol);
  border-radius: 14px 14px 14px 3px;
  background: var(--paper);
  color: var(--petrol);
  font: 500 20px/1.24 var(--font-read);
  box-shadow: none;
}
.reward-chat-bubble::after {
  content: "";
  position: absolute;
  bottom: -8px;
  left: 16px;
  width: 12px;
  height: 12px;
  border-left: 2px solid var(--petrol);
  border-bottom: 2px solid var(--petrol);
  background: var(--paper);
  transform: rotate(-45deg);
}
.reward-chat-bubble strong {
  display: block;
  margin-bottom: 6px;
  font-size: 22px;
  line-height: 1.1;
  font-weight: 900;
}
.reward-chat-bubble cite {
  display: block;
  margin-top: auto;
  padding-top: 7px;
  color: var(--text-muted);
  font: 500 12px/1.2 var(--font-read);
}
.reward-chat-action {
  flex: 1;
  min-height: 96px;
  margin-left: 16px;
  padding: 10px 12px;
  border: 2px solid var(--petrol);
  border-radius: 14px 3px 14px 14px;
  background: var(--petrol);
  color: var(--ivory);
}
.reward-chat-action-label {
  display: block;
  margin-bottom: 4px;
  font: 800 12px/1.1 var(--font-read);
}
.reward-chat-action p {
  margin: 0;
  font: 500 16px/1.25 var(--font-read);
}
.reward-chat-source {
  display: flex;
  flex-direction: column;
  margin: 10px 0 0;
  color: var(--text-muted);
  font: 500 12px/1.3 var(--font-read);
}
:global(.slidev-layout.fa-reward-chat-slide) {
  padding-top: 30px;
}
:global(.fa-reward-chat-slide h1) {
  margin-bottom: 16px;
  font-size: 46px;
}
:global(.fa-reward-chat-slide .fa-receipt) {
  margin-top: 10px;
  padding-top: 8px;
}
</style>
