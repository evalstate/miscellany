<script setup lang="ts">
type Example = {
  model: string;
  lead: string;
  quote: string;
  action: string;
  receipt: string;
  channel: string;
};
defineProps<{ examples: Example[] }>();
</script>

<template>
  <div
    class="hack-collage"
    aria-label="Six recorded reactions to finding benchmark reference material"
  >
    <article
      v-for="(example, i) in examples"
      :key="i"
      class="hack-collage-note"
      :class="[
        `hack-collage-note-${i}`,
        { 'hack-collage-inverse': i === 1 || i === 5 },
      ]"
    >
      <div class="hack-collage-bubble">
        <h3>{{ example.model }}</h3>
        <blockquote>
          <strong>{{ example.lead }}</strong>
          <span>{{ example.quote }}</span>
        </blockquote>
      </div>
      <p class="hack-collage-action">{{ example.action }}</p>
      <p class="hack-collage-receipt">{{ example.receipt }}</p>
    </article>
  </div>
</template>

<style scoped>
.hack-collage {
  position: relative;
  height: 365px;
}
.hack-collage-note {
  position: absolute;
  width: 31%;
}
.hack-collage-note-0 {
  left: 0;
  top: 0;
}
.hack-collage-note-1 {
  left: 35%;
  top: 14px;
  width: 29%;
}
.hack-collage-note-2 {
  right: 0;
  top: 2px;
  width: 31%;
}
.hack-collage-note-3 {
  left: 2%;
  top: 183px;
  width: 31%;
}
.hack-collage-note-4 {
  left: 37%;
  top: 196px;
  width: 29%;
}
.hack-collage-note-5 {
  right: 0;
  top: 180px;
  width: 30%;
}
.hack-collage-bubble {
  position: relative;
  padding: 12px 14px;
  border: 2px solid var(--petrol);
  border-radius: 16px 16px 16px 3px;
  background: var(--paper);
  color: var(--petrol);
}
.hack-collage-bubble::after {
  content: "";
  position: absolute;
  bottom: -7px;
  left: 18px;
  width: 10px;
  height: 10px;
  border-left: 2px solid var(--petrol);
  border-bottom: 2px solid var(--petrol);
  background: var(--paper);
  transform: rotate(-45deg);
}
.hack-collage-inverse .hack-collage-bubble {
  background: var(--petrol);
  color: var(--ivory);
  border-radius: 16px 16px 3px 16px;
}
.hack-collage-inverse .hack-collage-bubble::after {
  left: auto;
  right: 18px;
  background: var(--petrol);
}
.hack-collage-bubble h3 {
  margin: 0 0 7px;
  font: 800 16px/1.1 var(--font-read);
  letter-spacing: -0.015em;
}
.hack-collage-bubble blockquote {
  margin: 0;
  padding: 0;
  border: none;
  background: none;
  color: inherit;
  font: 500 16px/1.2 var(--font-read);
}
.hack-collage-bubble strong {
  display: block;
  margin-bottom: 4px;
  font: 900 21px/1.1 var(--font-read);
}
.hack-collage-action {
  margin: 10px 0 3px;
  font: 700 14px/1.2 var(--font-read);
}
.hack-collage-receipt {
  margin: 0;
  color: var(--text-muted);
  font: 500 12px/1.2 var(--font-read);
}
:global(.slidev-layout.fa-hack-collage-slide) {
  padding-top: 28px;
}
:global(.fa-hack-collage-slide .fa-receipt) {
  margin-top: 7px;
  padding-top: 7px;
}
</style>
