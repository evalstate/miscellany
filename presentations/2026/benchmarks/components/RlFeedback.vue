<script setup lang="ts">
import { computed, getCurrentInstance } from "vue";
import { useIsSlideActive } from "@slidev/client";

const props = withDefaults(defineProps<{ stage?: number }>(), { stage: 0 });
const training = computed(() => props.stage > 0);
const parallel = computed(() => props.stage > 1);
const active = useIsSlideActive();
const marker = `rlf-arrow-${getCurrentInstance()?.uid}`;
</script>

<template>
  <div class="rlf" :class="{ 'is-active': active, 'is-training': training }">
    <svg viewBox="0 0 868 344" role="img" :aria-label="training
      ? 'Agent trajectories and verifier rewards form a training batch. The learner updates the model weights, which feed the next rollouts.'
      : 'A model drives an agent in an environment. A verifier evaluates its work and produces a benchmark score.'">
      <defs>
        <marker :id="marker" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M1 1 L9 5 L1 9" class="rlf-arrowhead" />
        </marker>
      </defs>
      <g v-if="parallel" class="rlf-parallel">
        <rect x="218" y="38" width="248" height="88" rx="10" />
        <rect x="212" y="44" width="248" height="88" rx="10" />
      </g>
      <text x="330" y="22" text-anchor="middle" class="rlf-caption">{{ parallel ? 'Many tasks, parallel rollouts' : 'Task · harness · time limit' }}</text>

      <g class="rlf-node">
        <rect x="10" y="50" width="140" height="88" rx="10" />
        <text x="80" y="88" text-anchor="middle" class="rlf-title">Model</text>
        <text x="80" y="113" text-anchor="middle" class="rlf-small">current weights</text>
      </g>
      <g class="rlf-node">
        <rect x="206" y="50" width="248" height="88" rx="10" />
        <text x="330" y="88" text-anchor="middle" class="rlf-title">Agent + environment</text>
        <text x="330" y="113" text-anchor="middle" class="rlf-small">act · observe · continue</text>
      </g>
      <g class="rlf-node">
        <rect x="510" y="50" width="150" height="88" rx="10" />
        <text x="585" y="88" text-anchor="middle" class="rlf-title">Verifier</text>
        <text x="585" y="113" text-anchor="middle" class="rlf-small">check the work</text>
      </g>
      <g class="rlf-node rlf-result">
        <rect x="716" y="50" width="140" height="88" rx="10" />
        <text x="786" y="88" text-anchor="middle" class="rlf-title">{{ training ? 'Reward' : 'Score' }}</text>
        <text x="786" y="113" text-anchor="middle" class="rlf-small">{{ training ? 'training signal' : 'report the result' }}</text>
      </g>
      <g class="rlf-flow" :marker-end="`url(#${marker})`">
        <path d="M150 94 H197" />
        <path d="M454 94 H501" />
        <path d="M660 94 H707" />
      </g>

      <g v-if="training" class="rlf-training">
        <g class="rlf-flow rlf-flow-training" :marker-end="`url(#${marker})`">
          <path d="M330 138 V254 H395" />
          <path d="M786 138 V185 H487 V210" />
          <path d="M570 254 H633" />
          <path d="M734 290 V332 H80 V147" />
        </g>
        <text x="319" y="205" text-anchor="end" class="rlf-caption">Trajectory</text>
        <text x="319" y="226" text-anchor="end" class="rlf-small">actions + observations</text>
        <g class="rlf-node">
          <rect x="404" y="218" width="166" height="72" rx="10" />
          <text x="487" y="249" text-anchor="middle" class="rlf-title">Training batch</text>
          <text x="487" y="273" text-anchor="middle" class="rlf-small">traces + rewards</text>
        </g>
        <g class="rlf-node rlf-learner">
          <rect x="642" y="218" width="184" height="72" rx="10" />
          <text x="734" y="249" text-anchor="middle" class="rlf-title">Learner</text>
          <text x="734" y="273" text-anchor="middle" class="rlf-small">update the policy</text>
        </g>
        <text x="400" y="322" text-anchor="middle" class="rlf-caption">Updated model weights</text>
      </g>
      <text v-else x="434" y="245" text-anchor="middle" class="rlf-eval-note">The evaluation stops at a reported score.</text>
    </svg>
  </div>
</template>
