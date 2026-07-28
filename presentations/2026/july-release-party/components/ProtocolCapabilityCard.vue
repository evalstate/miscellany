<script setup lang="ts">
import { computed } from "vue";
import {
  FileText,
  FolderTree,
  MessageCircleQuestion,
  MessageSquare,
  Sparkles,
  Wrench,
} from "@lucide/vue";

const props = withDefaults(
  defineProps<{
    title: string;
    icon: "wrench" | "file" | "message" | "roots" | "sparkles" | "question";
    description?: string;
    showDescription?: boolean;
  }>(),
  {
    showDescription: true,
  },
);

const icons = {
  wrench: Wrench,
  file: FileText,
  message: MessageSquare,
  roots: FolderTree,
  sparkles: Sparkles,
  question: MessageCircleQuestion,
};

const Icon = computed(() => icons[props.icon]);
</script>

<template>
  <article class="protocol-card">
    <div class="protocol-card__content">
      <div class="protocol-card__icon">
        <component :is="Icon" :stroke-width="2.25" />
      </div>
      <div class="protocol-card__text">
        <h3>{{ title }}</h3>
        <p v-if="description && showDescription">{{ description }}</p>
      </div>
    </div>
  </article>
</template>

<style scoped>
.protocol-card {
  position: relative;
  min-height: 0;
  height: 100%;
  box-sizing: border-box;
  padding: var(--protocol-card-pad, 1rem);
  display: grid;
  align-items: center;
  border: 1px solid var(--deck-border);
  border-radius: calc(var(--deck-radius) + 4px);
  background:
    radial-gradient(
      circle at 82% 12%,
      rgba(245, 164, 0, 0.055),
      transparent 34%
    ),
    rgba(255, 255, 255, 0.84);
  box-shadow: 0 16px 34px rgba(0, 0, 0, 0.24);
  overflow: hidden;
  transition:
    opacity 220ms ease,
    transform 220ms ease,
    border-color 220ms ease,
    background 220ms ease,
    box-shadow 220ms ease;
}

.protocol-card__icon {
  color: var(--deck-muted);
  flex: none;
  transition:
    color 220ms ease,
    transform 220ms ease;
}

.protocol-card__content {
  display: flex;
  align-items: center;
  gap: var(--protocol-card-content-gap, 0.82rem);
  min-width: 0;
}

.protocol-card__icon :deep(svg) {
  width: var(--protocol-icon-size, 2.35rem);
  height: var(--protocol-icon-size, 2.35rem);
  display: block;
}

.protocol-card__text {
  min-width: 0;
}

.protocol-card h3 {
  margin: 0;
  color: var(--deck-text);
  font-size: var(--protocol-title-size, 1.18rem);
  line-height: 1.05;
  font-weight: 750;
  letter-spacing: -0.05em;
}

.protocol-card p {
  margin: 0.42rem 0 0;
  max-width: 19ch;
  color: var(--deck-dim);
  font-size: var(--protocol-description-size, 0.58rem);
  line-height: 1.35;
}
</style>
