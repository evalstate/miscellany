<script setup lang="ts">
import { computed } from 'vue'
import {
  FileText,
  FolderTree,
  MessageCircleQuestion,
  MessageSquare,
  Sparkles,
  Wrench,
} from '@lucide/vue'

const props = withDefaults(
  defineProps<{
    title: string
    icon: 'wrench' | 'file' | 'message' | 'roots' | 'sparkles' | 'question'
    description?: string
    emphasis?: 'none' | 'usage' | 'complexity'
    active?: boolean
    muted?: boolean
    metric?: string
    showDescription?: boolean
  }>(),
  {
    emphasis: 'none',
    active: false,
    muted: false,
    showDescription: true,
  },
)

const icons = {
  wrench: Wrench,
  file: FileText,
  message: MessageSquare,
  roots: FolderTree,
  sparkles: Sparkles,
  question: MessageCircleQuestion,
}

const Icon = computed(() => icons[props.icon])
</script>

<template>
  <article
    class="protocol-card"
    :class="[
      props.active && `protocol-card--active-${props.emphasis}`,
      props.muted && 'protocol-card--muted',
    ]"
  >
    <div class="protocol-card__content">
      <div class="protocol-card__icon">
        <component :is="Icon" :stroke-width="2.25" />
      </div>
      <div class="protocol-card__text">
        <h3>{{ title }}</h3>
        <p v-if="description && showDescription">{{ description }}</p>
      </div>
    </div>
    <div v-if="metric" class="protocol-card__metric">{{ metric }}</div>
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
    radial-gradient(circle at 82% 12%, rgba(245, 164, 0, 0.055), transparent 34%),
    rgba(20, 22, 27, 0.84);
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
  transition: color 220ms ease, transform 220ms ease;
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

.protocol-card__metric {
  position: absolute;
  top: 0.8rem;
  right: 0.85rem;
  padding: 0.16rem 0.45rem;
  border: 1px solid var(--deck-accent-line);
  border-radius: var(--deck-radius-sm);
  background: var(--deck-accent-bg);
  color: var(--deck-accent-hi);
  font-size: var(--protocol-metric-size, 0.54rem);
  font-weight: 750;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.protocol-card--muted {
  opacity: 0.36;
  filter: saturate(0.75);
}

.protocol-card--active-usage {
  transform: translateY(-3px);
  border-color: var(--deck-accent-line);
  background:
    radial-gradient(circle at 82% 12%, rgba(245, 164, 0, 0.20), transparent 38%),
    linear-gradient(180deg, rgba(245, 164, 0, 0.10), rgba(20, 22, 27, 0.88));
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.32), 0 0 32px rgba(245, 164, 0, 0.12);
}

.protocol-card--active-usage .protocol-card__icon,
.protocol-card--active-usage h3 {
  color: var(--deck-accent-hi);
}

.protocol-card--active-complexity {
  transform: translateY(-3px);
  border-color: var(--deck-info-line);
  background:
    radial-gradient(circle at 82% 12%, rgba(106, 163, 247, 0.18), transparent 38%),
    linear-gradient(180deg, rgba(106, 163, 247, 0.09), rgba(20, 22, 27, 0.88));
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.32), 0 0 32px rgba(106, 163, 247, 0.10);
}

.protocol-card--active-complexity .protocol-card__icon,
.protocol-card--active-complexity h3 {
  color: var(--deck-info);
}
</style>
