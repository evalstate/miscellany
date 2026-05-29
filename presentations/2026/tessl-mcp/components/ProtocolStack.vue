<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    emphasis?: 'none' | 'usage' | 'complexity'
    size?: 'sm' | 'md' | 'lg'
    showDescriptions?: boolean
  }>(),
  {
    emphasis: 'none',
    size: 'md',
    showDescriptions: false,
  },
)

const capabilities = [
  {
    id: 'tools',
    title: 'Tools',
    icon: 'wrench',
    description: 'Invoke actions in the outside world',
    zone: 'server',
  },
  {
    id: 'resources',
    title: 'Resources',
    icon: 'file',
    description: 'Expose context the model can read',
    zone: 'server',
  },
  {
    id: 'prompts',
    title: 'Prompts',
    icon: 'message',
    description: 'Package reusable instructions',
    zone: 'server',
  },
  {
    id: 'roots',
    title: 'Roots',
    icon: 'roots',
    description: 'Scope filesystem/project boundaries',
    zone: 'client',
  },
  {
    id: 'sampling',
    title: 'Sampling',
    icon: 'sparkles',
    description: 'Let servers request model turns',
    zone: 'client',
  },
  {
    id: 'elicitation',
    title: 'Elicitation',
    icon: 'question',
    description: 'Ask users for missing input',
    zone: 'client',
  },
] as const

const usageIds = new Set(['tools', 'resources', 'prompts'])
const complexityIds = new Set(['roots', 'sampling', 'elicitation'])

const annotation = computed(() => {
  if (props.emphasis === 'usage') {
    return {
      label: '~80% usage',
      text: 'Most real MCP integrations concentrate on server-side primitives.',
      tone: 'usage',
    }
  }
  if (props.emphasis === 'complexity') {
    return {
      label: '~80% complexity',
      text: 'The lower-traffic features carry much of the coordination burden.',
      tone: 'complexity',
    }
  }
  return null
})

function isActive(id: string) {
  if (props.emphasis === 'usage') return usageIds.has(id)
  if (props.emphasis === 'complexity') return complexityIds.has(id)
  return false
}

function isMuted(id: string) {
  return props.emphasis !== 'none' && !isActive(id)
}
</script>

<template>
  <div class="protocol-stack" :class="[`protocol-stack--${props.emphasis}`, `protocol-stack--${props.size}`]">
    <section class="protocol-box protocol-box--server" :class="props.emphasis === 'usage' && 'protocol-box--active-usage'">
      <header>
        <span>MCP Server</span>
        <strong v-if="props.emphasis === 'usage'">~80% usage</strong>
      </header>
      <div class="protocol-grid">
        <ProtocolCapabilityCard
          v-for="item in capabilities.filter((capability) => capability.zone === 'server')"
          :key="item.id"
          :title="item.title"
          :icon="item.icon"
          :description="item.description"
          :emphasis="props.emphasis"
          :active="isActive(item.id)"
          :muted="isMuted(item.id)"
          :metric="props.emphasis === 'usage' && isActive(item.id) ? 'high use' : undefined"
          :show-description="props.showDescriptions"
        />
      </div>
    </section>

    <aside v-if="annotation" class="protocol-annotation" :class="`protocol-annotation--${annotation.tone}`">
      <div class="protocol-annotation__label">{{ annotation.label }}</div>
      <p>{{ annotation.text }}</p>
    </aside>

    <section class="protocol-box protocol-box--client" :class="props.emphasis === 'complexity' && 'protocol-box--active-complexity'">
      <header>
        <span>MCP Client</span>
        <strong v-if="props.emphasis === 'complexity'">~80% complexity</strong>
      </header>
      <div class="protocol-grid">
        <ProtocolCapabilityCard
          v-for="item in capabilities.filter((capability) => capability.zone === 'client')"
          :key="item.id"
          :title="item.title"
          :icon="item.icon"
          :description="item.description"
          :emphasis="props.emphasis"
          :active="isActive(item.id)"
          :muted="isMuted(item.id)"
          :metric="props.emphasis === 'complexity' && isActive(item.id) ? 'hard part' : undefined"
          :show-description="props.showDescriptions"
        />
      </div>
    </section>
  </div>
</template>

<style scoped>
.protocol-stack {
  --stack-gap: 0.66rem;
  --protocol-card-pad: 0.78rem 1rem;
  --protocol-card-content-gap: 0.86rem;
  --protocol-card-height: 76px;
  --protocol-icon-size: 2.55rem;
  --protocol-title-size: 1.42rem;
  --protocol-description-size: 0.56rem;
  --protocol-box-header: 48px;
  --protocol-grid-pad: 0.75rem 1rem 1rem;
  width: min(1088px, 100%);
  height: var(--protocol-stack-height, 292px);
  margin: 0.9rem auto 0;
  display: grid;
  grid-template-rows: repeat(2, minmax(0, 1fr));
  gap: var(--stack-gap);
  position: relative;
}

.protocol-box {
  min-height: 0;
  padding: 0;
  display: grid;
  grid-template-rows: var(--protocol-box-header) minmax(0, 1fr);
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 5px);
  background:
    radial-gradient(circle at 90% 8%, rgba(245, 164, 0, 0.07), transparent 28%),
    rgba(11, 12, 15, 0.38);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
  overflow: hidden;
}

.protocol-box header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  min-width: 0;
  margin: 0;
  padding: 0 1.08rem;
  border-bottom: 1px solid color-mix(in srgb, var(--deck-border-2) 62%, transparent);
  background:
    linear-gradient(90deg, rgba(245, 164, 0, 0.055), transparent 46%),
    rgba(20, 22, 27, 0.32);
}

.protocol-box header span {
  color: var(--deck-text);
  font-size: 1.08rem;
  font-weight: 850;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.protocol-box header strong {
  color: var(--deck-accent-hi);
  font-size: 0.66rem;
  font-weight: 850;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.protocol-box--client {
  background:
    radial-gradient(circle at 90% 8%, rgba(106, 163, 247, 0.055), transparent 28%),
    rgba(11, 12, 15, 0.34);
}

.protocol-box--client header {
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.045), transparent 46%),
    rgba(20, 22, 27, 0.28);
}

.protocol-box--active-usage {
  border-color: var(--deck-accent-line);
  background:
    radial-gradient(circle at 90% 8%, rgba(245, 164, 0, 0.13), transparent 32%),
    rgba(245, 164, 0, 0.035);
}

.protocol-box--active-complexity {
  border-color: var(--deck-info-line);
  background:
    radial-gradient(circle at 90% 8%, rgba(106, 163, 247, 0.12), transparent 32%),
    rgba(106, 163, 247, 0.035);
}

.protocol-box--active-complexity header strong {
  color: var(--deck-info);
}

.protocol-grid {
  min-height: 0;
  padding: var(--protocol-grid-pad);
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--stack-gap);
  align-items: center;
  box-sizing: border-box;
}

.protocol-annotation {
  position: absolute;
  z-index: 3;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  justify-self: center;
  width: min(560px, 100%);
  padding: 0.38rem 0.75rem;
  border: 1px solid var(--deck-accent-line);
  border-radius: var(--deck-radius);
  background: color-mix(in srgb, var(--deck-bg) 82%, transparent);
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.24);
  text-align: center;
}

.protocol-annotation__label {
  color: var(--deck-accent-hi);
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.protocol-annotation p {
  margin: 0.18rem 0 0;
  color: var(--deck-muted);
  font-size: 0.62rem;
  line-height: 1.35;
}

.protocol-annotation--complexity {
  border-color: var(--deck-info-line);
}

.protocol-annotation--complexity .protocol-annotation__label {
  color: var(--deck-info);
}

.protocol-stack--sm {
  --protocol-stack-height: 246px;
  --stack-gap: 0.56rem;
  --protocol-card-pad: 0.66rem 0.78rem;
  --protocol-card-content-gap: 0.68rem;
  --protocol-card-height: 64px;
  --protocol-icon-size: 2rem;
  --protocol-title-size: 1.12rem;
  --protocol-description-size: 0.48rem;
  --protocol-box-header: 42px;
  --protocol-grid-pad: 0.6rem 0.72rem 0.76rem;
}

.protocol-stack--md {
  --protocol-stack-height: 292px;
}

.protocol-stack--lg {
  --protocol-stack-height: 342px;
  --stack-gap: 0.84rem;
  --protocol-card-pad: 0.95rem 1.18rem;
  --protocol-card-content-gap: 1rem;
  --protocol-card-height: 92px;
  --protocol-icon-size: 3rem;
  --protocol-title-size: 1.68rem;
  --protocol-description-size: 0.62rem;
  --protocol-box-header: 56px;
  --protocol-grid-pad: 0.92rem 1.12rem 1.12rem;
}

.protocol-stack--usage,
.protocol-stack--complexity {
  --protocol-stack-height: 292px;
}
</style>
