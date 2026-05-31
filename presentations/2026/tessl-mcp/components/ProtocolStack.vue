<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    showDescriptions?: boolean
  }>(),
  {
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

</script>

<template>
  <div class="protocol-stack">
    <div class="protocol-label protocol-label--server">
      <span>MCP Server</span>
    </div>

    <div class="protocol-grid protocol-grid--server">
      <ProtocolCapabilityCard
        v-for="item in capabilities.filter((capability) => capability.zone === 'server')"
        :key="item.id"
        :title="item.title"
        :icon="item.icon"
        :description="item.description"
        :show-description="props.showDescriptions"
      />
    </div>

    <div class="protocol-transport">
      <div class="protocol-transport__line" />
      <div class="protocol-transport__label">Transports</div>
    </div>

    <div class="protocol-grid protocol-grid--client">
      <ProtocolCapabilityCard
        v-for="item in capabilities.filter((capability) => capability.zone === 'client')"
        :key="item.id"
        :title="item.title"
        :icon="item.icon"
        :description="item.description"
        :show-description="props.showDescriptions"
      />
    </div>

    <div class="protocol-label protocol-label--client">
      <span>MCP Client</span>
    </div>
  </div>
</template>

<style scoped>
.protocol-stack {
  container-type: size;
  --stack-gap: clamp(0.5rem, 2.1cqh, 0.85rem);
  --protocol-card-pad: clamp(0.58rem, 2.2cqh, 0.88rem) clamp(0.72rem, 2.6cqw, 1.1rem);
  --protocol-card-content-gap: clamp(0.58rem, 2.4cqw, 0.98rem);
  --protocol-card-height: clamp(58px, 22cqh, 86px);
  --protocol-icon-size: clamp(2rem, 11cqh, 3rem);
  --protocol-title-size: clamp(1.08rem, 4.4cqh, 1.62rem);
  --protocol-description-size: clamp(0.48rem, 1.8cqh, 0.62rem);
  --protocol-label-height: clamp(44px, 14cqh, 64px);
  --protocol-zone-pad: 0;
  --protocol-transport-height: clamp(42px, 15cqh, 66px);
  width: 100%;
  height: 100%;
  min-height: 0;
  margin: 0;
  display: grid;
  grid-template-rows:
    var(--protocol-label-height)
    minmax(0, 1fr)
    var(--protocol-transport-height)
    minmax(0, 1fr)
    var(--protocol-label-height);
  gap: var(--stack-gap);
  position: relative;
}

.protocol-label,
.protocol-grid {
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 5px);
}

.protocol-label {
  min-width: 0;
  padding: 0 1.1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background:
    linear-gradient(90deg, rgba(245, 164, 0, 0.08), transparent 46%),
    rgba(20, 22, 27, 0.46);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.protocol-label--client {
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.065), transparent 46%),
    rgba(20, 22, 27, 0.42);
}

.protocol-label span {
  color: var(--deck-text);
  font-size: clamp(1rem, 4.2cqh, 1.42rem);
  font-weight: 850;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.protocol-grid {
  min-height: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--stack-gap);
  align-items: center;
  box-sizing: border-box;
  border-color: transparent;
}

.protocol-transport {
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 0.8rem;
}

.protocol-transport__line {
  grid-column: 1 / -1;
  grid-row: 1;
  height: 2px;
  background:
    linear-gradient(90deg, transparent, var(--deck-border-2) 22%, var(--deck-accent-line) 50%, var(--deck-border-2) 78%, transparent);
}

.protocol-transport__label {
  grid-column: 2;
  grid-row: 1;
  padding: clamp(0.24rem, 1cqh, 0.38rem) clamp(0.62rem, 2cqw, 0.9rem);
  border: 1px solid var(--deck-border-2);
  border-radius: 999px;
  background: color-mix(in srgb, var(--deck-bg) 86%, transparent);
  color: var(--deck-muted);
  font-size: clamp(0.52rem, 2cqh, 0.68rem);
  font-weight: 850;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.24);
}

</style>
