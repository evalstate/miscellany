<script setup lang="ts">
import ProtocolCapabilityCard from "./ProtocolCapabilityCard.vue";

const server = [
  { id: "tools", title: "Tools", icon: "wrench" },
  { id: "resources", title: "Resources", icon: "file" },
  { id: "prompts", title: "Prompts", icon: "message" },
] as const;

const client = [
  { id: "roots", title: "Roots", icon: "roots", deprecated: true },
  { id: "sampling", title: "Sampling", icon: "sparkles", deprecated: true },
  { id: "elicitation", title: "Elicitation", icon: "question", deprecated: false },
] as const;
</script>

<template>
  <section
    class="deprecation-map"
    aria-label="MCP capabilities with Roots, Sampling, and Logging marked deprecated"
  >
    <div class="protocol-grid protocol-grid--server">
      <ProtocolCapabilityCard
        v-for="item in server"
        :key="item.id"
        class="protocol-card-shell"
        :title="item.title"
        :icon="item.icon"
        :show-description="false"
      />
    </div>

    <div class="protocol-label protocol-label--server">
      <span>MCP Server</span>
      <small class="deprecated-pill">Logging · deprecated</small>
    </div>

    <div class="protocol-process-gap" aria-hidden="true">
      <div class="protocol-arrow-pair">
        <div class="protocol-block-arrow protocol-block-arrow--up" />
        <div class="protocol-block-arrow protocol-block-arrow--down" />
      </div>
      <div class="protocol-operation">
        <span>modern · 2026-07-28</span>
        <strong>Roots + Sampling remain available through MRTR</strong>
      </div>
    </div>

    <div class="protocol-label protocol-label--client">
      <span>MCP Client</span>
      <small>capabilities</small>
    </div>

    <div class="protocol-grid protocol-grid--client">
      <div
        v-for="item in client"
        :key="item.id"
        class="protocol-card-wrap"
        :class="{ 'is-deprecated': item.deprecated }"
      >
        <ProtocolCapabilityCard
          class="protocol-card-shell"
          :title="item.title"
          :icon="item.icon"
          :show-description="false"
        />
        <small v-if="item.deprecated" class="card-deprecated-badge">
          deprecated
        </small>
      </div>
    </div>
  </section>
</template>

<style scoped>
.deprecation-map {
  container-type: size;
  --stack-gap: clamp(0.4rem, 1.55cqh, 0.68rem);
  --protocol-card-pad: clamp(0.52rem, 1.8cqh, 0.78rem)
    clamp(0.7rem, 2.4cqw, 1rem);
  --protocol-card-content-gap: clamp(0.62rem, 2.45cqw, 1rem);
  --protocol-card-height: clamp(54px, 18.5cqh, 76px);
  --protocol-icon-size: clamp(1.82rem, 8.8cqh, 2.6rem);
  --protocol-title-size: clamp(1rem, 3.8cqh, 1.38rem);
  --protocol-description-size: clamp(0.48rem, 1.8cqh, 0.62rem);
  --protocol-label-height: clamp(40px, 11.5cqh, 56px);
  width: 100%;
  height: 100%;
  min-height: 0;
  display: grid;
  grid-template-rows:
    minmax(var(--protocol-card-height), 1fr)
    var(--protocol-label-height)
    clamp(52px, 16cqh, 70px)
    var(--protocol-label-height)
    minmax(var(--protocol-card-height), 1fr);
  gap: var(--stack-gap);
  position: relative;
  color: var(--deck-text);
}

.protocol-label,
.protocol-grid {
  position: relative;
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 5px);
}

.protocol-grid {
  z-index: 2;
  min-height: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--stack-gap);
  align-items: center;
  border-color: transparent;
}

.protocol-grid--server {
  margin-bottom: calc(var(--stack-gap) * -1.2);
  padding-bottom: calc(var(--stack-gap) * 0.8);
}

.protocol-grid--client {
  margin-top: calc(var(--stack-gap) * -1.2);
  padding-top: calc(var(--stack-gap) * 0.8);
}

.protocol-label {
  z-index: 3;
  min-width: 0;
  padding: 0 1.1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background:
    linear-gradient(90deg, rgba(245, 164, 0, 0.08), transparent 46%),
    rgba(255, 255, 255, 0.46);
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
}

.protocol-label--client {
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.065), transparent 46%),
    rgba(255, 255, 255, 0.42);
}

.protocol-label > span {
  color: var(--deck-text);
  font-size: clamp(1rem, 4.2cqh, 1.42rem);
  font-weight: 850;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.protocol-label > small {
  color: var(--deck-dim);
  font-size: clamp(0.48rem, 1.7cqh, 0.62rem);
  font-weight: 850;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.protocol-label > .deprecated-pill {
  padding: 0.22rem 0.4rem;
  color: color-mix(in srgb, var(--deck-no) 86%, var(--deck-text));
  border: 1px solid rgba(240, 107, 90, 0.5);
  border-radius: 999px;
  background: rgba(254, 242, 242, 0.86);
}

.protocol-process-gap {
  position: relative;
  z-index: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(200px, 0.48fr);
  align-items: center;
  gap: clamp(1rem, 3.2cqw, 2rem);
  background:
    linear-gradient(90deg, transparent, rgba(185, 179, 165, 0.055), transparent)
      50% 50% / 46% 1px no-repeat;
}

.protocol-arrow-pair {
  position: absolute;
  top: calc(-1 * (var(--protocol-label-height) + var(--stack-gap)));
  left: 47%;
  z-index: 1;
  display: flex;
  justify-content: center;
  gap: clamp(1rem, 2.4cqw, 1.42rem);
  height: calc(
    100% + var(--protocol-label-height) + var(--protocol-label-height) +
      var(--stack-gap) + var(--stack-gap)
  );
  transform: translateX(-50%);
}

.protocol-block-arrow {
  position: relative;
  width: clamp(36px, 4.8cqw, 56px);
  height: 100%;
  opacity: 0.68;
}

.protocol-block-arrow::before {
  content: "";
  position: absolute;
  top: 4%;
  bottom: 4%;
  left: 50%;
  width: 4px;
  border-radius: 999px;
  background: rgba(185, 179, 165, 0.3);
  box-shadow:
    0 0 0 1px rgba(185, 179, 165, 0.16),
    0 0 12px rgba(185, 179, 165, 0.04);
  transform: translateX(-50%);
}

.protocol-block-arrow::after {
  content: "";
  position: absolute;
  left: 50%;
  width: 0;
  height: 0;
  border-inline: 8px solid transparent;
  transform: translateX(-50%);
  opacity: 0.72;
}

.protocol-block-arrow--up::after {
  top: 1%;
  border-bottom: 12px solid rgba(185, 179, 165, 0.3);
}

.protocol-block-arrow--down::after {
  bottom: 1%;
  border-top: 12px solid rgba(185, 179, 165, 0.3);
}

.protocol-operation {
  grid-column: 2;
  z-index: 2;
  min-width: 0;
  padding: 0.52rem 0.62rem;
  border: 1px solid var(--deck-border);
  border-radius: var(--deck-radius-sm);
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.1);
}

.protocol-operation span,
.protocol-operation strong {
  display: block;
}

.protocol-operation span {
  color: var(--deck-accent-hi);
  font: 800 clamp(0.42rem, 1.8cqh, 0.54rem)/1 var(--deck-font-mono);
  letter-spacing: 0.1em;
}

.protocol-operation strong {
  margin-top: 0.26rem;
  color: var(--deck-text);
  font-size: clamp(0.58rem, 2.4cqh, 0.72rem);
  line-height: 1.12;
}

.protocol-card-wrap {
  position: relative;
  min-width: 0;
  height: 100%;
}

.protocol-card-shell {
  width: 100%;
  height: 100%;
}

.protocol-card-wrap.is-deprecated .protocol-card-shell {
  border-color: rgba(240, 107, 90, 0.58);
  background:
    radial-gradient(circle at 82% 12%, rgba(240, 107, 90, 0.08), transparent 34%),
    rgba(254, 242, 242, 0.82);
  box-shadow: 0 16px 34px rgba(127, 29, 29, 0.12);
}

.protocol-card-wrap.is-deprecated
  .protocol-card-shell
  :deep(.protocol-card__icon),
.protocol-card-wrap.is-deprecated
  .protocol-card-shell
  :deep(h3) {
  color: color-mix(in srgb, var(--deck-no) 82%, var(--deck-text));
}

.card-deprecated-badge {
  position: absolute;
  right: 0.42rem;
  bottom: 0.34rem;
  padding: 0.12rem 0.28rem;
  color: color-mix(in srgb, var(--deck-no) 86%, var(--deck-text));
  border: 1px solid rgba(240, 107, 90, 0.42);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.82);
  font: 800 clamp(0.34rem, 1.4cqh, 0.46rem)/1 var(--deck-font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
</style>
