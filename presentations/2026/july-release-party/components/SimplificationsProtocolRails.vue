<script setup lang="ts">
import {
  FileText,
  FolderTree,
  MessageCircleQuestion,
  MessageSquare,
  Sparkles,
  Wrench,
} from "@lucide/vue";

const server = [
  { title: "Tools", icon: Wrench },
  { title: "Resources", icon: FileText },
  { title: "Prompts", icon: MessageSquare },
];

const client = [
  { title: "Roots", icon: FolderTree, deprecated: true },
  { title: "Sampling", icon: Sparkles, deprecated: true },
  { title: "Elicitation", icon: MessageCircleQuestion },
];
</script>

<template>
  <section class="simplified-rails" aria-label="Simplified MCP capabilities">
    <div class="cap-row">
      <article v-for="item in server" :key="item.title" class="cap-slab">
        <component :is="item.icon" :stroke-width="2.2" />
        <strong>{{ item.title }}</strong>
      </article>
    </div>

    <div class="actor-rail actor-rail--server">MCP Server</div>

    <div class="message-rails" aria-hidden="true">
      <div class="message-rail" />
      <div class="message-rail" />
    </div>

    <div class="actor-rail actor-rail--client">MCP Client</div>

    <div class="cap-row">
      <article
        v-for="item in client"
        :key="item.title"
        class="cap-slab"
        :class="{ 'cap-slab--deprecated': item.deprecated }"
      >
        <component :is="item.icon" :stroke-width="2.2" />
        <strong>{{ item.title }}</strong>
        <small v-if="item.deprecated">deprecated</small>
      </article>
    </div>
  </section>
</template>

<style scoped>
.simplified-rails {
  container-type: size;
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-rows: 1fr 0.46fr 0.36fr 0.46fr 1fr;
  gap: clamp(0.24rem, 1.75cqh, 0.5rem);
  color: var(--deck-text);
}

.cap-row {
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: clamp(0.35rem, 1.8cqw, 0.72rem);
}

.cap-slab,
.actor-rail {
  min-width: 0;
  min-height: 0;
  border: 1px solid var(--deck-border-2);
  border-radius: calc(var(--deck-radius) + 6px);
  background:
    radial-gradient(circle at 78% 18%, rgba(255, 198, 73, 0.06), transparent 34%),
    rgba(255, 255, 255, 0.62);
}

.cap-slab {
  position: relative;
  padding: clamp(0.34rem, 2.2cqh, 0.68rem) clamp(0.42rem, 1.6cqw, 0.78rem);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: clamp(0.32rem, 1.4cqw, 0.58rem);
  overflow: hidden;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.2);
}

.cap-slab svg {
  flex: none;
  width: clamp(1rem, 8cqh, 1.72rem);
  height: clamp(1rem, 8cqh, 1.72rem);
  color: var(--deck-muted);
}

.cap-slab strong {
  min-width: 0;
  color: var(--deck-text);
  font-size: clamp(0.68rem, 4.4cqh, 1.02rem);
  line-height: 1;
  letter-spacing: -0.045em;
  white-space: nowrap;
}

.cap-slab--deprecated {
  border-color: rgba(240, 107, 90, 0.66);
  background:
    linear-gradient(135deg, rgba(240, 107, 90, 0.1), rgba(255, 255, 255, 0.52)),
    rgba(255, 255, 255, 0.5);
}

.cap-slab--deprecated svg,
.cap-slab--deprecated strong {
  color: color-mix(in srgb, var(--deck-no) 82%, var(--deck-text));
}

.cap-slab small {
  position: absolute;
  right: 0.38rem;
  bottom: 0.28rem;
  padding: 0.08rem 0.24rem;
  color: color-mix(in srgb, var(--deck-no) 82%, var(--deck-text));
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(240, 107, 90, 0.36);
  border-radius: 999px;
  font: 700 clamp(0.36rem, 2.2cqh, 0.5rem)/1 var(--deck-font-mono);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.actor-rail {
  padding: 0 clamp(0.58rem, 2cqw, 0.92rem);
  display: flex;
  align-items: center;
  color: var(--deck-text);
  font-size: clamp(0.62rem, 3.2cqh, 0.86rem);
  font-weight: 850;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.actor-rail--server {
  background:
    linear-gradient(90deg, rgba(245, 164, 0, 0.08), transparent 50%),
    rgba(255, 255, 255, 0.54);
}

.actor-rail--client {
  background:
    linear-gradient(90deg, rgba(106, 163, 247, 0.07), transparent 50%),
    rgba(255, 255, 255, 0.54);
}

.message-rails {
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: center;
  gap: clamp(0.62rem, 3.5cqw, 1.35rem);
  padding-inline: 15%;
}

.message-rail {
  position: relative;
  height: clamp(2px, 1.4cqh, 4px);
  border-radius: 999px;
  background: linear-gradient(90deg, transparent, rgba(185, 179, 165, 0.38), transparent);
}
</style>
