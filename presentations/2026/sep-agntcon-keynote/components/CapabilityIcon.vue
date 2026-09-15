<script setup lang="ts">
import { FileText, FolderTree, MessageCircleQuestion, MessageSquare, Sparkles, Wrench } from '@lucide/vue';

defineProps<{ name: 'roots' | 'sampling' | 'elicitation' | 'tools' | 'resources' | 'prompts' }>();
const icons = {
  roots: FolderTree,
  sampling: Sparkles,
  elicitation: MessageCircleQuestion,
  tools: Wrench,
  resources: FileText,
  prompts: MessageSquare,
};
</script>

<template>
  <component :is="icons[name]" class="capability-icon" :stroke-width="2" aria-hidden="true" />
</template>
