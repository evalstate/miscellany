<script setup lang="ts">
import { onMounted, ref } from "vue";

const video = ref<HTMLVideoElement>();

onMounted(() => {
  if (!video.value) return;
  video.value.muted = true;
  video.value.defaultMuted = true;
  video.value.play().catch(() => {});
});
</script>

<template>
  <video
    ref="video"
    class="hf-mcp-video"
    :src="'videos/dynamic-space-final.mp4'"
    autoplay
    loop
    muted
    playsinline
  />
</template>
