<script setup lang="ts">
import { onMounted, ref } from "vue";
import { onSlideEnter, onSlideLeave } from "@slidev/client";

const video = ref<HTMLVideoElement>();

function prepareVideo() {
  if (!video.value) return;
  video.value.muted = true;
  video.value.defaultMuted = true;
}

function playFromStart() {
  if (!video.value) return;
  prepareVideo();
  video.value.currentTime = 0;
  video.value.play().catch(() => {});
}

function stopAndReset() {
  if (!video.value) return;
  video.value.pause();
  video.value.currentTime = 0;
}

onMounted(prepareVideo);
onSlideEnter(playFromStart);
onSlideLeave(stopAndReset);
</script>

<template>
  <video
    ref="video"
    class="hf-mcp-video"
    :src="'videos/dynamic-space-final.mp4'"
    loop
    muted
    playsinline
    preload="auto"
  />
</template>
