<template>
  <div class="reliable-video" :class="{ 'reliable-video--failed': failed }">
    <video
      v-if="blobUrl"
      :src="blobUrl"
      autoplay
      loop
      muted
      playsinline
      preload="auto"
      @canplay="loading = false"
      @error="onVideoError"
    />

    <div v-if="loading" class="reliable-video__status">Loading video preview…</div>

    <div v-if="failed" class="reliable-video__fallback">
      <p>Video preview could not load.</p>
      <button type="button" @click="load">Retry</button>
      <a :href="src" target="_blank" rel="noreferrer">Open MP4</a>
    </div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps({
  src: { type: String, required: true },
})

const blobUrl = ref('')
const failed = ref(false)
const loading = ref(true)
let controller

function revoke() {
  if (blobUrl.value) URL.revokeObjectURL(blobUrl.value)
  blobUrl.value = ''
}

async function load() {
  controller?.abort()
  controller = new AbortController()
  revoke()
  failed.value = false
  loading.value = true

  try {
    const response = await fetch(props.src, {
      cache: 'force-cache',
      signal: controller.signal,
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    blobUrl.value = URL.createObjectURL(await response.blob())
  } catch (error) {
    if (error.name === 'AbortError') return
    loading.value = false
    failed.value = true
  }
}

function onVideoError() {
  loading.value = false
  failed.value = true
}

onMounted(load)
onBeforeUnmount(() => {
  controller?.abort()
  revoke()
})
</script>
