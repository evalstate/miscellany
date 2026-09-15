<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { useRoute } from 'vue-router';
import { useSlideContext } from '@slidev/client';
import ProtocolAdoptionChart from './ProtocolAdoptionChart.vue';
import ToolQualityVersionChart from './ToolQualityVersionChart.vue';

const props = defineProps<{ chart: 'protocol-adoption' | 'tool-quality-version'; title: string; subtitle: string; footer?: string; footnote?: string }>();
const components = { 'protocol-adoption': ProtocolAdoptionChart, 'tool-quality-version': ToolQualityVersionChart };
type ChartApi = { durationMs: number; renderAt: (ms: number) => unknown; state: unknown; configure?: (options: Record<string, unknown>) => void };
const instance = ref<ChartApi>();
const root = ref<HTMLElement>();
const route = useRoute();
const capture = computed(() => route.query.capture === '1');
const { $page, $nav } = useSlideContext();
const active = computed(() => $nav.value.currentSlideNo === $page.value);
// Opt-in automation bridge. Ordinary presentation pages expose no global API.
const bridge = {
  chart: props.chart,
  get durationMs() { return instance.value?.durationMs ?? 0; },
  get state() { return instance.value?.state; },
  async renderAt(ms: number) { instance.value?.renderAt(ms); await nextTick(); return instance.value?.state; },
  async configure(options: Record<string, unknown>) {
    if (!instance.value?.configure) throw new Error('This chart has no configurable client/viewport');
    instance.value.configure(options); await nextTick(); return instance.value.state;
  },
  bounds() {
    const rect = root.value?.querySelector('svg')?.getBoundingClientRect();
    if (!rect) throw new Error('Chart SVG not mounted');
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
  },
};
const browser = () => window as typeof window & { __deckCapture?: typeof bridge };
function unregister() { if (browser().__deckCapture === bridge) delete browser().__deckCapture; }
watch([capture, active, instance], () => {
  if (capture.value && active.value && instance.value) browser().__deckCapture = bridge;
  else unregister();
}, { flush: 'post' });
onBeforeUnmount(unregister);
</script>

<template>
  <div ref="root" class="native-chart-stage" :data-chart="chart" :data-capture="capture">
    <component :is="components[chart]" ref="instance" :capture="capture" :title="title" :subtitle="subtitle" :footer="footer" :footnote="footnote" />
  </div>
</template>
