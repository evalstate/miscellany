<script setup lang="ts">
import { computed, watch } from 'vue';
import { useSlideContext } from '@slidev/client';
import { useTimedStoryboard } from '../composables/useTimedStoryboard';

const props = defineProps<{ variant: 'legacy' | 'modern' }>();
type Direction = 'right' | 'left';
type Capability = 'tools' | 'resources' | 'prompts' | 'roots' | 'sampling' | 'elicitation';
type Frame = {
  phase: 'message' | 'hold';
  direction: Direction;
  duration: number;
  flash?: Capability;
};
// Independent visual activations, not paired requests and responses.
const activation = (direction: Direction, capability: Capability): Frame[] => [
  { phase: 'message', direction, duration: 850 },
  { phase: 'hold', direction, flash: capability, duration: 700 },
];
const frames = computed<Frame[]>(() => props.variant === 'legacy'
  ? [
      ...activation('right', 'tools'), ...activation('left', 'sampling'),
      ...activation('right', 'resources'), ...activation('left', 'elicitation'),
      ...activation('right', 'prompts'), ...activation('left', 'roots'),
    ]
  : [
      ...activation('right', 'tools'), ...activation('right', 'resources'),
      ...activation('right', 'prompts'),
    ]);
const { active, animationKey, isRunning, play, stop } = useTimedStoryboard(frames, { endDelay: 0 });
const lanes = computed<Direction[]>(() => props.variant === 'legacy' ? ['right', 'left'] : ['right']);
const activeLane = computed(() => props.variant === 'modern' ? 'right' : active.value?.direction);
const { $page, $nav } = useSlideContext();
watch(() => $nav.value.currentSlideNo, (page) => { if (page !== $page.value) stop(); });
</script>

<template>
  <div class="communication-traffic" :class="`communication-traffic-${variant}`"
    :data-variant="variant" :data-phase="active?.phase ?? 'idle'"
    :data-direction="active?.direction ?? 'none'" :data-flash="active?.flash ?? 'none'">
    <div class="legacy-channels">
      <div v-for="lane in lanes" :key="lane" class="legacy-lane" :class="`legacy-lane-${lane}`">
        <span>{{ lane === 'right' ? 'Client → Server' : 'Server → Client' }}</span>
        <i>
          <b v-if="active?.phase === 'message' && activeLane === lane" :key="animationKey"
            class="communication-packet" :class="`communication-packet-${active.direction}`"
            :style="{ '--travel-duration': `${active.duration}ms` }" aria-hidden="true"></b>
        </i>
      </div>
    </div>
    <div class="communication-controls" @click.stop @keydown.enter.stop @keydown.space.stop>
      <button type="button" @click="isRunning ? stop() : play(true)">{{ isRunning ? 'Stop' : 'Start' }}</button>
      <button type="button" :disabled="!isRunning" @click="play(true)">Restart</button>
    </div>
  </div>
</template>
