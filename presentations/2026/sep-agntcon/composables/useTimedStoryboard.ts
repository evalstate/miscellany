import { computed, onBeforeUnmount, ref, type ComputedRef } from 'vue';

export type TimedStoryboardFrame = { duration?: number };
type TimedStoryboardOptions = { defaultDuration?: number; endDelay?: number };

export function useTimedStoryboard<T extends TimedStoryboardFrame>(
  frames: ComputedRef<readonly T[]>, options: TimedStoryboardOptions = {},
) {
  const defaultDuration = options.defaultDuration ?? 1120;
  const endDelay = options.endDelay ?? 160;
  const activeStep = ref(-1);
  const animationKey = ref(0);
  const isRunning = ref(false);
  const isPaused = ref(false);
  const isLooping = ref(false);
  const playbackRate = ref(1);
  const timers: ReturnType<typeof window.setTimeout>[] = [];
  let elapsed = 0;
  let segmentStarted = 0;
  const active = computed<T | undefined>(() => frames.value[activeStep.value]);

  function clearTimers() { while (timers.length) window.clearTimeout(timers.pop()); }
  function captureElapsed() {
    if (isRunning.value) elapsed += (performance.now() - segmentStarted) * playbackRate.value;
  }
  function schedule() {
    segmentStarted = performance.now();
    let at = 0;
    frames.value.forEach((frame, index) => {
      if (index > activeStep.value) {
        timers.push(window.setTimeout(() => {
          activeStep.value = index;
          animationKey.value += 1;
        }, Math.max(0, at - elapsed) / playbackRate.value));
      }
      at += frame.duration ?? defaultDuration;
    });
    timers.push(window.setTimeout(() => {
      if (isLooping.value) play(true);
      else { clearTimers(); isRunning.value = false; }
    }, Math.max(0, at + endDelay - elapsed) / playbackRate.value));
  }
  function play(loop = false) {
    clearTimers();
    elapsed = 0;
    activeStep.value = -1;
    isLooping.value = loop;
    isPaused.value = false;
    isRunning.value = true;
    schedule();
  }
  function pause() {
    if (!isRunning.value) return;
    captureElapsed(); clearTimers();
    isRunning.value = false; isPaused.value = true;
  }
  function resume() {
    if (!isPaused.value) return;
    isPaused.value = false; isRunning.value = true; schedule();
  }
  function setPlaybackRate(rate: number) {
    if (!Number.isFinite(rate) || rate <= 0) return;
    captureElapsed(); clearTimers(); playbackRate.value = rate;
    if (isRunning.value) schedule();
  }
  function stop(reset = true) {
    clearTimers(); isLooping.value = false; isRunning.value = false; isPaused.value = false;
    if (reset) { activeStep.value = -1; elapsed = 0; }
  }
  onBeforeUnmount(() => stop(false));
  return { active, activeStep, animationKey, isLooping, isRunning, isPaused, playbackRate, play, pause, resume, setPlaybackRate, stop };
}
