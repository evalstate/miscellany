import {
  computed,
  onBeforeUnmount,
  ref,
  type ComputedRef,
} from "vue";

export type TimedStoryboardFrame = {
  duration?: number;
};

type TimedStoryboardOptions = {
  defaultDuration?: number;
  endDelay?: number;
};

export function useTimedStoryboard<T extends TimedStoryboardFrame>(
  frames: ComputedRef<readonly T[]>,
  options: TimedStoryboardOptions = {},
) {
  const defaultDuration = options.defaultDuration ?? 1120;
  const endDelay = options.endDelay ?? 160;

  const activeStep = ref(-1);
  const animationKey = ref(0);
  const isRunning = ref(false);
  const isLooping = ref(false);
  const timers: ReturnType<typeof window.setTimeout>[] = [];

  const active = computed<T | undefined>(
    () => frames.value[activeStep.value],
  );

  function clearTimers() {
    while (timers.length) window.clearTimeout(timers.pop());
  }

  function pulseStep(index: number) {
    activeStep.value = index;
    animationKey.value += 1;
  }

  function play(loop = false) {
    clearTimers();
    isLooping.value = loop;
    isRunning.value = true;
    activeStep.value = -1;

    let delay = 0;
    frames.value.forEach((frame, index) => {
      timers.push(window.setTimeout(() => pulseStep(index), delay));
      delay += frame.duration ?? defaultDuration;
    });

    timers.push(
      window.setTimeout(() => {
        if (isLooping.value) {
          activeStep.value = -1;
          play(true);
        } else {
          isRunning.value = false;
        }
      }, delay + endDelay),
    );
  }

  function stop(reset = true) {
    clearTimers();
    isLooping.value = false;
    isRunning.value = false;
    if (reset) activeStep.value = -1;
  }

  onBeforeUnmount(() => stop(false));

  return {
    active,
    activeStep,
    animationKey,
    isLooping,
    isRunning,
    play,
    stop,
  };
}
