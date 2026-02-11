function runSequenceAnimation({
  selectors,
  cyclePause = 2000,
  fadeOutOffset = 4000,
  initialDelay = 500,
  lastDelayOverride,
  onShow,
  scrollContainerSelector,
} = {}) {
  const groups = (selectors || []).flatMap((selector) =>
    Array.from(document.querySelectorAll(selector))
  );

  if (groups.length === 0) {
    return;
  }

  const delays = groups.map((el) => parseInt(el.dataset.delay, 10) || 0);
  const maxDelay = Math.max(...delays, 0);
  const finalDelay = Math.max(maxDelay, lastDelayOverride || 0);
  const scrollContainer = scrollContainerSelector
    ? document.querySelector(scrollContainerSelector)
    : null;

  let timers = [];

  function clearTimers() {
    timers.forEach((id) => clearTimeout(id));
    timers = [];
  }

  function schedule(fn, delay) {
    const id = setTimeout(fn, delay);
    timers.push(id);
    return id;
  }

  function runAnimation() {
    clearTimers();
    groups.forEach((el) => el.classList.remove('show', 'fade-out'));
    if (scrollContainer) {
      scrollContainer.scrollTop = 0;
    }

    // Force reflow to ensure class removal is processed before re-adding
    void document.body.offsetHeight;

    groups.forEach((el) => {
      const delay = parseInt(el.dataset.delay, 10) || 0;
      schedule(() => {
        el.classList.add('show');
        if (onShow) {
          onShow(el);
        }
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }
      }, delay);
    });

    const fadeOutTime = finalDelay + fadeOutOffset;
    schedule(() => {
      groups.forEach((el) => el.classList.add('fade-out'));
    }, fadeOutTime);

    schedule(runAnimation, fadeOutTime + 1000 + cyclePause);
  }

  function start() {
    clearTimers();
    schedule(runAnimation, initialDelay);
  }

  if (document.readyState === 'complete') {
    start();
  } else {
    window.addEventListener('load', start, { once: true });
  }

  window.addEventListener('pageshow', () => {
    start();
  });
}
