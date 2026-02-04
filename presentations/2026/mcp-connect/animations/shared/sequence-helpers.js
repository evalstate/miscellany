function runSequenceAnimation({
  selectors,
  cyclePause = 2000,
  fadeOutOffset = 4000,
  initialDelay = 500,
  lastDelayOverride,
  onShow,
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

  function runAnimation() {
    groups.forEach((el) => el.classList.remove('show', 'fade-out'));

    groups.forEach((el) => {
      const delay = parseInt(el.dataset.delay, 10) || 0;
      setTimeout(() => {
        el.classList.add('show');
        if (onShow) {
          onShow(el);
        }
      }, delay);
    });

    const fadeOutTime = finalDelay + fadeOutOffset;
    setTimeout(() => {
      groups.forEach((el) => el.classList.add('fade-out'));
    }, fadeOutTime);

    setTimeout(runAnimation, fadeOutTime + 1000 + cyclePause);
  }

  setTimeout(runAnimation, initialDelay);
}
