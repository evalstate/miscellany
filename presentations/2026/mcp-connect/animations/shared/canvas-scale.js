function scaleCanvas({ selector = '.canvas-root', minScale = 0.25, maxScale = 4 } = {}) {
  const canvases = Array.from(document.querySelectorAll(selector));
  if (canvases.length === 0) {
    return;
  }

  function updateBaseSize(canvas) {
    if (canvas.dataset.baseWidth && canvas.dataset.baseHeight) {
      return;
    }

    const { width, height } = canvas.getBoundingClientRect();
    if (width && height) {
      canvas.dataset.baseWidth = width;
      canvas.dataset.baseHeight = height;
    }
  }

  function updateScale() {
    canvases.forEach((canvas) => {
      updateBaseSize(canvas);
      const baseWidth = parseFloat(canvas.dataset.baseWidth || '0');
      const baseHeight = parseFloat(canvas.dataset.baseHeight || '0');
      if (!baseWidth || !baseHeight) {
        return;
      }

      const scale = Math.min(window.innerWidth / baseWidth, window.innerHeight / baseHeight);
      const clamped = Math.max(minScale, Math.min(scale, maxScale));
      canvas.style.transform = `scale(${clamped})`;
      canvas.style.transformOrigin = 'center';
    });
  }

  const observer = new ResizeObserver(() => updateScale());
  canvases.forEach((canvas) => {
    observer.observe(canvas);
    updateBaseSize(canvas);
  });

  window.addEventListener('resize', updateScale);
  window.addEventListener('load', () => requestAnimationFrame(updateScale));
  requestAnimationFrame(updateScale);
}
