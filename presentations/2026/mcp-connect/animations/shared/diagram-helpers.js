function setMessagePosition(message, x, y) {
  message.style.left = `${x}px`;
  message.style.top = `${y}px`;
}

function animateMessageBetween(message, fromX, fromY, toX, toY, duration, callback) {
  const startTime = performance.now();

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    setMessagePosition(
      message,
      fromX + (toX - fromX) * eased,
      fromY + (toY - fromY) * eased
    );

    if (progress < 1) {
      requestAnimationFrame(step);
    } else if (callback) {
      callback();
    }
  }

  requestAnimationFrame(step);
}

function animatePathBetween(message, waypoints, totalDuration, callback) {
  if (waypoints.length < 2) {
    if (callback) {
      callback();
    }
    return;
  }

  let totalDist = 0;
  for (let i = 0; i < waypoints.length - 1; i++) {
    const dx = waypoints[i + 1].x - waypoints[i].x;
    const dy = waypoints[i + 1].y - waypoints[i].y;
    totalDist += Math.sqrt(dx * dx + dy * dy);
  }

  const durations = [];
  for (let i = 0; i < waypoints.length - 1; i++) {
    const dx = waypoints[i + 1].x - waypoints[i].x;
    const dy = waypoints[i + 1].y - waypoints[i].y;
    const legDist = Math.sqrt(dx * dx + dy * dy);
    durations.push((legDist / totalDist) * totalDuration);
  }

  let i = 0;
  function nextLeg() {
    if (i < waypoints.length - 1) {
      animateMessageBetween(
        message,
        waypoints[i].x, waypoints[i].y,
        waypoints[i + 1].x, waypoints[i + 1].y,
        durations[i],
        () => {
          i++;
          nextLeg();
        }
      );
    } else if (callback) {
      callback();
    }
  }

  nextLeg();
}

function configureMessage(elements, {
  type,
  arrow,
  method,
  detail,
  sessionText,
  sessionError = false,
  direction = 'right',
} = {}) {
  const container = elements.container;
  const header = elements.header || container.querySelector('.message-header');

  container.className = `message ${type} visible`;
  elements.type.textContent = method;
  elements.arrow.textContent = arrow;
  elements.detail.textContent = detail;

  if (elements.session) {
    if (sessionText) {
      elements.session.innerHTML = sessionText;
      elements.session.style.display = 'block';
      elements.session.className = 'message-session' + (sessionError ? ' error' : '');
    } else {
      elements.session.style.display = 'none';
    }
  }

  if (header) {
    header.style.flexDirection = direction === 'left' ? 'row-reverse' : 'row';
  }
}

function runStepSequence(steps, { initialDelay = 100, stepPause, startOnLoad = true } = {}) {
  if (!steps || steps.length === 0) {
    return;
  }

  const pause = stepPause ?? (typeof ANIMATION !== 'undefined' ? ANIMATION.STEP_PAUSE : 1500);
  let currentStep = 0;
  let animationTimer = null;

  function runStep() {
    if (currentStep >= steps.length) {
      currentStep = 0;
    }

    const step = steps[currentStep];
    step.setup();

    animationTimer = setTimeout(() => {
      step.animate(() => {
        if (step.after) {
          step.after();
        }
        currentStep++;
        animationTimer = setTimeout(runStep, pause);
      });
    }, initialDelay);
  }

  const start = () => {
    if (animationTimer) {
      clearTimeout(animationTimer);
    }
    currentStep = 0;
    runStep();
  };

  if (startOnLoad) {
    if (document.readyState === 'complete') {
      start();
    } else {
      window.addEventListener('load', start, { once: true });
    }

    window.addEventListener('pageshow', start);
  } else {
    start();
  }
}
