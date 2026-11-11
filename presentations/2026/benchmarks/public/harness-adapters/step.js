// What goes into one training step: sessions run in parallel, a task's group of 8 rollouts joins the queue once all
// of them are done, and the trainer packs finished rollouts into 4 micro-batches up to a token budget before stepping.
// Illustrative timings, simulated once up front (seeded), then drawn at whatever time the clock is at.
(() => {
  const root = document.getElementById("step-fx");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const svg = root.querySelector("svg");
  const caption = root.querySelector(".lora-fx__caption");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SLOTS = 6, G = 8, MB = 4, CAP = 10, STALE = 4;  // tasks running, rollouts per task, micro-batches, budget, staleness
  const TRAIN_S = 1.0, DT = 0.05, LOOP = 56;

  // Simulation
  let seed = 7;
  const rand = () => { seed = (seed + 0x6d2b79f5) | 0; let x = Math.imul(seed ^ (seed >>> 15), 1 | seed); x ^= x + Math.imul(x ^ (x >>> 7), 61 | x); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
  let nextId = 1, version = 1;
  const newTask = (start, slow) => ({
    id: nextId++, start, version,
    rollouts: Array.from({ length: G }, () => {
      const dur = slow ? 30 + rand() * 4 : 2 + rand() * 3.5;
      return { dur, len: 0.3 + Math.min(dur, 8) / 6, pass: rand() < 0.6 };
    }),
  });
  const end = (task) => task.start + Math.max(...task.rollouts.map((r) => r.dur));
  const slots = Array.from({ length: SLOTS }, (_, i) => newTask(0, i === 4));
  const queue = [], frames = [], steps = [], drops = [];
  let mbs = [[]], trainingUntil = null, stepGroups = [];
  const fits = (task) => {
    // Would the whole group fit in what's left of this step's micro-batches?
    let k = mbs.length - 1, used = mbs[k].reduce((a, r) => a + r.len, 0);
    for (const r of task.rollouts) {
      if (used + r.len > CAP) { k += 1; used = 0; }
      if (k >= MB) return false;
      used += r.len;
    }
    return true;
  };
  let started = SLOTS;
  for (let i = 0; i * DT <= LOOP; i++) {
    const t = i * DT;
    slots.forEach((task, s) => {
      if (t >= end(task)) { queue.push({ task, at: t }); slots[s] = newTask(t, ++started % 11 === 0); }
    });
    if (trainingUntil !== null && t >= trainingUntil) {
      version += 1;
      steps.push({ at: t, version, groups: stepGroups.length, rollouts: stepGroups.length * G });
      mbs = [[]]; stepGroups = []; trainingUntil = null;
    }
    while (trainingUntil === null && queue.length) {
      const { task } = queue[0];
      if (version - task.version > STALE) { drops.push({ at: t, task }); queue.shift(); continue; }
      if (!fits(task)) { trainingUntil = t + TRAIN_S; break; }
      queue.shift();
      for (const r of task.rollouts) {
        if (mbs.at(-1).reduce((a, x) => a + x.len, 0) + r.len > CAP) mbs.push([]);
        mbs.at(-1).push(r);
      }
      stepGroups.push(task.id);
    }
    frames.push({
      slots: slots.slice(), queue: queue.map((q) => q.task), mbs: mbs.map((m) => m.slice()),
      training: trainingUntil !== null, version,
    });
  }

  // Static scene
  const el = (tag, attrs = {}, parent = svg, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    parent.append(e);
    return e;
  };
  const panel = (x, w, title, sub) => {
    el("rect", { x, y: 58, width: w, height: 278, rx: 10, fill: "var(--surface-1)", stroke: "var(--border)", "stroke-width": 1.5 });
    el("text", { x: x + 14, y: 80, class: "lora-fx__chip-title" }, svg, title);
    el("text", { x: x + 14, y: 96, class: "lora-fx__chip-sub" }, svg, sub);
  };
  panel(16, 392, "agent sessions, running at once", "6 tasks × 8 rollouts, one sandbox each");
  panel(422, 170, "finished tasks", "all 8 rollouts scored");
  panel(606, 378, "one gradient step", "4 micro-batches, each up to a token budget");
  for (let k = 0; k < MB; k++) {
    const y = 118 + k * 40;
    el("text", { x: 620, y: y + 16, class: "lora-fx__chip-sub" }, svg, `batch ${k + 1}`);
    el("rect", { x: 680, y, width: CAP * 26, height: 22, rx: 5, fill: "var(--page)", stroke: "var(--border)", "stroke-dasharray": "3 3" });
  }
  el("text", { x: 940, y: 112, "text-anchor": "end", class: "lora-fx__hint" }, svg, "token budget →");
  el("text", { x: 500, y: 364, "text-anchor": "middle", class: "lora-fx__hint" }, svg,
    "sessions keep running while the trainer computes: generation never waits for a step");
  const layer = el("g");

  // Drawing
  const C = 2 * Math.PI * 9;
  let speed = 1.5, paused = false, onScreen = false, t = 0, last = performance.now();

  function draw() {
    const u = t % LOOP;
    const f = frames[Math.min(frames.length - 1, Math.floor(u / DT))];
    layer.replaceChildren();
    // Running tasks: each rollout's ring fills as its session runs, then shows pass or fail
    f.slots.forEach((task, s) => {
      const y = 126 + s * 34;
      const stale = f.version - task.version > STALE;
      el("text", { x: 30, y: y + 4, class: "lora-fx__chip-title", style: "font-size:12.5px" }, layer, `task ${task.id}`);
      el("text", { x: 92, y: y + 4, class: "lora-fx__chip-sub", style: stale ? "fill: var(--bad)" : "" }, layer, `v${task.version}`);
      task.rollouts.forEach((r, j) => {
        const cx = 140 + j * 32, p = Math.max(0, Math.min(1, (u - task.start) / r.dur));
        el("circle", { cx, cy: y, r: 9, fill: "var(--page)", stroke: "var(--border)", "stroke-width": 2 }, layer);
        if (p >= 1) el("circle", { cx, cy: y, r: 9, fill: r.pass ? "var(--good)" : "var(--bad)" }, layer);
        else el("circle", { cx, cy: y, r: 9, fill: "none", stroke: "var(--accent)", "stroke-width": 2.5,
          "stroke-dasharray": `${C * p} ${C}`, transform: `rotate(-90 ${cx} ${y})` }, layer);
      });
    });
    // Queue of finished groups
    f.queue.slice(0, 6).forEach((task, k) => {
      const y = 112 + k * 34;
      el("rect", { x: 436, y, width: 142, height: 26, rx: 7, fill: "color-mix(in oklab, var(--accent) 12%, var(--page))", stroke: "var(--accent)", "stroke-width": 1.2 }, layer);
      el("text", { x: 507, y: y + 17, "text-anchor": "middle", style: "font: 11px var(--mono); fill: var(--text-primary)" }, layer, `task ${task.id} · v${task.version}`);
    });
    const drop = drops.findLast((d) => d.at <= u && u - d.at < 2.4);
    if (drop) {
      const a = 1 - (u - drop.at) / 2.4;
      el("text", { x: 507, y: 316, "text-anchor": "middle", opacity: a, style: "font: 600 11.5px var(--sans); fill: var(--bad)" }, layer, `task ${drop.task.id} dropped:`);
      el("text", { x: 507, y: 330, "text-anchor": "middle", opacity: a, style: "font: 11px var(--sans); fill: var(--bad)" }, layer, `over ${STALE} versions old`);
    }
    // Micro-batches being packed
    f.mbs.forEach((mb, k) => {
      let x = 680;
      for (const r of mb) {
        el("rect", { x: x + 1, y: 119 + k * 40, width: r.len * 26 - 2, height: 20, rx: 4,
          fill: `color-mix(in oklab, ${r.pass ? "var(--good)" : "var(--bad)"} 55%, var(--page))` }, layer);
        x += r.len * 26;
      }
    });
    // The adapter: computing, then the new version loaded into vLLM
    const lastStep = steps.findLast((s) => s.at <= u);
    const fresh = lastStep && u - lastStep.at < 1.6;
    const color = f.training ? "var(--fx-accent)" : fresh ? "var(--good)" : "var(--border)";
    el("rect", { x: 680, y: 282, width: 260, height: 36, rx: 9, fill: "var(--page)", stroke: color, "stroke-width": f.training || fresh ? 2 : 1.5 }, layer);
    el("text", { x: 810, y: 305, "text-anchor": "middle", style: "font: 600 13px var(--sans); fill: var(--text-primary)" }, layer,
      f.training ? `forward + backward → v${f.version + 1}` : fresh ? `adapter v${f.version} loaded into vLLM` : `adapter v${f.version}, serving`);
    caption.textContent = lastStep
      ? `Step ${lastStep.version - 1} trained on ${lastStep.groups} task${lastStep.groups === 1 ? "" : "s"} (${lastStep.rollouts} rollouts). How many fit depends on how long the rollouts are: long sessions fill the token budget with fewer tasks.`
      : "Sessions run in parallel. A task joins the queue once all 8 of its rollouts are scored, since its advantages are computed within the group.";
  }

  function frame(now) {
    const dt = Math.min(0.25, (now - last) / 1000);
    last = now;
    if (!paused && onScreen) t += dt * speed;
    draw();
    requestAnimationFrame(frame);
  }
  if (reduced) { t = 30; draw(); } else requestAnimationFrame(frame);
  const play = root.querySelector("[data-step-play]");
  play.addEventListener("click", () => { paused = !paused; play.textContent = paused ? "play" : "pause"; });
  root.querySelector("[data-step-replay]").addEventListener("click", () => { t = 0; });
  const input = root.querySelector("[data-step-speed]");
  const val = root.querySelector(".lora-fx__speed-val");
  input.addEventListener("input", () => { speed = +input.value; val.textContent = `${speed.toFixed(1)}×`; });
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; }).observe(root);
})();
