// "One base model, many adapters": an animated diagram of how LoRA adapters are served in training and in evaluation.
// Plain SVG, one clock for every animation so pause, speed and the mode switch apply everywhere at once.
(() => {
  const root = document.getElementById("lora-fx");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const HARNESSES = ["pi", "hermes", "opencode"];
  // LFM3-1.7B held-out pass@1 (150 SmolDataEnvs tasks), from eval/*_final-heldout/summary.json
  const EVALS = [
    { label: "base model", harness: null, pass: { pi: 0.7, hermes: 0.62, opencode: 0.693 } },
    { label: "pi · ckpt 10", harness: "pi", pass: 0.72 }, { label: "pi · ckpt 20", harness: "pi", pass: 0.687 },
    { label: "hermes · ckpt 10", harness: "hermes", pass: 0.64 }, { label: "hermes · ckpt 20", harness: "hermes", pass: 0.687 },
    { label: "opencode · ckpt 10", harness: "opencode", pass: 0.687 }, { label: "opencode · ckpt 20", harness: "opencode", pass: 0.785 },
  ];
  const color = (h) => (h ? `var(--series-${h})` : "var(--axis-ink)");
  const svg = root.querySelector("svg");
  const caption = root.querySelector(".lora-fx__caption");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs = {}, parent = svg, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    parent.append(e);
    return e;
  }

  // One clock: every tween advances by real time × speed, and stops while paused or off screen.
  let speed = 1.5, userPaused = false, onScreen = false, runId = 0, last = performance.now();
  const tweens = new Set();
  const CANCEL = Symbol("cancel");
  function frame(now) {
    // Capped, so a tab coming back from the background does not jump through a whole phase at once
    const dt = Math.min(250, now - last) * (userPaused || !onScreen ? 0 : speed);
    last = now;
    for (const t of [...tweens]) {
      t.elapsed += dt;
      const p = Math.min(1, t.elapsed / t.duration);
      t.fn(p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);
      if (p >= 1) { tweens.delete(t); t.resolve(); }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  const tween = (duration, fn = () => {}) =>
    new Promise((resolve, reject) => tweens.add({ elapsed: 0, duration: reduced ? 1 : duration, fn, resolve, reject }));
  const wait = (ms) => tween(ms);
  function cancelAll() {
    runId++;
    for (const t of tweens) t.reject(CANCEL);
    tweens.clear();
  }

  // A packet: a dot, with an optional label riding on it, moving from a to b.
  async function packet([x1, y1], [x2, y2], fill, ms, label) {
    const g = el("g", { class: "lora-fx__packet" });
    el("circle", { r: 5.5, fill, stroke: "var(--surface-1)", "stroke-width": 2 }, g);
    if (label) el("text", { y: -10, class: "lora-fx__packet-label", "text-anchor": "middle" }, g, label);
    try {
      await tween(ms, (p) => g.setAttribute("transform", `translate(${x1 + (x2 - x1) * p},${y1 + (y2 - y1) * p - Math.sin(p * Math.PI) * 14})`));
    } finally {
      g.remove();
    }
  }

  function zone(x, y, w, h, label) {
    el("rect", { x, y, width: w, height: h, rx: 10, class: "lora-fx__zone" });
    const tw = label.length * 6.6 + 16;
    el("rect", { x: x + 12, y: y - 9, width: tw, height: 18, rx: 4, class: "lora-fx__zone-tag" });
    el("text", { x: x + 20, y: y + 4, class: "lora-fx__zone-label" }, svg, label);
  }

  function chip(x, y, w, h, fill, title, sub) {
    const g = el("g", { transform: `translate(${x},${y})`, class: "lora-fx__chip" });
    el("rect", { width: w, height: h, rx: 7, fill: `color-mix(in oklab, ${fill} 16%, var(--surface-1))`, stroke: fill, "stroke-width": 1.5 }, g);
    el("text", { x: 10, y: sub ? h / 2 - 2 : h / 2 + 5, class: "lora-fx__chip-title" }, g, title);
    const subText = sub ? el("text", { x: 10, y: h / 2 + 13, class: "lora-fx__chip-sub" }, g, sub) : null;
    return { g, subText };
  }

  function base(x, y, w, h, text) {
    el("rect", { x, y, width: w, height: h, rx: 8, class: "lora-fx__base" });
    for (let i = x + 14; i < x + w - 8; i += 14) el("line", { x1: i, y1: y + 6, x2: i, y2: y + h - 6, class: "lora-fx__base-stripe" });
    const lh = Math.min(28, h - 8);
    el("rect", { x: x + w / 2 - 150, y: y + h / 2 - lh / 2, width: 300, height: lh, rx: 6, class: "lora-fx__base-label-bg" });
    el("text", { x: x + w / 2, y: y + h / 2 + 5, class: "lora-fx__base-label", "text-anchor": "middle" }, svg, text);
  }

  const say = (text) => { caption.textContent = text; };

  // ─── Training: one vLLM, three trainers, each updating only its own adapter ───
  async function training(id) {
    svg.replaceChildren();
    const rows = { pi: 218, hermes: 288, opencode: 358 };
    zone(20, 150, 200, 250, "harness sandboxes");
    zone(300, 60, 400, 340, "GPU 0 · vLLM");
    zone(780, 150, 200, 250, "trainers");
    base(320, 84, 360, 70, "base weights · loaded once · frozen");
    el("text", { x: 500, y: 182, class: "lora-fx__hint", "text-anchor": "middle" }, svg, "adapters, served by name");
    const version = { pi: 1, hermes: 1, opencode: 1 };
    const live = {}, bars = {}, trainerSub = {};
    for (const h of HARNESSES) {
      const y = rows[h];
      chip(36, y - 24, 168, 48, color(h), h, "8 sessions per task");
      live[h] = chip(340, y - 22, 150, 44, color(h), `${h} · v${version[h]}`, "serving");
      const t = chip(796, y - 24, 168, 48, color(h), `trainer · ${h}`, `adapter v${version[h]}`);
      trainerSub[h] = t.subText;
      el("rect", { x: 806, y: y + 16, width: 148, height: 4, rx: 2, class: "lora-fx__track" });
      bars[h] = el("rect", { x: 806, y: y + 16, width: 0, height: 4, rx: 2, fill: color(h) });
    }
    while (id === runId) {
      say("1 · The three harnesses run their sessions in parallel. Each request names its adapter, so one vLLM serves all three policies on top of the same base weights.");
      await Promise.all(HARNESSES.flatMap((h, i) => [0, 1, 2].map(async (k) => {
        await wait(i * 180 + k * 420);
        await packet([204, rows[h]], [340, rows[h]], color(h), 900, k === 0 ? `${h}-v${version[h]}` : null);
        await packet([340, rows[h]], [204, rows[h]], color(h), 700);
      })));
      say("2 · Each finished rollout goes to its trainer: the exact token ids and logprobs the harness saw, no re-tokenisation.");
      await Promise.all(HARNESSES.map(async (h, i) => {
        await wait(i * 200);
        await packet([490, rows[h]], [796, rows[h]], color(h), 1300, "tokens + logprobs");
      }));
      say("3 · A gradient step changes only the trainer's own LoRA adapter (rank 32). The base weights never move.");
      await Promise.all(HARNESSES.map((h, i) => wait(i * 150).then(() => tween(1400, (p) => bars[h].setAttribute("width", 148 * p)))));
      say("4 · The new adapter is loaded into vLLM by name, next to the current one. Requests already running on the old version finish on it, so nothing pauses.");
      await Promise.all(HARNESSES.map(async (h, i) => {
        await wait(i * 180);
        const next = version[h] + 1;
        trainerSub[h].textContent = `adapter v${next}`;
        bars[h].setAttribute("width", 0);
        await packet([796, rows[h]], [552, rows[h]], color(h), 1200, `v${next}`);
        const fresh = chip(500, rows[h] - 22, 150, 44, color(h), `${h} · v${next}`, "loading");
        fresh.g.style.opacity = 0;
        await tween(400, (p) => { fresh.g.style.opacity = p; });
        fresh.subText.textContent = "serving";
        live[h].subText.textContent = "finishing";
        await wait(700);
        await tween(500, (p) => { live[h].g.style.opacity = 1 - p; });
        live[h].g.remove();
        await tween(500, (p) => fresh.g.setAttribute("transform", `translate(${500 - 160 * p},${rows[h] - 22})`));
        live[h] = fresh;
        version[h] = next;
      }));
      await wait(900);
    }
  }

  // ─── Evaluation: every checkpoint at once, as adapters on one vLLM ───
  async function evaluation(id) {
    svg.replaceChildren();
    const rowY = (i) => 134 + i * 38;
    zone(20, 106, 200, 282, "one server per checkpoint");
    zone(300, 60, 400, 328, "GPU 0 · vLLM");
    zone(780, 106, 200, 282, "held-out pass@1");
    base(320, 72, 360, 34, "LFM3-1.7B base weights · loaded once");
    const chips = [], bars = [], values = [];
    EVALS.forEach((e, i) => {
      const y = rowY(i);
      chip(32, y - 15, 176, 30, color(e.harness), e.label);
      const c = chip(360, y - 15, 280, 30, color(e.harness), e.harness ? `adapter · ${e.label}` : "no adapter · the base model");
      c.g.style.opacity = 0;
      chips.push(c);
      el("rect", { x: 796, y: y - 4, width: 120, height: 8, rx: 4, class: "lora-fx__track" });
      bars.push(el("rect", { x: 796, y: y - 4, width: 0, height: 8, rx: 4, fill: color(e.harness) }));
      values.push(el("text", { x: 924, y: y + 4, class: "lora-fx__value" }, svg, ""));
    });
    const footer = el("text", { x: 500, y: 410, class: "lora-fx__hint", "text-anchor": "middle" }, svg, "");
    while (id === runId) {
      chips.forEach((c) => { c.g.style.opacity = 0; });
      bars.forEach((b) => b.setAttribute("width", 0));
      values.forEach((v) => { v.textContent = ""; });
      footer.textContent = "";
      say("1 · One vLLM loads the base weights once, then every checkpoint as a LoRA adapter, side by side, each under its own name.");
      for (const c of chips) await tween(260, (p) => { c.g.style.opacity = p; });
      say("2 · Each checkpoint gets its own sandbox server, running the same 150 held-out tasks. Every request names its adapter, so all checkpoints run at the same time.");
      const target = EVALS.map((e) => (e.harness ? e.pass : (e.pass.pi + e.pass.hermes + e.pass.opencode) / 3));
      const rounds = 4;
      await Promise.all(EVALS.map(async (e, i) => {
        for (let r = 0; r < rounds; r++) {
          await wait(i * 90 + r * 160);
          const fill = color(e.harness);
          await packet([208, rowY(i)], [360, rowY(i)], fill, 700);
          await packet([640, rowY(i)], [796, rowY(i)], fill, 700);
          const p = target[i] * (r + 1) / rounds;
          bars[i].setAttribute("width", 120 * p);
        }
      }));
      EVALS.forEach((e, i) => {
        values[i].textContent = `${(100 * target[i]).toFixed(0)}%${e.harness ? "" : " avg"}`;
      });
      say("3 · Every checkpoint is scored in the same job, on the same tasks, served exactly as it was trained: nothing merged or converted.");
      await wait(2600);
      footer.textContent = `${EVALS.length} policies · 1 GPU · base weights loaded once. Merging instead: ${EVALS.length} full copies, ${EVALS.length} servers.`;
      say("4 · Merging each checkpoint into full weights would load the model once per checkpoint, with a server, and usually a job, for each one.");
      await wait(4200);
    }
  }

  const modes = { training, evaluation };
  let mode = "training";
  function start() {
    cancelAll();
    const id = runId;
    modes[mode](id).catch((e) => { if (e !== CANCEL) throw e; });
  }

  root.querySelectorAll("[data-lora-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.dataset.loraMode;
      root.querySelectorAll("[data-lora-mode]").forEach((x) => x.classList.toggle("on", x === b));
      start();
    }));
  const playBtn = root.querySelector("[data-lora-play]");
  playBtn.addEventListener("click", () => {
    userPaused = !userPaused;
    playBtn.textContent = userPaused ? "play" : "pause";
  });
  root.querySelector("[data-lora-replay]").addEventListener("click", start);
  const speedInput = root.querySelector("[data-lora-speed]");
  const speedVal = root.querySelector(".lora-fx__speed-val");
  speedInput.addEventListener("input", () => {
    speed = +speedInput.value;
    speedVal.textContent = `${speed.toFixed(1)}×`;
  });
  new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; }).observe(root);
  start();
})();
