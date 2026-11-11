// "From a shared trajectory to an RL task": the traj2envs pipeline, animated. Illustrative flow, no real counts: each
// dot is a trajectory, dropped ones fall out with a rejection reason from traj2envs' report.json.
(() => {
  const root = document.getElementById("pipe-fx");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const svg = root.querySelector("svg");
  const caption = root.querySelector(".lora-fx__caption");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FAMILY = { repair: "#8e6bd8", build: "#d29a12", extend: "#c2417a", qa: "#2f97b8" };
  const LEVELS = ["easy", "medium", "hard"];
  const STAGES = [
    // drop: rejection codes traj2envs records at that stage (replay itself rejects nothing)
    { name: "screen", sub: ["atif-scan drops", "tampered traces"], drop: ["gate"] },
    { name: "replay", sub: ["files rebuilt", "per command"], drop: null },
    { name: "find work", sub: ["repair · build", "extend · qa"], drop: ["llm_declined", "answer_in_question", "duplicate"] },
    { name: "build & check", sub: ["Docker image", "mutation tests"], drop: ["weak_tests", "image_build_failed", "check_fails_differently"] },
    { name: "trial matrix", sub: ["oracle → 1", "do nothing → 0"], drop: ["trials_failed"] },
    { name: "publish & score", sub: ["LFM2.5-24B scores", "solve rate → level"], drop: null },
  ];
  const CAPTIONS = [
    "1 · Screen: agent sessions people shared on the Hugging Face Hub come in as trajectories. atif-scan drops those with reward tampering or benchmark leakage.",
    "2 · Replay: every write and edit is applied to a virtual filesystem, so each shell command comes with the files just before it ran.",
    "3 · Find work: each family looks for a verifiable unit of work. Only here, and only for build and extend tasks, a teacher LLM (Mistral-Large-4) writes the interface contract, a stub and the hidden tests.",
    "4 · Build & check: the environment is built as a Docker image and checked offline, with no LLM: the teacher's tests must pass on the reference, fail on the stub, and catch most injected bugs.",
    "5 · Trial matrix: the original solution must score full reward, twice; doing nothing, a stub or tampering with the tests must score 0.",
    "6 · Publish & score: the image is pinned by digest. LFM2.5-24B-A2B, the largest model we train, attempts each task several times, and its solve rate sets the difficulty.",
  ];
  const X = STAGES.map((_, i) => 178 + i * 128);
  const TRACK = 148, DROP = 300, BOX_W = 116, OUT_X = 922;
  const fates = [3, null, 2, 4, null, 0, 3, null, 2, null, 2, 4, null, 3, 2, null, null, 4, 0, null];
  const families = ["build", "repair", "qa", "build", "extend", "qa", "repair", "build"];

  const el = (tag, attrs = {}, parent = svg, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    parent.append(e);
    return e;
  };

  // Static scene
  el("rect", { x: 14, y: 60, width: 960, height: 312, rx: 10, class: "lora-fx__zone" });
  el("rect", { x: 26, y: 51, width: 210, height: 18, rx: 4, class: "lora-fx__zone-tag" });
  el("text", { x: 34, y: 64, class: "lora-fx__zone-label" }, svg, "Repo2RLEnvs run · per trajectory");
  // Where they come from: sessions people shared on the Hugging Face Hub
  el("text", { x: 70, y: TRACK - 50, "text-anchor": "middle", "font-size": 24 }, svg, "🤗");
  el("text", { x: 70, y: TRACK - 30, class: "lora-fx__chip-title", "text-anchor": "middle" }, svg, "trajectories");
  el("text", { x: 70, y: TRACK - 15, "text-anchor": "middle", style: "font: 600 12px var(--sans); fill: var(--accent)" }, svg, "from the HF Hub");
  for (let k = 0; k < 4; k++) el("rect", { x: 40 + k * 4, y: TRACK - 4 + k * 5, width: 58, height: 28, rx: 5, class: "lora-fx__base" });
  el("text", { x: 70, y: TRACK + 52, class: "lora-fx__chip-sub", "text-anchor": "middle" }, svg, "pi · Codex");
  el("text", { x: 70, y: TRACK + 66, class: "lora-fx__chip-sub", "text-anchor": "middle" }, svg, "Claude Code …");
  el("line", { x1: 108, y1: TRACK, x2: OUT_X - 40, y2: TRACK, stroke: "var(--border)", "stroke-width": 2 });
  const boxes = STAGES.map((s, i) => {
    const g = el("g");
    // The dots ride the track through the gap between a stage's title and its description
    const r = el("rect", { x: X[i] - BOX_W / 2, y: TRACK - 38, width: BOX_W, height: 84, rx: 9, fill: "var(--surface-1)", stroke: "var(--border)", "stroke-width": 1.5 }, g);
    el("text", { x: X[i], y: TRACK - 15, class: "lora-fx__chip-title", "text-anchor": "middle" }, g, `${i + 1} · ${s.name}`);
    el("text", { x: X[i], y: TRACK + 24, class: "lora-fx__chip-sub", "text-anchor": "middle" }, g, s.sub[0]);
    el("text", { x: X[i], y: TRACK + 38, class: "lora-fx__chip-sub", "text-anchor": "middle" }, g, s.sub[1]);
    return r;
  });
  // The only step that calls an LLM: it writes the build and extend tasks' tests, which steps 4 and 5 then check
  el("rect", { x: X[2] - 64, y: 74, width: 128, height: 24, rx: 12, fill: "color-mix(in oklab, var(--accent) 14%, var(--surface-1))", stroke: "var(--accent)", "stroke-width": 1.2 });
  el("text", { x: X[2], y: 90, "text-anchor": "middle", style: "font: 600 11px var(--sans); fill: var(--text-primary)" }, svg, "teacher LLM writes tests");
  el("line", { x1: X[2], y1: 98, x2: X[2], y2: TRACK - 38, stroke: "var(--accent)", "stroke-width": 1.2, "stroke-dasharray": "3 3" });
  el("rect", { x: OUT_X - 34, y: TRACK - 70, width: 72, height: 228, rx: 9, fill: "var(--surface-1)", stroke: "var(--fx-accent)", "stroke-width": 1.5 });
  el("text", { x: OUT_X + 2, y: TRACK - 52, class: "lora-fx__chip-title", "text-anchor": "middle" }, svg, "dataset");
  el("text", { x: OUT_X + 2, y: TRACK - 38, class: "lora-fx__chip-sub", "text-anchor": "middle" }, svg, "tasks/<id>");
  el("line", { x1: 120, y1: DROP + 22, x2: X[4] + 60, y2: DROP + 22, stroke: "var(--border)", "stroke-dasharray": "3 4" });
  el("text", { x: 120, y: DROP + 40, class: "lora-fx__hint" }, svg, "dropped, with the reason recorded in report.json");
  const shelf = el("g");
  const layer = el("g");

  // One clock, as in the LoRA diagram: speed and pause apply everywhere
  let speed = 1.5, paused = false, onScreen = false, t = 0, last = performance.now();
  const CYCLE = 0.55;  // seconds between two trajectories entering
  const SEG = 0.85;    // seconds from one stage to the next
  const LOOP = fates.length * CYCLE + 7 * SEG + 2;

  function state(i, time) {
    // Where trajectory i is at `time`: before entry, on the track, falling out, or landed in the dataset
    const age = time - i * CYCLE;
    if (age < 0) return null;
    const fate = fates[i];
    const stops = fate === null ? STAGES.length : fate + 1;
    const xs = [108, ...X.slice(0, stops)];
    const leg = Math.floor(age / SEG);
    if (leg < stops) {
      const p = (age - leg * SEG) / SEG;
      const e = p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2;
      return { x: xs[leg] + (xs[leg + 1] - xs[leg]) * e, y: TRACK, stage: leg, phase: "track" };
    }
    const rest = age - stops * SEG;
    if (fate !== null) {
      if (rest > 1.6) return null;
      const p = Math.min(1, rest / 0.7);
      return { x: X[fate], y: TRACK + (DROP - TRACK) * p * p, stage: fate, phase: "drop", alpha: 1 - Math.max(0, (rest - 0.9) / 0.7), reason: STAGES[fate].drop[i % STAGES[fate].drop.length] };
    }
    const p = Math.min(1, rest / 0.6);
    return { x: X[5] + (OUT_X - X[5]) * p, y: TRACK, stage: 5, phase: p >= 1 ? "landed" : "out" };
  }

  function colorOf(i, st) {
    // A trajectory gets its family's color once a family has found work in it (stage 3 onwards)
    if (st.stage < 2) return "var(--axis-ink)";
    return FAMILY[families[i % families.length]];
  }

  function draw() {
    layer.replaceChildren();
    const active = Math.min(5, Math.floor((t % LOOP) / (LOOP / 6)));
    boxes.forEach((b, i) => b.setAttribute("stroke", i === active ? "var(--fx-accent)" : "var(--border)"));
    caption.textContent = CAPTIONS[active];
    const kept = [];
    fates.forEach((_, i) => {
      const st = state(i, t % LOOP);
      if (!st) return;
      if (st.phase === "landed") { kept.push(i); return; }
      const g = el("g", { opacity: st.alpha ?? 1 }, layer);
      el("circle", { cx: st.x, cy: st.y, r: 6, fill: colorOf(i, st), stroke: "var(--surface-1)", "stroke-width": 2 }, g);
      if (st.phase === "drop" && st.y > TRACK + 60) el("text", { x: st.x, y: st.y + 20, class: "lora-fx__packet-label", "text-anchor": "middle" }, g, st.reason);
    });
    shelf.replaceChildren();
    kept.slice(-6).forEach((i, k) => {
      const fam = families[i % families.length];
      const y = TRACK - 26 + k * 30;
      el("rect", { x: OUT_X - 28, y, width: 60, height: 24, rx: 5, fill: `color-mix(in oklab, ${FAMILY[fam]} 16%, var(--surface-1))`, stroke: FAMILY[fam], "stroke-width": 1.2 }, shelf);
      el("text", { x: OUT_X + 2, y: y + 11, class: "lora-fx__chip-sub", "text-anchor": "middle", style: "font-size:10px" }, shelf, fam);
      el("text", { x: OUT_X + 2, y: y + 21, class: "lora-fx__chip-sub", "text-anchor": "middle", style: "font-size:9px" }, shelf, LEVELS[(i * 7) % 3]);
    });
  }

  function frame(now) {
    const dt = Math.min(0.25, (now - last) / 1000);
    last = now;
    if (!paused && onScreen) t += dt * speed;
    draw();
    requestAnimationFrame(frame);
  }
  if (reduced) { t = LOOP - 2.5; draw(); } else requestAnimationFrame(frame);

  const play = root.querySelector("[data-pipe-play]");
  play.addEventListener("click", () => { paused = !paused; play.textContent = paused ? "play" : "pause"; });
  root.querySelector("[data-pipe-replay]").addEventListener("click", () => { t = 0; });
  const input = root.querySelector("[data-pipe-speed]");
  const val = root.querySelector(".lora-fx__speed-val");
  input.addEventListener("input", () => { speed = +input.value; val.textContent = `${speed.toFixed(1)}×`; });
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; }).observe(root);
})();
