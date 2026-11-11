// "Four task types": for each Repo2RLEnvs family, what the agent gets, what it does in the sandbox and how the verifier
// scores it. Illustrative tasks, not taken from the dataset.
(() => {
  const root = document.getElementById("tt-fx");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const svg = root.querySelector("svg");
  const caption = root.querySelector(".lora-fx__caption");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FAMILY = { repair: "#8e6bd8", build: "#d29a12", extend: "#c2417a", qa: "#2f97b8" };
  const TONE = { cmd: "var(--text-primary)", out: "var(--text-muted)", bad: "var(--bad)", good: "var(--good)" };
  const SCENES = [
    {
      family: "repair",
      request: ["The test suite fails.", "Fix the bug."],
      files: ["src/dates.py", "tests/test_dates.py"],
      note: "codebase as it was when the check failed",
      agent: [["$ pytest -q", "cmd"], ["1 failed: test_parse_iso", "bad"], ["$ edit src/dates.py", "cmd"], ["$ pytest -q", "cmd"], ["4 passed", "good"]],
      grader: "the trajectory's own failing check",
      checks: [["tests untouched", true], ["pytest exits 0", true]],
      reward: "1.0",
      caption: "repair · The agent gets the codebase at the moment a check failed in the original session. Reward 1 if the same check now passes, with the tests untouched.",
    },
    {
      family: "build",
      request: ["Build a CLI that converts", "a CSV file to JSON."],
      files: ["(empty workspace)"],
      note: "contract: csv2json(path) → list[dict]",
      agent: [["$ write csv2json/core.py", "cmd"], ["$ write csv2json/cli.py", "cmd"], ["$ python -m csv2json data.csv", "cmd"], ['[{"id": 1, "name": "ada"}, …]', "out"], ["$ pytest -q  ·  3 passed", "good"]],
      grader: "hidden tests, written by the teacher",
      checks: [["test_header", true], ["test_quoted_fields", true], ["test_types", true], ["test_empty_file", false], ["test_cli_output", true]],
      reward: "0.8",
      caption: "build · The agent gets the first request and an interface contract, in an empty workspace. The reward is the fraction of hidden tests that pass; the original project is the oracle.",
    },
    {
      family: "extend",
      request: ["Add a --json flag that", "prints a summary."],
      files: ["csv2json/core.py", "csv2json/cli.py", "tests/test_core.py"],
      note: "the project as the original agent left it",
      agent: [["$ cat csv2json/cli.py", "cmd"], ["$ edit csv2json/cli.py", "cmd"], ["$ python -m csv2json --json data.csv", "cmd"], ['{"rows": 3, "columns": 2}', "out"], ["$ pytest -q  ·  6 passed", "good"]],
      grader: "hidden tests for the change",
      checks: [["test_json_flag", true], ["test_json_empty", true], ["test_default_output", true]],
      reward: "1.0",
      caption: "extend · The agent gets the finished project and a later request from the same session. Hidden tests for that change decide the reward.",
    },
    {
      family: "qa",
      request: ["How many rows are in the", "train split of <dataset>?"],
      files: ["(no files) · public Hub data"],
      note: "answer computed from a public dataset",
      agent: [["$ python fetch.py", "cmd"], ["  ds = load_dataset(…)", "out"], ["  print(len(ds['train']))", "out"], ["12804", "out"], ["$ echo 12804 > answer.txt", "cmd"]],
      grader: "exact or numeric match",
      checks: [["answer 12804", true], ["expected 12804", true]],
      reward: "1.0",
      caption: "qa · The agent answers a factual question about public Hub data. Its answer is graded by exact or numeric match; the expected answer must appear in a tool output from the original session.",
    },
  ];
  const SCENE = 9;                   // seconds per task type
  const LINE_T0 = 0.8, LINE_DT = 0.8; // agent lines
  const CHECK_T0 = 5.2, CHECK_DT = 0.45;
  const TOP = 50, H = 230;
  const COLS = [{ x: 20, w: 280, label: "task" }, { x: 330, w: 380, label: "agent · in the sandbox" }, { x: 740, w: 240, label: "verifier" }];

  const el = (tag, attrs = {}, parent = svg, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    parent.append(e);
    return e;
  };

  // Static scene: three cards and the arrows between them
  const cards = COLS.map((c) => {
    const r = el("rect", { x: c.x, y: TOP, width: c.w, height: H, rx: 10, fill: "var(--surface-1)", stroke: "var(--border)", "stroke-width": 1.5 });
    el("text", { x: c.x + 14, y: TOP + 22, class: "lora-fx__zone-label" }, svg, c.label);
    return r;
  });
  for (const [a, b] of [[0, 1], [1, 2]]) {
    const x1 = COLS[a].x + COLS[a].w, x2 = COLS[b].x, y = TOP + H / 2;
    el("line", { x1: x1 + 2, y1: y, x2: x2 - 6, y2: y, stroke: "var(--border)", "stroke-width": 2 });
    el("path", { d: `M${x2 - 8} ${y - 5} L${x2 - 1} ${y} L${x2 - 8} ${y + 5} Z`, fill: "var(--border)" });
  }
  const layer = el("g");
  const tabs = [...root.querySelectorAll("[data-tt-family]")];

  // One clock, as in the other diagrams: speed and pause apply everywhere
  let speed = 1, paused = false, onScreen = false, t = 0, last = performance.now();
  const LOOP = SCENES.length * SCENE;

  function draw() {
    layer.replaceChildren();
    const k = Math.floor((t % LOOP) / SCENE);
    const s = SCENES[k], col = FAMILY[s.family];
    const time = reduced ? SCENE : (t % LOOP) - k * SCENE;
    tabs.forEach((b, i) => b.classList.toggle("on", i === k));
    cards.forEach((c) => c.setAttribute("stroke", "var(--border)"));
    caption.textContent = s.caption;

    // Task: the family tag, the request, the workspace the agent starts from
    const [tc, ac, vc] = COLS;
    el("rect", { x: tc.x + tc.w - 78, y: TOP + 9, width: 64, height: 20, rx: 10, fill: `color-mix(in oklab, ${col} 16%, var(--surface-1))`, stroke: col, "stroke-width": 1.2 }, layer);
    el("text", { x: tc.x + tc.w - 46, y: TOP + 23, "text-anchor": "middle", style: `font: 600 11px var(--mono); fill: ${col}` }, layer, s.family);
    s.request.forEach((line, i) => el("text", { x: tc.x + 14, y: TOP + 58 + i * 18, class: "lora-fx__chip-title" }, layer, line));
    el("text", { x: tc.x + 14, y: TOP + 112, class: "lora-fx__zone-label" }, layer, "workspace");
    s.files.forEach((f, i) => el("text", { x: tc.x + 14, y: TOP + 132 + i * 17, class: "lora-fx__chip-sub" }, layer, f));
    el("text", { x: tc.x + 14, y: TOP + H - 14, class: "lora-fx__hint" }, layer, s.note);

    // Agent: commands and outputs, typed out one line at a time
    if (time > LINE_T0 && time < CHECK_T0) cards[1].setAttribute("stroke", col);
    s.agent.forEach(([text, tone], i) => {
      const p = (time - LINE_T0 - i * LINE_DT) / 0.45;
      if (p <= 0) return;
      const shown = tone === "cmd" ? text.slice(0, Math.ceil(text.length * Math.min(1, p))) : text;
      el("text", { x: ac.x + 16, y: TOP + 56 + i * 30, style: `font: 12.5px var(--mono); fill: ${TONE[tone]}; white-space: pre`, opacity: Math.min(1, p * 2) }, layer, shown);
    });

    // Verifier: what decides the reward, each check, then the reward itself
    el("text", { x: vc.x + 14, y: TOP + 46, class: "lora-fx__hint" }, layer, s.grader);
    if (time > CHECK_T0) cards[2].setAttribute("stroke", col);
    s.checks.forEach(([label, ok], i) => {
      const p = (time - CHECK_T0 - i * CHECK_DT) / 0.3;
      if (p <= 0) return;
      const g = el("g", { opacity: Math.min(1, p) }, layer);
      el("text", { x: vc.x + 14, y: TOP + 74 + i * 22, style: `font: 700 13px var(--sans); fill: ${ok ? "var(--good)" : "var(--bad)"}` }, g, ok ? "✓" : "✗");
      el("text", { x: vc.x + 32, y: TOP + 74 + i * 22, class: "lora-fx__chip-sub" }, g, label);
    });
    const pr = (time - CHECK_T0 - s.checks.length * CHECK_DT - 0.2) / 0.4;
    if (pr > 0) {
      const g = el("g", { opacity: Math.min(1, pr) }, layer);
      el("text", { x: vc.x + 14, y: TOP + H - 16, class: "lora-fx__zone-label" }, g, "reward");
      el("text", { x: vc.x + vc.w - 16, y: TOP + H - 14, "text-anchor": "end", style: `font: 700 26px var(--mono); fill: ${col}` }, g, s.reward);
    }
  }

  function frame(now) {
    const dt = Math.min(0.25, (now - last) / 1000);
    last = now;
    if (!paused && onScreen) t += dt * speed;
    draw();
    requestAnimationFrame(frame);
  }
  if (reduced) draw(); else requestAnimationFrame(frame);

  tabs.forEach((b, i) => b.addEventListener("click", () => { t = i * SCENE; draw(); }));
  const play = root.querySelector("[data-tt-play]");
  play.addEventListener("click", () => { paused = !paused; play.textContent = paused ? "play" : "pause"; });
  root.querySelector("[data-tt-replay]").addEventListener("click", () => { t = Math.floor((t % LOOP) / SCENE) * SCENE; });
  const input = root.querySelector("[data-tt-speed]");
  const val = root.querySelector(".lora-fx__speed-val");
  input.addEventListener("input", () => { speed = +input.value; val.textContent = `${speed.toFixed(1)}×`; });
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; }).observe(root);
})();
