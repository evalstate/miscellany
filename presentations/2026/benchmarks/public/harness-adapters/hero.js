// The pull figure: "one base model, an adapter per harness", then "every checkpoint, evaluated in one job". It draws on
// the page itself, in the page's colors (light or dark), like the multi-harness RL post's banner. render(t) draws the
// frame at time t (seconds) deterministically, so video/render.py can also record it frame by frame.
window.mountHero = function mountHero(container, { title = false, logos = "logos" } = {}) {
  const W = 960, H = 540, TOP = title ? 0 : 52;  // without the title, the scene starts at the counters
  const HARNESSES = ["pi", "opencode", "hermes"];
  const LOGO = { pi: `${logos}/pi.svg`, opencode: `${logos}/opencode.png`, hermes: `${logos}/hermes.png` };
  const col = (h) => (h === "base" ? "var(--axis-ink)" : `var(--series-${h})`);
  const tint = (c, a) => `color-mix(in oklab, ${c} ${a}%, var(--page))`;
  const T_TRAIN_END = 8.2, T_EVAL = 8.9, T_END = 14.6;
  // LFM3-1.7B held-out pass@1 on 150 SmolDataEnvs tasks (eval/*_final-heldout/summary.json)
  const EVALS = [
    { label: "base model", h: "base", pass: (0.7 + 0.62 + 0.693) / 3, note: "avg" },
    { label: "pi · ckpt 10", h: "pi", pass: 0.72 }, { label: "pi · ckpt 20", h: "pi", pass: 0.687 },
    { label: "opencode · ckpt 10", h: "opencode", pass: 0.687 }, { label: "opencode · ckpt 20", h: "opencode", pass: 0.785 },
    { label: "hermes · ckpt 10", h: "hermes", pass: 0.64 }, { label: "hermes · ckpt 20", h: "hermes", pass: 0.687 },
  ];

  // A fixed pseudo-random schedule, so every render of time t is the same frame
  let seed = 7;
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const span = (t, a, b) => clamp((t - a) / (b - a));

  container.classList.add("hs");
  container.innerHTML = "";
  const stage = document.createElement("div");
  stage.className = "hs-stage";
  stage.style.top = `${-TOP}px`;
  container.append(stage);
  const el = (cls, parent, html = "", style = {}) => {
    const e = document.createElement("div");
    e.className = cls;
    e.innerHTML = html;
    Object.assign(e.style, style);
    parent.append(e);
    return e;
  };
  const logo = (h, size) => `<img class="hs-logo" src="${LOGO[h]}" alt="" style="width:${size}px;height:${size}px" data-h="${h}">`;

  const cv = document.createElement("canvas");
  cv.width = W * 2; cv.height = H * 2;
  stage.append(cv);
  const titleEl = el("hs-title", stage);
  titleEl.hidden = !title;
  const counters = el("hs-counters", stage);
  const train = el("hs-scene", stage);
  const evalScene = el("hs-scene", stage);

  // ─── Training scene ───
  el("hs-label", train, "agent harnesses", { left: "36px", top: "100px" });
  el("hs-label", train, "model server · one vLLM · GPU 0", { left: "566px", top: "100px" });
  el("hs-label", train, "trainers · one LoRA adapter each", { left: "566px", top: "322px" });
  const cardY = { pi: 120, opencode: 210, hermes: 300 };
  const cards = {}, slots = {}, trainers = {};
  for (const h of HARNESSES) {
    const c = el("hs-card hs-harness", train, `<div class="hs-name">${logo(h, 26)}${h}</div>
      <div class="hs-sub">8 sessions per task</div><div class="hs-tag"></div><div class="hs-bars"></div>`, { top: `${cardY[h]}px` });
    cards[h] = { tag: c.querySelector(".hs-tag"), bars: c.querySelector(".hs-bars") };
  }
  el("hs-card hs-server", train, `<div class="hs-base"><span>base weights · loaded once · frozen</span></div>`);
  const slotY = { pi: 172, opencode: 218, hermes: 264 };
  for (const h of HARNESSES) slots[h] = [0, 1].map(() => el("hs-slot", train, "", { borderColor: col(h), top: `${slotY[h]}px`, width: "150px" }));
  HARNESSES.forEach((h, i) => {
    const t = el("hs-card hs-trainer", train, `<b>${logo(h, 14)}${h}</b><small></small><div class="hs-track"><div class="hs-fill" style="background:${col(h)}"></div></div>`, { left: `${566 + i * 123}px` });
    trainers[h] = { sub: t.querySelector("small"), fill: t.querySelector(".hs-fill") };
  });
  el("hs-label", train, "<span class='hs-live'>●</span> live · at the model server", { left: "36px", top: "406px" });
  const feedTop = el("hs-feed", train, "", { top: "420px" });
  const feedBot = el("hs-feed", train, "", { top: "480px" });
  // Every request crosses safeagents on its way to vLLM: the hub is where tool calls are checked
  el("hs-label hs-guard", train, "safeagents", { left: `${486 - 8}px`, top: "238px" });
  el("hs-guard-note", train, "guard: every tool call<br>checked before it runs", { left: `${486 - 170}px`, top: "272px", width: "152px" });

  // Requests: each harness sends one every ~0.32 s; each names its harness's current adapter
  const UPDATES = [];
  for (const [k, base] of [[2, 2.7], [3, 5.7]]) HARNESSES.forEach((h, i) => UPDATES.push({ h, v: k, t: base + i * 0.45 }));
  const version = (h, t) => 1 + UPDATES.filter((u) => u.h === h && u.t <= t).length;
  const REQS = [];
  for (const h of HARNESSES) {
    let t = 0.5 + rand() * 0.3;
    while (t < T_TRAIN_END - 1) { REQS.push({ h, t }); t += 0.24 + rand() * 0.2; }
  }
  REQS.sort((a, b) => a.t - b.t);

  // ─── Evaluation scene ───
  el("hs-label", evalScene, "one sandbox server per checkpoint", { left: "36px", top: "100px" });
  el("hs-label", evalScene, "model server · one vLLM · GPU 0", { left: "566px", top: "100px" });
  el("hs-card", evalScene, `<div class="hs-base" style="margin:10px 12px 0"><span>LFM3-1.7B base weights · loaded once</span></div>`, { left: "566px", top: "122px", width: "358px", height: "296px" });
  const evalRows = EVALS.map((e, i) => {
    const y = 124 + i * 41;
    el("hs-card hs-ckpt", evalScene, `<span>${e.h === "base" ? "" : logo(e.h, 16)}${e.label}</span><small>150 tasks</small>`, { top: `${y}px`, borderColor: col(e.h), background: tint(col(e.h), 10) });
    const chip = el("hs-slot", evalScene, `<b>${e.h === "base" ? "no adapter" : "adapter"}</b>`, { left: "578px", top: `${172 + i * 34}px`, height: "28px", width: "118px", borderColor: col(e.h), background: tint(col(e.h), 10) });
    const score = el("hs-score", evalScene, `<div class="hs-bar"><i style="background:${col(e.h)}"></i></div><span></span>`, { left: "710px", top: `${173 + i * 34}px` });
    return { y: y + 15, chip, fill: score.querySelector("i"), val: score.querySelector("span"), e };
  });
  el("hs-label", evalScene, "<span class='hs-live'>●</span> live · scored in the same job", { left: "36px", top: "432px" });
  const evalFeed = el("hs-feed", evalScene, "", { top: "452px" });

  // The canvas reads the page's colors on every frame, so a theme switch shows at once
  const ctx = cv.getContext("2d");
  ctx.scale(2, 2);
  const NODE = { x: 486, y: 255 };
  let C = {};
  function readColors() {
    const s = getComputedStyle(container);
    const v = (n) => s.getPropertyValue(n).trim();
    C = { page: v("--page"), line: v("--border"), muted: v("--text-muted"), accent: v("--fx-accent"), base: v("--axis-ink"),
          pi: v("--series-pi"), opencode: v("--series-opencode"), hermes: v("--series-hermes") };
  }
  function curve(ax, ay, bx, by, color, alpha) {
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.bezierCurveTo((ax + bx) / 2, ay, (ax + bx) / 2, by, bx, by); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  function along(ax, ay, bx, by, p) {
    const q = 1 - p, mx = (ax + bx) / 2;
    return [q ** 3 * ax + 3 * q * q * p * mx + 3 * q * p * p * mx + p ** 3 * bx, q ** 3 * ay + 3 * q * q * p * ay + 3 * q * p * p * by + p ** 3 * by];
  }
  function dot(x, y, color, r = 4.5) {
    ctx.fillStyle = C.page; ctx.beginPath(); ctx.arc(x, y, r + 1.6, 0, 7); ctx.fill();
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  }
  function hub(t, active) {
    const pulse = (t * 1.6) % 1;
    ctx.globalAlpha = 0.5 * (1 - pulse) * active; ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(NODE.x, NODE.y, 10 + pulse * 18, 0, 7); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillStyle = C.page; ctx.strokeStyle = C.accent; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(NODE.x, NODE.y, 9, 0, 7); ctx.fill(); ctx.stroke();
  }
  function feedRow(node, color, who, sub, what, badge, note) {
    node.style.setProperty("--c", color);
    node.style.borderColor = tint(color, 45);
    node.style.background = tint(color, 6);
    node.innerHTML = `<div class="hs-who">${who}<span>${sub}</span></div><div class="hs-what">${what}</div>
      <div class="hs-badge" style="background:${color}">${badge}</div>${note ? `<div class="hs-note">${note}</div>` : ""}`;
  }

  function render(t) {
    readColors();
    ctx.clearRect(0, 0, W, H);
    if (t < T_EVAL) {
      ctx.globalAlpha = 1 - span(t, T_TRAIN_END, T_EVAL);
      ctx.strokeStyle = C.line; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(NODE.x, 108); ctx.lineTo(NODE.x, 398); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    const fade = span(t, T_TRAIN_END, T_EVAL);
    train.style.opacity = 1 - fade;
    evalScene.style.opacity = fade;
    if (t < T_EVAL) {
      titleEl.textContent = "One base model, an adapter per harness";
      const done = REQS.filter((r) => r.t + 0.9 <= t).length;
      const updates = UPDATES.filter((u) => u.t <= t).length;
      counters.innerHTML = `<b>3</b>harnesses<i>·</i><b>1</b>base model<i>·</i><b>${done}</b>requests<i>·</i><b class="hs-accent">${updates}</b>adapter updates`;
      for (const h of HARNESSES) {
        curve(326, cardY[h] + 39, NODE.x, NODE.y, C[h], 0.3 * (1 - fade));
        curve(NODE.x, NODE.y, 578, slotY[h] + 18, C[h], 0.3 * (1 - fade));
      }
      for (const r of REQS) {
        const p = (t - r.t) / 0.9;
        if (p < 0 || p > 1) continue;
        const a = [326, cardY[r.h] + 39], m = [NODE.x, NODE.y], b = [578, slotY[r.h] + 18];
        const [x, y] = p < 0.6 ? along(...a, ...m, ease(p / 0.6)) : along(...m, ...b, ease((p - 0.6) / 0.4));
        dot(x, y, C[r.h], 4);
      }
      hub(t, 1 - fade);  // drawn over the packets, so they pass under it
      for (const h of HARNESSES) {
        const v = version(h, t);
        cards[h].tag.textContent = `→ ${h}-v${v}`;
        cards[h].tag.style.cssText = `color:${col(h)};background:${tint(col(h), 14)}`;
        const mine = REQS.filter((r) => r.h === h && r.t <= t).slice(-26);
        cards[h].bars.innerHTML = mine.map(() => `<i style="background:${col(h)}"></i>`).join("");
        // The adapter being served, and the next one arriving next to it
        const last = UPDATES.filter((u) => u.h === h && u.t <= t).pop();
        const [cur, next] = slots[h];
        const since = last ? t - last.t : 99;
        cur.innerHTML = since < 1.4 ? `<b>${h} · v${v - 1}</b><small>finishing in-flight</small>` : `<b>${h} · v${v}</b><small>serving</small>`;
        cur.style.left = "578px";
        cur.style.opacity = since < 0.9 ? 1 : since < 1.4 ? 1 - (since - 0.9) / 0.5 : 1;
        cur.style.background = tint(col(h), 10);
        next.innerHTML = `<b>${h} · v${v}</b><small>${since < 0.5 ? "loading" : "serving"}</small>`;
        next.style.background = tint(col(h), 10);
        next.style.opacity = since < 1.4 ? span(since, 0, 0.3) : 0;
        next.style.left = `${740 - 162 * ease(span(since, 0.9, 1.4))}px`;
        // Trainer: fills as its rollouts arrive, then sends the new adapter up
        const nextU = UPDATES.find((u) => u.h === h && u.t > t);
        const prevT = last ? last.t : 0.5;
        trainers[h].fill.style.width = `${nextU ? 100 * span(t, prevT + 0.3, nextU.t - 0.6) : 100 * span(t, prevT + 0.3, T_TRAIN_END)}%`;
        trainers[h].sub.textContent = nextU && t > nextU.t - 0.6 ? `→ v${v + 1}` : `adapter v${v}`;
        if (nextU && t > nextU.t - 0.6) {
          const i = HARNESSES.indexOf(h), p = ease(span(t, nextU.t - 0.6, nextU.t));
          const ax = 566 + i * 123 + 56, ay = 338, bx = 815, by = slotY[h] + 18;
          dot(ax + (bx - ax) * p, ay + (by - ay) * p, C[h], 5.5);
        }
      }
      const lastU = UPDATES.filter((u) => u.t <= t).pop();
      const lastR = REQS.filter((r) => r.t <= t).pop();
      if (lastU && t - lastU.t < 2.2) feedRow(feedTop, col(lastU.h), `trainer · ${lastU.h}`, "gradient step", `load_lora_adapter ${lastU.h}-v${lastU.v}`, "NEW ADAPTER", `requests on v${lastU.v - 1} finish on v${lastU.v - 1}`);
      else feedRow(feedTop, "var(--axis-ink)", "vLLM", "one server", "base weights loaded once, shared by every harness", "SHARED", "");
      if (lastR) {
        const cmd = { pi: "bash · python solve.py --check", opencode: "read · data/train.csv", hermes: "terminal · pytest -q" }[lastR.h];
        feedRow(feedBot, col(lastR.h), lastR.h, "request", cmd, `ADAPTER ${lastR.h}-v${version(lastR.h, lastR.t)}`.toUpperCase(), "");
      }
    } else {
      titleEl.textContent = "…and every checkpoint, evaluated in one job";
      const sessions = Math.round(1350 * span(t, T_EVAL + 0.6, T_END - 1.4));
      counters.innerHTML = `<b>7</b>policies<i>·</i><b>1</b>GPU<i>·</i><b>${sessions.toLocaleString("en-US")}</b>agent sessions<i>·</i><b class="hs-accent">0</b>merged copies`;
      evalRows.forEach((r, i) => {
        const appear = span(t, T_EVAL + i * 0.12, T_EVAL + i * 0.12 + 0.35);
        r.chip.style.opacity = appear;
        curve(286, r.y, 578, 186 + i * 34, C[r.e.h] || C.base, 0.25 * appear);
        for (let k = 0; k < 3; k++) {
          const p = ((t - T_EVAL - 0.5 - i * 0.07 - k * 0.33) / 1.0) % 1;
          if (t < T_EVAL + 0.5 || t > T_END - 1.2 || p < 0) continue;
          const [x, y] = along(286, r.y, 578, 186 + i * 34, ease(p));
          dot(x, y, C[r.e.h] || C.base, 3.6);
        }
        r.fill.style.width = `${100 * r.e.pass * ease(span(t, T_EVAL + 0.7, T_END - 1.4))}%`;
        r.val.textContent = t > T_END - 1.4 ? `${Math.round(100 * r.e.pass)}%${r.e.note ? " " + r.e.note : ""}` : "";
      });
      if (t > T_END - 1.4) feedRow(evalFeed, col("opencode"), "opencode · ckpt 20", "held-out pass@1", "69% → 79% against the base model, +9.4 ± 6.3 points", "SCORED", "");
      else feedRow(evalFeed, "var(--axis-ink)", "7 policies", "same tasks, same job", "each request names its checkpoint's adapter", "RUNNING", "");
    }
  }

  // Fit the 960 px scene to the container's width
  const fit = () => {
    const s = container.clientWidth / W;
    stage.style.transform = `scale(${s})`;
    container.style.height = `${(H - TOP) * s}px`;
  };
  fit();
  new ResizeObserver(fit).observe(container);
  return { render, duration: T_END };
};
