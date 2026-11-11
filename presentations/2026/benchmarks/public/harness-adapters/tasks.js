// A few traj2envs tasks, file by file, as the agent and the verifier see them (after the multi-harness RL post's task
// browser). The tasks are copied from harness-hub/traj2envs-rl into tasks.json, so the page needs no Hub request.
(() => {
  const root = document.getElementById("task-browser");
  if (!root) return;
  const FAMILY = { repair: "#8e6bd8", build: "#d29a12", extend: "#c2417a", qa: "#2f97b8" };
  const chips = root.querySelector(".tb-tasks");
  const summary = root.querySelector(".tb-q");
  const tree = root.querySelector(".tb-tree");
  const viewer = root.querySelector(".tb-file");
  const title = root.querySelector(".tb-path");
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  let tasks = [], current = null;

  // Files the reader opens first: what the agent is asked, then how it's graded
  const ORDER = ["instruction.md", "task.toml", "tests/test.sh", "environment/Dockerfile"];
  const rank = (p) => (ORDER.includes(p) ? ORDER.indexOf(p) : ORDER.length) + (p.split("/").length - 1) * 0.01;

  function showFile(path) {
    for (const b of tree.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.path === path));
    title.textContent = path;
    viewer.innerHTML = esc(current.files[path]);
  }

  function showTask(task) {
    current = task;
    for (const b of chips.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.id === task.id));
    const rate = task.solve_rate == null ? "" : `, solved in ${Math.round(100 * task.solve_rate)}% of attempts by LFM2.5-24B-A2B`;
    summary.innerHTML = `<b style="color:${FAMILY[task.family]}">${task.family}</b> · ${task.level ?? "no level"}${rate}.
      <span>From ${esc(task.source)}.</span>`;
    const paths = Object.keys(task.files).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
    tree.innerHTML = `<div class="tb-root">${esc(task.id)}/</div>` + paths.map((p) =>
      `<button type="button" class="tb-entry" data-path="${esc(p)}" style="padding-left:${10 + 12 * (p.split("/").length - 1)}px">${esc(p.split("/").pop())}${p.includes("/") ? `<span>${esc(p.slice(0, p.lastIndexOf("/")))}/</span>` : ""}</button>`).join("");
    for (const b of tree.querySelectorAll("button")) b.addEventListener("click", () => showFile(b.dataset.path));
    showFile(paths[0]);
  }

  fetch("tasks.json").then((r) => r.json()).then((data) => {
    tasks = data;
    chips.innerHTML = tasks.map((t) =>
      `<button type="button" class="tb-chip" data-id="${esc(t.id)}"><i style="background:${FAMILY[t.family]}"></i>${t.family} · ${t.level ?? "?"}</button>`).join("");
    for (const b of chips.querySelectorAll("button")) b.addEventListener("click", () => showTask(tasks.find((t) => t.id === b.dataset.id)));
    showTask(tasks[0]);
  }).catch(() => { summary.textContent = "The example tasks could not be loaded."; });
})();
