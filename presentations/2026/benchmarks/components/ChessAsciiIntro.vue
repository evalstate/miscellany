<script setup lang="ts">
// Opening loop: the chess-best-move board image, then the ASCII renders that
// agents printed of it, each shown with the harness/model pair it came from.
//
// Integrity rules for this component:
// - Every character comes from data/chess-intro.json, which
//   analysis/chess-intro/extract.py copies from masked tool output. Nothing
//   here generates, repairs or restyles characters.
// - Transitions only ever show a cell from the outgoing or the incoming
//   frame. There is no scramble or tween between characters.
// - Each grid is drawn over the part of the board its script sampled, so the
//   renders stay registered with the image and with each other.
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useIsSlideActive } from "@slidev/client";
import raw from "../data/chess-intro.json";

// `full`: a bare full-slide terminal (the opening slide; use with
// `class: fa-chess-slide`). Without it: board on the left, explanatory copy
// on the right (use with `class: fa-chess-panel`). The default slot is laid
// over the terminal, for a title.
// `scheme` picks the terminal colours (defined as --term-* in style.css):
// "petrol" is the deck's own, "green" and "amber" are phosphor. `raster`
// lays scan lines over the full-screen version.
const props = withDefaults(defineProps<{ full?: boolean; scheme?: "petrol" | "green" | "amber"; raster?: boolean }>(), { full: false, scheme: "petrol", raster: false });

type Block = { cols: number; rows: number; region: number[]; lines: string[] };
type Board = { cols: number; rows: number; gap: number[] | null; lines: string[] };
type Source = { harness: string; version: string; model: string; step: number; date: string; reward: number };
const data = raw as unknown as {
  fen: string;
  image: string;
  sources: Record<string, Source>;
  boards: Record<string, Board>;
  squares: Record<string, Record<string, Block>>;
};

const B = 640; // board image size in px; all geometry is in image pixels
const S = 80;
const FULL = { x: 0, y: 0, w: B };
const sq = (name: string) => ({ x: (name.charCodeAt(0) - 97) * S, y: (8 - Number(name[1])) * S, w: S });

type Content =
  | { kind: "image" }
  | { kind: "pixels"; n: number }
  | { kind: "board"; id: string }
  | { kind: "square"; sq: string; id: string }
  | { kind: "mosaic" };
type Pattern = "scan" | "noise" | "diag" | "radial";
type Shot = {
  content: Content;
  cam: { x: number; y: number; w: number };
  trans: number; // seconds to replace the previous frame
  pattern: Pattern;
  move?: number; // seconds for the camera to arrive
  hold: number;
  label: string;
  title: string;
  body: string;
  note: string; // the one line the full-screen version shows
  quote?: string;
  src?: string;
};

// The mosaic gives each occupied square to the source used least so far.
const occupied: string[] = [];
data.fen.split("/").forEach((row, r) => {
  let c = 0;
  for (const ch of row) {
    if (/\d/.test(ch)) c += Number(ch);
    else occupied.push("abcdefgh"[c++] + (8 - r));
  }
});
const mosaic: { sq: string; id: string }[] = [];
const used: Record<string, number> = {};
for (const name of [...occupied].sort((a, b) => Object.keys(data.squares[a] ?? {}).length - Object.keys(data.squares[b] ?? {}).length)) {
  const ids = Object.keys(data.squares[name] ?? {});
  if (!ids.length) continue;
  const id = ids.reduce((best, cur) => ((used[cur] ?? 0) < (used[best] ?? 0) ? cur : best));
  used[id] = (used[id] ?? 0) + 1;
  mosaic.push({ sq: name, id });
}
const mosaicIds = Object.keys(data.sources).filter((id) => used[id]);

const D8 = sq("d8");
const closeNotes: Record<string, string> = {"ds4f": "mask of what differs from the corners, thickened 1 px", "grok46": "board colours dropped, rest by brightness", "claude47": "# where the square differs from its background", "gpt55": "# piece, + edge, 32 chars across", "gem3": "ramp starts with a space: black prints as nothing", "glm51": "every pixel across, every 2nd row down"};
const closeUps: [string, string][] = [
  ["ds4f", "A mask of everything that differs from the square's corners, thickened by one pixel."],
  ["grok46", "Board colours are dropped. What's left is drawn by brightness."],
  ["claude47", "One character per two pixels: # wherever the square differs from its background."],
  ["gpt55", "32 characters across. # for dark piece pixels, + where only an edge falls in the cell."],
  ["gem3", "This ramp starts with a space, so the darkest pixels print as nothing. The black queen all but vanishes."],
  ["glm51", "Every pixel across and every second row down, so the queen prints 60 wide."],
];

const shots: Shot[] = [
  {
    content: { kind: "image" }, cam: FULL, trans: 1.6, pattern: "noise", move: 0.01, hold: 3.2,
    label: "chess_board.png",
    note: "640 × 640 px. White to move.",
    title: "The task is a picture",
    body: "A 640 × 640 PNG of a chess position. White to move: write the best move to a file.",
  },
  {
    content: { kind: "pixels", n: 128 }, cam: FULL, trans: 1.4, pattern: "noise", hold: 0.7,
    label: "greyscale, 128 × 128",
    note: "greyscale, 128 × 128",
    title: "This model can't see it",
    body: "It is text only, so it shrinks the image to greyscale and prints the brightness.",
    quote: "I can try to output a downsampled ASCII art representation to understand the position.",
    src: "ds4f-board",
  },
  {
    content: { kind: "board", id: "ds4f-board" }, cam: FULL, trans: 2.6, pattern: "scan", hold: 3.4,
    label: "step 4 · tool output",
    note: "128 × 128 chars. 512 bytes cut by the harness output cap.",
    title: "So it prints the board",
    body: "128 × 128 characters. The harness capped the output and cut 512 bytes from the middle: the model never saw those rows.",
    src: "ds4f-board",
  },
  {
    content: { kind: "board", id: "ds4f-board" }, cam: { x: 160, y: 0, w: 240 }, trans: 0.01, pattern: "noise", move: 2.6, hold: 1.6,
    label: "c8 to e8 · step 4",
    note: "16 chars to a square",
    title: "So it prints the board",
    body: "Sixteen characters to a square. Bishop, queen, and an empty e8.",
    src: "ds4f-board",
  },
  {
    content: { kind: "board", id: "glm52-board" }, cam: FULL, trans: 1.6, pattern: "diag", move: 2.2, hold: 2.6,
    label: "step 6 · tool output",
    note: "80 × 40 chars",
    title: "It isn't the only one",
    body: "80 × 40 characters: ten across each square and five down.",
    src: "glm52-board",
  },
  {
    content: { kind: "board", id: "gem31-board" }, cam: FULL, trans: 1.6, pattern: "diag", hold: 2.6,
    label: "step 4 · tool output",
    note: "80 × 40 chars, ramp reversed",
    title: "Same idea, ramp reversed",
    body: "80 × 40 again, but here dark pixels get the heavy characters.",
    quote: "Numpy is not installed, so I will write a script using just the standard library and Pillow.",
    src: "gem31-board",
  },
  {
    content: { kind: "board", id: "gem31-board" }, cam: D8, trans: 0.01, pattern: "noise", move: 2.4, hold: 0.9,
    label: "d8 · step 4",
    note: "the queen is 10 chars wide",
    title: "Fifty characters aren't enough",
    body: "At this size the queen on d8 is ten characters wide. So the agents crop one square at a time.",
    src: "gem31-board",
  },
  ...closeUps.map(([id, body], i): Shot => ({
    content: { kind: "square", sq: "d8", id }, cam: D8, trans: 1.0, pattern: i % 2 ? "noise" : "radial", hold: 2.3,
    label: `d8 · step ${data.sources[id].step}`,
    title: "One queen, six renderers",
    body,
    note: closeNotes[id],
    src: id,
  })),
  {
    content: { kind: "mosaic" }, cam: FULL, trans: 1.2, pattern: "radial", move: 2.8, hold: 5,
    label: "25 pieces, as printed",
    note: "every piece is one agent's own print of that square",
    title: `One position, ${mosaicIds.length} traces`,
    body: "Every piece below is one agent's own print of that square.",
  },
];
const starts: number[] = [];
let total = 0;
for (const s of shots) {
  starts.push(total);
  total += Math.max(s.trans, s.move ?? 0) + s.hold;
}
const POSTER = starts[2] + shots[2].trans + 0.1; // still frame for overview, export and inactive slides

const canvas = ref<HTMLCanvasElement>();
const index = ref(2);
const paused = ref(false);
const active = useIsSlideActive();
const shot = computed(() => shots[index.value]);
const source = computed(() => (shot.value.src ? data.sources[shot.value.src] : undefined));
const legend = computed(() => mosaicIds.map((id) => data.sources[id]));
// The terminal has one narrow line for the model, so effort and release-stage
// suffixes are dropped there: "DeepSeek V4 Flash (max)" -> "DeepSeek V4 Flash".
const short = (model: string) => model.replace(/\s*\((max|high|medium|low)\)$/i, "").replace(/\s+preview$/i, "");

// Backing store; the slide is scaled up by Slidev on a projector.
const PXH = 1380;
const PXW = props.full ? Math.round((PXH * 980) / 552) : PXH;
let ctx: CanvasRenderingContext2D;
let img: HTMLImageElement | undefined;
let grey: Uint8Array | undefined;
let colours = { ink: "#fff", ground: "#000", raised: "#111" };
let fontFamily = "monospace";
let raf = 0;
let clock = 0;
let last = 0;

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const clamp = (t: number) => Math.min(1, Math.max(0, t));
const hash = (x: number, y: number) => {
  const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return s - Math.floor(s);
};

// Order in which cells flip, 0..1, for a point in view coordinates (0..1).
function order(p: Pattern, u: number, v: number, x: number, y: number) {
  if (p === "scan") return v * 0.99 + u * 0.01;
  if (p === "noise") return hash(Math.round(x), Math.round(y));
  if (p === "diag") return (u + v) * 0.46 + hash(Math.round(x), Math.round(y)) * 0.08;
  return Math.hypot(u - 0.5, v - 0.5) * 1.2 + hash(Math.round(x), Math.round(y)) * 0.15;
}

// Glyphs are stamped from a small atlas rebuilt whenever the cell size changes
// (every frame during a zoom), so text stays sharp at any magnification.
const CHARS = " .:-=+*#%@";
let atlas: { key: string; ink: HTMLCanvasElement; inv: HTMLCanvasElement; w: number; h: number } | undefined;
function glyphs(cw: number, ch: number) {
  const key = `${cw.toFixed(2)}x${ch.toFixed(2)}|${fontFamily}`;
  if (atlas?.key === key) return atlas;
  const w = Math.ceil(cw) + 2;
  const h = Math.ceil(ch) + 2;
  const make = (fill: string) => {
    const c = document.createElement("canvas");
    c.width = w * CHARS.length;
    c.height = h;
    const g = c.getContext("2d")!;
    g.fillStyle = fill;
    // Below about 12 device px a glyph thins out to grey, so small cells get a
    // slightly oversized glyph (same character, more ink) to hold contrast.
    const fit = Math.min(ch, cw / 0.6);
    g.font = `500 ${fit * (0.98 + 0.5 * Math.min(1, Math.max(0, (12 - fit) / 8)))}px ${fontFamily}`;
    g.textAlign = "center";
    g.textBaseline = "middle";
    for (let i = 1; i < CHARS.length; i++) g.fillText(CHARS[i], i * w + w / 2, h / 2 + ch * 0.04);
    return c;
  };
  atlas = { key, ink: make(colours.ink), inv: make(colours.ground), w, h };
  return atlas;
}

type Test = ((x: number, y: number) => number) | null; // 0 hidden, 1 shown, 2 leading edge
type View = { x: number; y: number; k: number }; // board px -> canvas px

function drawGrid(lines: string[], cols: number, rows: number, rx: number, ry: number, rw: number, rh: number, view: View, test: Test) {
  const cw = (rw / cols) * view.k;
  const ch = (rh / rows) * view.k;
  const ox = (rx - view.x) * view.k;
  const oy = (ry - view.y) * view.k;
  const c0 = Math.max(0, Math.floor(-ox / cw));
  const c1 = Math.min(cols, Math.ceil((PXW - ox) / cw));
  const r0 = Math.max(0, Math.floor(-oy / ch));
  const r1 = Math.min(rows, Math.ceil((PXH - oy) / ch));
  const a = glyphs(cw, ch);
  for (let r = r0; r < r1; r++) {
    const line = lines[r];
    const y = oy + r * ch;
    for (let c = c0; c < c1; c++) {
      const x = ox + c * cw;
      const state = test ? test(rx + ((c + 0.5) * rw) / cols, ry + ((r + 0.5) * rh) / rows) : 1;
      if (!state) continue;
      const i = CHARS.indexOf(line[c]);
      if (state === 2) {
        ctx.fillStyle = colours.ink;
        ctx.fillRect(x, y, cw + 0.5, ch + 0.5);
        if (i > 0) ctx.drawImage(a.inv, i * a.w, 0, a.w, a.h, Math.round(x - 1), Math.round(y - 1), a.w, a.h);
      } else if (i > 0) {
        ctx.drawImage(a.ink, i * a.w, 0, a.w, a.h, Math.round(x - 1), Math.round(y - 1), a.w, a.h);
      }
    }
  }
}

function drawBlock(name: string, id: string, view: View, test: Test) {
  const b = data.squares[name]?.[id];
  if (!b) return;
  const o = sq(name);
  const [x0, y0, x1, y1] = b.region;
  drawGrid(b.lines, b.cols, b.rows, o.x + x0, o.y + y0, x1 - x0, y1 - y0, view, test);
}

function drawContent(c: Content, view: View, test: Test) {
  if (c.kind === "image") {
    if (!img) return;
    if (!test) {
      ctx.drawImage(img, -view.x * view.k, -view.y * view.k, B * view.k, B * view.k);
      return;
    }
    const n = 80;
    const step = B / n;
    const d = step * view.k;
    for (let r = 0; r < n; r++)
      for (let q = 0; q < n; q++) {
        const state = test((q + 0.5) * step, (r + 0.5) * step);
        if (state === 1) ctx.drawImage(img, q * step, r * step, step, step, (q * step - view.x) * view.k, (r * step - view.y) * view.k, d + 0.5, d + 0.5);
        else if (state === 2) {
          ctx.fillStyle = colours.ink;
          ctx.fillRect((q * step - view.x) * view.k, (r * step - view.y) * view.k, d + 0.5, d + 0.5);
        }
      }
  } else if (c.kind === "pixels") {
    if (!grey) return;
    const step = B / c.n;
    const d = step * view.k;
    for (let r = 0; r < c.n; r++)
      for (let q = 0; q < c.n; q++) {
        const state = test ? test((q + 0.5) * step, (r + 0.5) * step) : 1;
        if (!state) continue;
        // Brightness as the amount of terminal ink, so it follows the colour scheme.
        ctx.globalAlpha = state === 2 ? 1 : grey[r * c.n + q] / 255;
        ctx.fillStyle = colours.ink;
        ctx.fillRect((q * step - view.x) * view.k, (r * step - view.y) * view.k, d + 0.5, d + 0.5);
      }
    ctx.globalAlpha = 1;
  } else if (c.kind === "board") {
    const b = data.boards[c.id];
    if (b.gap) {
      // Rows the harness cut from the output: left empty and marked, never restored.
      const y = ((b.gap[0] * B) / b.rows - view.y) * view.k;
      const h = (((b.gap[1] - b.gap[0]) * B) / b.rows) * view.k;
      ctx.fillStyle = colours.raised;
      if (!test || test(B / 2, (b.gap[0] * B) / b.rows)) ctx.fillRect(-view.x * view.k, y, B * view.k, h);
    }
    drawGrid(b.lines, b.cols, b.rows, 0, 0, B, B, view, test);
  } else if (c.kind === "square") {
    // Full screen is wider than one square, so show the same source's neighbours too.
    if (props.full) for (const name in data.squares) drawBlock(name, c.id, view, test);
    else drawBlock(c.sq, c.id, view, test);
  } else {
    for (const m of mosaic) drawBlock(m.sq, m.id, view, test);
  }
}

function frame(t: number) {
  let i = starts.length - 1;
  while (i > 0 && starts[i] > t) i--;
  if (index.value !== i) index.value = i;
  const s = shots[i];
  const prev = shots[(i + shots.length - 1) % shots.length];
  const local = t - starts[i];
  const m = ease(clamp(local / (s.move ?? 0.01)));
  const cam = { x: prev.cam.x + (s.cam.x - prev.cam.x) * m, y: prev.cam.y + (s.cam.y - prev.cam.y) * m, w: prev.cam.w * Math.pow(s.cam.w / prev.cam.w, m) };
  // cam is the square of interest; the view is centred on it and fills the canvas.
  const side = cam.w * (props.full ? 1.05 : 1);
  const k = PXH / side;
  const view = { x: cam.x + cam.w / 2 - PXW / k / 2, y: cam.y + cam.w / 2 - side / 2, k };
  const p = clamp(local / s.trans);

  ctx.fillStyle = colours.ground;
  ctx.fillRect(0, 0, PXW, PXH);
  if (p >= 1) {
    drawContent(s.content, view, null);
    return;
  }
  const front = (s.pattern === "scan" ? p : ease(p)) * 1.06 - 0.03;
  const band = s.pattern === "scan" ? 0.012 : 0.03;
  const f = (x: number, y: number) => order(s.pattern, ((x - view.x) * k) / PXW, ((y - view.y) * k) / PXH, x, y);
  drawContent(prev.content, view, (x, y) => (f(x, y) > front ? 1 : 0));
  drawContent(s.content, view, (x, y) => {
    const d = front - f(x, y);
    return d < 0 ? 0 : d < band ? 2 : 1;
  });
}

function tick(now: number) {
  raf = 0;
  if (!active.value || paused.value) return;
  clock = (clock + Math.min(0.1, (now - last) / 1000)) % total;
  last = now;
  frame(clock);
  raf = requestAnimationFrame(tick);
}
function run() {
  if (raf || !ctx) return;
  if (active.value && !paused.value) {
    last = performance.now();
    raf = requestAnimationFrame(tick);
  } else if (!active.value) {
    clock = 0;
    frame(POSTER);
  }
}
function toggle() {
  paused.value = !paused.value;
  run();
}

onMounted(async () => {
  ctx = canvas.value!.getContext("2d")!;
  const css = getComputedStyle(canvas.value!);
  colours = { ink: css.getPropertyValue("--term-ink").trim(), ground: css.getPropertyValue("--term-ground").trim(), raised: css.getPropertyValue("--term-raised").trim() };
  fontFamily = css.getPropertyValue("--font-code").trim() || "monospace";
  const picture = new Image();
  picture.src = `${import.meta.env.BASE_URL}${data.image}`;
  await Promise.all([picture.decode().catch(() => undefined), document.fonts?.load(`500 20px ${fontFamily}`).catch(() => undefined)]);
  if (picture.naturalWidth) {
    img = picture;
    // Luminance grid for the "greyscale" beat: Rec. 601, as PIL's convert('L') uses.
    const n = 128;
    const off = document.createElement("canvas");
    off.width = off.height = n;
    const g = off.getContext("2d", { willReadFrequently: true })!;
    g.drawImage(picture, 0, 0, n, n);
    const px = g.getImageData(0, 0, n, n).data;
    grey = new Uint8Array(n * n);
    for (let j = 0; j < n * n; j++) grey[j] = Math.round(0.299 * px[j * 4] + 0.587 * px[j * 4 + 1] + 0.114 * px[j * 4 + 2]);
  }
  atlas = undefined;
  frame(POSTER);
  run();
});
watch(active, run);
onBeforeUnmount(() => cancelAnimationFrame(raf));
</script>

<template>
  <div v-if="full" class="fa-chess-term" :class="`fa-term--${scheme}`" @click.stop="toggle">
    <canvas ref="canvas" :width="PXW" :height="PXH" />
    <div v-if="$slots.default" class="fa-chess-overlay"><slot /></div>
    <ul v-if="shot.content.kind === 'mosaic'" class="fa-chess-term-legend">
      <li v-for="s in legend" :key="s.harness + s.model + s.step">
        <b>{{ short(s.model) }}</b>
        {{ s.harness }} {{ s.version }}
      </li>
    </ul>
    <div class="fa-chess-status">
      <p class="fa-chess-prompt"><span aria-hidden="true">❯</span> {{ shot.label }}<i v-if="!paused" aria-hidden="true" /></p>
      <template v-if="source">
        <p class="fa-chess-model">{{ short(source.model) }}</p>
        <p>{{ source.harness }} {{ source.version }}</p>
        <p>step {{ source.step }} · {{ source.date }}</p>
        <p>n 1 · reward {{ source.reward }}</p>
      </template>
      <template v-else-if="shot.content.kind === 'mosaic'">
        <p>n {{ legend.length }}, 1 trial each</p>
        <p>30 Apr – 21 Sep 2026</p>
      </template>
      <p v-else>regenerated from the task's generator</p>
      <p>TB 2.1 chess-best-move</p>
      <p class="fa-chess-note">{{ shot.note }}</p>
    </div>
    <div v-if="raster" class="fa-chess-raster" aria-hidden="true" />
  </div>

  <div v-if="!full" class="fa-chess-intro">
    <figure class="fa-chess-stage" :class="`fa-term--${scheme}`" @click.stop="toggle">
      <figcaption>
        <span>{{ shot.label }}</span>
        <span v-if="paused">paused</span>
      </figcaption>
      <canvas ref="canvas" :width="PX" :height="PX" />
    </figure>

    <div class="fa-chess-copy">
      <h1>{{ shot.title }}</h1>
      <p class="fa-chess-body">{{ shot.body }}</p>
      <blockquote v-if="shot.quote">“{{ shot.quote }}”</blockquote>

      <dl v-if="source" class="fa-chess-source">
        <div>
          <dt>model</dt>
          <dd>{{ source.model }}</dd>
        </div>
        <div>
          <dt>harness</dt>
          <dd class="is-second">{{ source.harness }} <span>{{ source.version }}</span></dd>
        </div>
      </dl>

      <ul v-if="shot.content.kind === 'mosaic'" class="fa-chess-legend">
        <li v-for="s in legend" :key="s.harness + s.model + s.step">
          <b>{{ s.model }}</b> {{ s.harness }}
        </li>
      </ul>

      <div class="fa-chess-receipt">
        <Receipt v-if="source" n="1 trial" :date="source.date" :model="source.model" :harness="`${source.harness} ${source.version} · TB 2.1 chess-best-move`">
          step {{ source.step }} · reward {{ source.reward }} · characters verbatim
        </Receipt>
        <Receipt v-else-if="shot.content.kind === 'mosaic'" :n="`${legend.length} trials, 1 each`" date="30 Apr – 21 Sep 2026" model="as listed" harness="as listed" source="TB 2.1 chess-best-move tool output, characters verbatim" />
        <Receipt v-else source="TB 2.1 chess-best-move, image regenerated from the task's generator" />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ---------- full-screen terminal ---------- */
.fa-chess-term {
  position: absolute;
  inset: 0;
  background: var(--term-ground);
  color: var(--term-ink);
  font-family: var(--font-code);
  cursor: pointer;
  overflow: hidden;
}
.fa-chess-term canvas {
  display: block;
  width: 100%;
  height: 100%;
}
.fa-chess-status,
.fa-chess-term-legend {
  position: absolute;
  bottom: 14px;
  width: 196px;
  margin: 0;
  padding: 10px 12px;
  background: var(--term-ground);
  border: 2px solid var(--term-raised);
  border-radius: var(--radius-sm);
  font: 400 12.5px/1.45 var(--font-code);
  color: var(--term-dim);
}
.fa-chess-status { right: 14px; }
.fa-chess-status p {
  margin: 0;
  font: inherit;
  color: inherit;
}
.fa-chess-term-legend b,
.fa-chess-status .fa-chess-model {
  font-weight: 500;
  font-size: 15px;
  color: var(--term-ink);
  white-space: nowrap;
}
.fa-chess-raster {
  position: absolute;
  inset: 0;
  pointer-events: none;
  /* hard-edged scan lines, not a blend: 2px clear, 1px dark */
  background: repeating-linear-gradient(to bottom, transparent 0 2px, var(--term-raster) 2px 3px);
}
.fa-chess-prompt {
  padding-bottom: 6px;
  margin-bottom: 6px !important;
  border-bottom: 2px solid var(--term-raised);
  color: var(--term-ink) !important;
}
.fa-chess-prompt i {
  display: inline-block;
  width: 0.6em;
  height: 1.05em;
  margin-left: 4px;
  vertical-align: text-bottom;
  background: var(--term-ink);
  animation: fa-chess-blink 1.1s steps(1) infinite;
}
@keyframes fa-chess-blink {
  50% { opacity: 0; }
}
.fa-chess-note {
  padding-top: 6px;
  margin-top: 6px !important;
  border-top: 2px solid var(--term-raised);
}
.fa-chess-term-legend {
  left: 14px;
  list-style: none;
  display: grid;
  gap: 5px;
  line-height: 1.25;
}
.fa-chess-term-legend li {
  margin: 0;
  padding: 0;
}
.fa-chess-term-legend li::before { content: none; }
.fa-chess-term-legend b { display: block; }
.fa-chess-overlay {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 0 48px;
  text-align: center;
  pointer-events: none;
  font-family: var(--font-voice);
  font-variation-settings: var(--voice-settings);
  font-weight: 900;
  font-size: 64px;
  line-height: 1;
  letter-spacing: var(--track-voice);
  color: var(--term-ink);
}

/* ---------- board + copy panel ---------- */
.fa-chess-intro {
  display: grid;
  grid-template-columns: 460px 1fr;
  gap: 36px;
  height: 496px; /* 552px slide less the slide padding; the stage is label + square canvas */
  align-items: start;
}
.fa-chess-stage {
  margin: 0;
  display: flex;
  flex-direction: column;
  border: var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: var(--term-ground);
  cursor: pointer;
}
.fa-chess-stage figcaption {
  display: flex;
  justify-content: space-between;
  padding: 9px 14px;
  font: 500 14px/1 var(--font-code);
  color: var(--term-dim);
  border-bottom: 2px solid var(--term-raised);
}
.fa-chess-stage canvas {
  display: block;
  width: 456px;
  height: 456px;
}
.fa-chess-copy {
  display: flex;
  flex-direction: column;
  min-width: 0;
  height: 100%;
  padding-top: 6px;
}
.fa-chess-copy h1 {
  font-size: 44px;
  margin-bottom: 14px;
  min-height: 2.04em;
}
.fa-chess-body {
  margin: 0 0 14px;
  font-size: 19px;
  line-height: 1.4;
  color: var(--text-body);
}
.fa-chess-copy blockquote {
  margin: 0 0 14px;
  padding: 0;
  border: 0;
  background: none;
  font: italic 400 17px/1.4 var(--font-read);
  color: var(--text-muted);
}
.fa-chess-source {
  display: grid;
  gap: 6px;
  margin: 0;
}
.fa-chess-source div {
  display: grid;
  grid-template-columns: 72px 1fr;
  align-items: baseline;
}
.fa-chess-source dt {
  font: 500 13px/1.4 var(--font-read);
  color: var(--text-muted);
}
.fa-chess-source dd {
  margin: 0;
  font: 700 22px/1.2 var(--font-read);
  color: var(--petrol);
}
.fa-chess-source dd span {
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}
.fa-chess-source dd.is-second {
  font-size: 18px;
  font-weight: 600;
}
.fa-chess-legend b { width: 190px; }
.fa-chess-legend {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 3px;
  font-size: 15px;
  line-height: 1.3;
}
.fa-chess-legend b {
  display: inline-block;
  width: 132px;
}
.fa-chess-legend li {
  margin: 0;
  padding: 0;
}
.fa-chess-legend li::before {
  content: none;
}
.fa-chess-receipt {
  margin-top: auto;
}
</style>
