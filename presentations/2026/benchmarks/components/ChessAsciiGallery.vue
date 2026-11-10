<script setup lang="ts">
// The task's board image beside the same square (d8, the black queen) as six
// different agents printed it. Every character is verbatim tool output from
// data/chess-intro.json (analysis/chess-intro/extract.py, masked).
// Each grid is set on square cells, so it keeps the proportions of the
// pixels it was sampled from; a terminal would show it about twice as tall.
import raw from "../data/chess-intro.json";

type Block = { cols: number; rows: number; region: number[]; lines: string[] };
type Source = { harness: string; version: string; model: string };
const data = raw as unknown as { image: string; sources: Record<string, Source>; squares: Record<string, Record<string, Block>> };

const SQUARE = "d8";
const TILE = 128; // px available for the art inside a tile
const ids = ["ds4f", "grok46", "claude47", "gpt55", "glm51", "gpt55-agav"];
const tiles = ids.map((id) => {
  const b = data.squares[SQUARE][id];
  const s = data.sources[id];
  // a monospace cell is 0.6em wide: size the type so the grid fills the tile
  const font = TILE / (b.cols * 0.6);
  return {
    id,
    model: s.model.replace(/\s*\((max|high)\)$/, ""),
    harness: s.harness,
    text: b.lines.join("\n"),
    font: `${font.toFixed(2)}px`,
    // row height follows the pixels each character stood for: square for most
    // agents, twice as tall where one sampled every pixel across and every second row down
    leading: `${(font * 0.6 * ((b.region[3] - b.region[1]) / b.rows / ((b.region[2] - b.region[0]) / b.cols))).toFixed(2)}px`,
  };
});
const image = `${import.meta.env.BASE_URL}${data.image}`;
</script>

<template>
  <div class="fa-gallery">
    <figure class="fa-gallery-board">
      <div>
        <img :src="image" alt="The chess board image the task supplies" />
        <span aria-hidden="true" />
      </div>
      <figcaption>The task's image. The marked square is d8.</figcaption>
    </figure>
    <div class="fa-gallery-tiles">
      <figure v-for="t in tiles" :key="t.id">
        <pre :style="{ fontSize: t.font, lineHeight: t.leading }">{{ t.text }}</pre>
        <figcaption><b>{{ t.model }}</b>{{ t.harness }}</figcaption>
      </figure>
    </div>
  </div>
</template>

<style scoped>
.fa-gallery { display: grid; grid-template-columns: 290px 1fr; gap: 28px; align-items: start; }
.fa-gallery figure { margin: 0; }
.fa-gallery-board div { position: relative; border: var(--border); border-radius: var(--radius-lg); overflow: hidden; }
.fa-gallery-board img { display: block; width: 100%; }
/* d8 is the fourth file, top rank: one eighth of the board each way */
.fa-gallery-board span { position: absolute; left: 37.5%; top: 0; width: 12.5%; height: 12.5%; border: 3px solid var(--orange); }
.fa-gallery-board figcaption { margin-top: 8px; font: 500 14px/1.3 var(--font-read); color: var(--text-muted); }
.fa-gallery-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px 14px; }
.fa-gallery-tiles figure { border: var(--border); border-radius: var(--radius-lg); overflow: hidden; background: var(--petrol); }
.fa-gallery-tiles pre {
  height: 134px;
  margin: 0;
  padding: 3px 0 0;
  display: grid;
  place-content: center;
  border: 0 !important;
  border-radius: 0;
  background: transparent !important;
  color: var(--ivory);
  font-family: var(--font-code);
  font-weight: 500;
  letter-spacing: 0;
  white-space: pre;
  overflow: hidden;
}
.fa-gallery-tiles figcaption {
  padding: 4px 10px 5px;
  border-top: 2px solid var(--petrol-2);
  font: 500 12px/1.2 var(--font-read);
  color: rgba(255, 247, 232, 0.72);
}
.fa-gallery-tiles figcaption b { display: block; font-weight: 800; font-size: 14px; color: var(--ivory); }
</style>
