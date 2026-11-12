<script setup lang="ts">
import { computed } from "vue";
import { useIsSlideActive } from "@slidev/client";

type Site = {
  id: string;
  title: string;
  model: string;
  date: string;
  domain: string;
  n: string;
  harness: string;
  press: string;
  pressSource: string;
};
type Box = [number, number, number];
const props = defineProps<{ stage: number; sites: Site[]; heading: string }>();
const active = useIsSlideActive();
const focus = computed(() => props.stage >= 3 && props.stage <= 5 ? props.stage - 3 : -1);
const count = computed(() => Math.min(props.stage + 1, 3));
const kinds = ["page", "panel", "press"] as const;

// All values refer to the deck's 980 × 552 canvas. Widths preserve the
// original screenshots' proportions. Papers stay level; data never tilts.
const ratios = [
  { page: 1440 / 720, panel: 1320 / 650, press: 1642 / 269 },
  { page: 1440 / 620, panel: 1536 / 896, press: 1164 / 352 },
  { page: 1440 / 620, panel: 1540 / 868, press: 1164 / 264 },
];
const desk: Box[][] = [
  [[10, 72, 330], [20, 234, 310], [10, 410, 330]],
  [[318, 92, 350], [338, 220, 314], [350, 412, 230]],
  [[660, 72, 310], [660, 216, 310], [678, 404, 280]],
];
const solo: Box[] = [[20, 72, 610], [420, 224, 530], [20, 408, 340]];
const two: Box[][] = [
  [[12, 86, 350], [18, 256, 330], [14, 424, 330]],
  [[382, 70, 580], [470, 196, 490], [460, 414, 260]],
];
const zoom: Box[] = [[95, 78, 790], [160, 78, 660], [150, 78, 680]];

function paperStyle(i: number, kind: typeof kinds[number]) {
  const k = kinds.indexOf(kind);
  let box = desk[i][k];
  if (props.stage === 0 && i === 0) box = solo[k];
  if (props.stage === 1 && i < 2) box = two[i][k];
  if (focus.value === i && kind === "panel") box = zoom[i];
  const [x, y, w] = box;
  const h = w / ratios[i][kind];
  const visible = i < count.value && (kind !== "press" || i === 0 || props.stage >= 2);
  return {
    left: `${x / 980 * 100}%`,
    top: `${y / 552 * 100}%`,
    width: `${w / 980 * 100}%`,
    height: `${h / 552 * 100}%`,
    opacity: visible ? 1 : 0,
    visibility: visible ? "visible" as const : "hidden" as const,
    zIndex: focus.value === i && kind === "panel" ? 6 : 1 + k,
  };
}
function src(site: Site, kind: typeof kinds[number]) {
  return `${import.meta.env.BASE_URL}screens/launch-research/${kind === "press" ? site.press : `${site.id}-${kind}.png`}`;
}
const visibleSites = computed(() => props.sites.slice(0, count.value));
const sources = computed(() => visibleSites.value.map(s =>
  `${s.model}, ${s.date}`
).join(" · "));
const pressSources = computed(() => (props.stage === 1 ? props.sites.slice(0, 1) : visibleSites.value).map(s => s.pressSource.replace(" 2026", "").replace(",", "")).join(" · "));
const selected = computed(() => props.sites[focus.value]);
</script>

<template>
  <div class="launch-desk" :class="{ 'is-active': active, 'is-focused': focus >= 0 }">
    <h1>{{ heading }}</h1>
    <div class="launch-desk-papers" aria-label="Launch-page, benchmark and press screenshots accumulating on one canvas">
      <template v-for="(site, i) in sites" :key="site.id">
        <figure
          v-for="kind in kinds"
          :key="kind"
          class="launch-desk-paper"
          :class="[`launch-desk-paper--${kind}`, { 'is-selected': focus === i && kind === 'panel' }]"
          :style="paperStyle(i, kind)"
          :aria-hidden="i >= count || (kind === 'press' && stage === 1 && i !== 0)"
        >
          <img :src="src(site, kind)" :alt="`${site.title} · ${site.model}: ${kind === 'page' ? 'official launch page' : kind === 'panel' ? 'original benchmark panel' : 'press headline'}`" />
        </figure>
      </template>
    </div>
    <div class="launch-desk-scrim" :class="{ on: focus >= 0 }" aria-hidden="true"></div>

    <div class="launch-desk-receipt" aria-live="polite">
      <Receipt
        v-if="selected"
        :n="selected.n"
        :date="selected.date"
        :model="`${selected.model} and comparators shown`"
        :harness="selected.harness"
        :source="`${selected.domain} · original panel captured 11 Nov 2026`"
      >Developer-reported; settings differ. This is how results are presented, not a harmonised ranking.</Receipt>
      <Receipt v-else :source="`${sources} · captured 11 Nov 2026`">
        Press: {{ pressSources }}. n / harness versions not shown in panels; developer-reported; settings differ.
      </Receipt>
    </div>
  </div>
</template>
