<script setup lang="ts">
import { computed, ref, useId } from 'vue';
import { useNav } from '@slidev/client';
import { useRoute } from 'vue-router';
import data from '../data/client-adoption-comparison/normalized.json';
import provenance from '../data/client-adoption-comparison/provenance.json';
import dailyUrl from '../data/client-adoption-comparison/daily.csv?url';
import summaryUrl from '../data/client-adoption-comparison/summary.csv?url';
import statsUrl from '../data/client-adoption-comparison/stats.json?url';
import provenanceUrl from '../data/client-adoption-comparison/provenance.json?url';
import readmeUrl from '../data/client-adoption-comparison/README.md?url';
import normalizedUrl from '../data/client-adoption-comparison/normalized.json?url';

withDefaults(defineProps<{ title?: string; capture?: boolean }>(), {
  title: '2026-07-28 Protocol Adoption (Tool Calls)',
  capture: false,
});
const { isPrintMode } = useNav();
const route = useRoute();
const printMode = computed(() => isPrintMode.value || /^\/print(?:\/|$)/.test(route.path));
const detailsOpen = ref(false);
const selectedId = ref(data.series[0].id);
const selected = computed(() => data.series.find(s => s.id === selectedId.value)!);
const uid = `client-adoption-${useId().replace(/[^a-zA-Z0-9_-]/g, '-')}`;
const purple = '#6430d8', slate = '#475569';
const pct = (v: number | null) => v === null ? '—' : `${(v * 100).toFixed(1)}%`;
const x = (i: number) => 54 + i / (data.dates.length - 1) * 420;
const y = (v: number) => 238 - v * 150;
const fields = ['trailing7_share', 'daily_share'] as const;
const ticks = [{ i: 0, label: 'Jul 28' }, { i: 24, label: 'Aug 21' }, { i: 48, label: 'Sep 14' }];
// Geometry only: every point is a supplied rate. Nulls break paths, including
// incomplete strict-seven windows; isolated observations remain visible as dots.
const panels = data.series.map((s, panelIndex) => ({
  ...s,
  displayLabel: ({ 'chat-ui-mcp': 'Hugging Face Chat UI', 'codex-mcp-client': 'Codex CLI' } as Record<string, string>)[s.id] ?? s.id,
  transform: `translate(${32 + panelIndex % 3 * 516}, ${150 + Math.floor(panelIndex / 3) * 342})`,
  maxActivity: Math.max(1, ...s.rows.map(r => r.all_calls)),
  lines: fields.map(field => {
    let connected = false;
    const path = s.rows.map((r, i) => {
      const v = r[field];
      if (v === null) { connected = false; return ''; }
      const command = connected ? 'L' : 'M';
      connected = true;
      return `${command}${x(i)},${y(v)}`;
    }).join(' ');
    return { field, path, color: field === 'daily_share' ? purple : slate,
      points: s.rows.flatMap((r, i) => r[field] === null ? [] : [{ i, date: r.date, value: r[field]! }]) };
  }),
}));
const downloads = [
  { name: 'daily.csv', url: dailyUrl }, { name: 'summary.csv', url: summaryUrl },
  { name: 'stats.json', url: statsUrl }, { name: 'provenance.json', url: provenanceUrl },
  { name: 'README.md', url: readmeUrl }, { name: 'normalized.json', url: normalizedUrl },
];
const columns = ['date', 'all_calls', 'valid_calls', 'modern_calls', 'daily_share',
  'trailing7_days', 'trailing7_valid_calls', 'trailing7_modern_calls', 'trailing7_share'] as const;
</script>

<template>
  <section class="client-adoption-comparison" data-chart="client-adoption-comparison"
    data-through="2026-09-14" data-min-valid-calls="100">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" role="img"
      :aria-labelledby="`${uid}-title ${uid}-description`" data-role="comparison-svg">
      <title :id="`${uid}-title`">{{ title }}</title>
      <desc :id="`${uid}-description`">Six clients in source order. Daily and strict trailing-seven shares of reviewed valid tool calls using protocol 2026-07-28, July 28 through September 14, 2026 UTC. Common 0–100% axes. Blank rates are unavailable, not zero. Miniature all-protocol activity bars use each client's own scale.</desc>
      <rect width="1600" height="900" fill="white" />
      <rect x="45" y="36" width="9" height="48" rx="2" fill="#6430d8" />
      <text x="80" y="77" fill="#172032" :style="{ fontSize: '54px', fontWeight: 800 }" data-role="chart-title">{{ title }}</text>
      <g :style="{ fontSize: '23px' }" fill="#475569" data-role="legend">
        <line x1="50" x2="94" y1="115" y2="115" :stroke="purple" stroke-width="3" />
        <circle cx="72" cy="115" r="3.5" :fill="purple" />
        <text x="105" y="123">Daily</text>
        <line x1="212" x2="256" y1="115" y2="115" :stroke="slate" stroke-width="5" />
        <text x="268" y="123">Rolling 7 days</text>
        <text x="1548" y="123" text-anchor="end">Last 7: Sep 8–14, 2026</text>
      </g>
      <g v-for="panel in panels" :key="panel.id" :transform="panel.transform"
        data-role="client-panel" :data-client="panel.id">
        <text x="54" y="30" fill="#172032" :style="{ fontSize: '28px', fontWeight: 700 }" data-role="client-name">{{ panel.displayLabel }}</text>
        <text x="54" y="64" :fill="slate" :style="{ fontSize: '24px', fontWeight: 700 }"
          data-role="last7" :data-value="panel.summary.last7_share">Last 7: {{ pct(panel.summary.last7_share) }}</text>
        <g v-for="tick in [0, 0.5, 1]" :key="tick" data-role="rate-tick">
          <line x1="54" x2="474" :y1="y(tick)" :y2="y(tick)" stroke="#e2e8f0" stroke-width="1.5" />
          <text x="46" :y="y(tick) + 7" text-anchor="end" fill="#64748b" :style="{ fontSize: '19px' }">{{ tick * 100 }}%</text>
        </g>
        <g v-for="line in panel.lines" :key="line.field" :data-series="line.field">
          <path :d="line.path" fill="none" :stroke="line.color" :stroke-width="line.field === 'daily_share' ? 2.5 : 4.5" stroke-linejoin="round" stroke-linecap="round" />
          <circle v-for="point in line.points" :key="point.date" :cx="x(point.i)" :cy="y(point.value)"
            :r="line.field === 'daily_share' ? 2.7 : 2.3" :fill="line.color"
            :data-date="point.date" :data-value="point.value" data-role="rate-point">
            <title>{{ panel.id }} · {{ point.date }} · {{ line.field }}: {{ pct(point.value) }}</title>
          </circle>
        </g>
        <g data-role="activity" data-scale="independent" :data-max="panel.maxActivity">
          <rect v-for="(row, i) in panel.rows" :key="row.date" :x="x(i) - 3.4"
            :y="292 - row.all_calls / panel.maxActivity * 34" width="6.8"
            :height="row.all_calls / panel.maxActivity * 34" :fill="purple" :style="{ opacity: 0.35 }"
            :data-date="row.date" :data-count="row.all_calls">
            <title>{{ row.date }} · {{ row.all_calls.toLocaleString('en-US') }} all-protocol calls</title>
          </rect>
        </g>
        <text v-for="tick in ticks" :key="tick.i" :x="x(tick.i)" y="318" text-anchor="middle"
          fill="#64748b" :style="{ fontSize: '20px' }">{{ tick.label }}</text>
      </g>
      <g data-role="activity-legend">
        <rect x="50" y="861" width="7" height="12" :fill="purple" :style="{ opacity: 0.35 }" />
        <rect x="60" y="853" width="7" height="20" :fill="purple" :style="{ opacity: 0.35 }" />
        <text x="82" y="873" fill="#64748b" :style="{ fontSize: '21px' }">Mini bars: daily activity · own scale per client</text>
      </g>
    </svg>
    <button v-if="!capture && !printMode" type="button" class="cac-control" data-action="provenance"
      :aria-expanded="detailsOpen" :aria-controls="`${uid}-details`" @click.stop="detailsOpen = !detailsOpen"
      @keydown.stop @pointerdown.stop>Data &amp; provenance</button>
    <aside v-if="detailsOpen && !capture && !printMode" :id="`${uid}-details`" class="cac-details"
      aria-label="Data and provenance" data-role="data-overlay" @keydown.stop @keydown.esc="detailsOpen = false" @pointerdown.stop @click.stop>
      <button type="button" @click="detailsOpen = false">Close data &amp; provenance</button>
      <p><strong>Private review output — review before public sharing.</strong> Self-reported identities, not users or proven product migrations. Coverage does not establish complete real-world traffic.</p>
      <p>Completed UTC dates only: July 28–September 14, 2026; last7 September 8–14. Supplied CSV shares require ≥100 valid calls. Rolling7 is count-weighted over exactly seven calendar days. Null means unavailable, not zero. No rates or denominators are recomputed. Unknown/unreviewed protocols are excluded from rates, retained in activity counts. Fixed five excluded; missing hashes retained.</p>
      <p>Mini bars show actual all-protocol daily calls, independently scaled per client: compare activity patterns, not popularity. chat-ui-mcp is distinct from chat-ui-intern; codex-mcp-client is distinct from openai-mcp (Codex). This is not the September 15 dashboard snapshot.</p>
      <nav aria-label="Download source data"><a v-for="file in downloads" :key="file.name" :href="file.url" :download="file.name">{{ file.name }}</a></nav>
      <label>Client <select v-model="selectedId" data-control="table-client"><option v-for="s in data.series" :key="s.id" :value="s.id">{{ s.id }}</option></select></label>
      <p>Supplied last7: {{ pct(selected.summary.last7_share) }} · {{ selected.summary.last7_modern_calls }} modern / {{ selected.summary.last7_valid_calls }} valid calls · {{ selected.summary.last7_days }} days.</p>
      <div class="cac-table"><table>
        <caption>{{ selected.id }} · source values; shares are fractions, — means unavailable</caption>
        <thead><tr><th v-for="column in columns" :key="column" scope="col">{{ column }}</th></tr></thead>
        <tbody><tr v-for="row in selected.rows" :key="row.date"><td v-for="column in columns" :key="column">{{ row[column] ?? '—' }}</td></tr></tbody>
      </table></div>
      <details><summary>Full original provenance</summary><pre>{{ JSON.stringify(provenance, null, 2) }}</pre></details>
    </aside>
  </section>
</template>

<style scoped>
.client-adoption-comparison { position: relative; width: 100%; height: 100%; min-width: 0; min-height: 0; overflow: hidden; background: white; color: #172032; font-family: Arial, Helvetica, sans-serif; }
.client-adoption-comparison svg { display: block; width: 100%; height: 100%; font-family: Arial, Helvetica, sans-serif; }
.client-adoption-comparison text { font-variant-numeric: tabular-nums; }
.cac-control { position: absolute; bottom: 1%; right: 2%; }
.client-adoption-comparison button, .client-adoption-comparison select { font: 600 11px Arial, sans-serif; color: #475569; background: white; border: 1px solid #cbd5e1; border-radius: 5px; padding: 5px 8px; cursor: pointer; }
.client-adoption-comparison :is(button, select, a, summary):focus-visible { outline: 3px solid #6430d8; outline-offset: 2px; }
.cac-details { position: absolute; inset: 8px; z-index: 2; overflow: auto; background: white; border: 2px solid #6430d8; border-radius: 6px; padding: 16px; font-size: 13px; line-height: 1.5; }
.cac-details p { margin: 10px 0; }
.cac-details nav { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }
.cac-details a { color: #6430d8; text-decoration: underline; }
.cac-table { overflow: auto; margin: 12px 0; }
.cac-table table { width: 100%; border-collapse: collapse; font-size: 11px; }
.cac-table :is(th, td) { padding: 6px; border-bottom: 1px solid #e2e8f0; text-align: right; }
.cac-table caption { text-align: left; }
.cac-details pre { font-size: 11px; white-space: pre-wrap; overflow-wrap: anywhere; }
@media print { .cac-control, .cac-details { display: none !important; } }
</style>
