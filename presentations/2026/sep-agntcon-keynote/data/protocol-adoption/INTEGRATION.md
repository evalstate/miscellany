# Native Slidev integration

`components/ProtocolAdoptionChart.vue` ports the source 1600×900 SVG, using native
Vue SVG elements and an instance-local RAF clock. Give its parent an explicit
width and height; the root fills both. The SVG keeps its aspect ratio. Controls
consume space below the stage; `capture` removes them so the stage fills the
wrapper. No deck/shared styles, narrative, or shared composables were changed.

## Parent ref API

```ts
chart.configure({ client: 'chat-ui-mcp', viewport: 'seven', speed: 1 })
chart.renderAt(15_000)
await nextTick() // Vue DOM flush before taking a screenshot
// state.position === 24.5; state.index === 24
```

- `seek(position: number)`: pause, clamp fractional day index to 0–49.
- `renderAt(elapsedMs: number)`: pause and seek to
  `clamp(elapsedMs * speed * 49 / 30000, 0, 49)`. Absolute elapsed source time,
  not accumulated playback time. No RAF is started. This always renders the
  exact fraction, independent of reduced-motion preferences. Use speed 1 for
  a 30-second capture. Negative values clamp to zero; nonfinite values throw.
- `play()`, `pause()`, `reset()`: manual playback; reset pauses at index zero.
  Play from the end replays. Pause preserves the exact fraction. Play is blocked
  on an inactive slide, hidden document, or print render. Capture mode hides UI
  and never autostarts, but explicit API `play()` is permitted.
- `configure({ client?, viewport?, speed? })`: exact source client identity,
  `full`/`seven`, finite positive speed. Validates atomically. It does not reset
  position or implicitly pause. Source UI speeds are 0.5, 1, 2, 4; custom positive
  speeds are also supported and appear in the selector.
- `durationMs`: 30000 (the source duration at 1×).
- `state`: read-only computed frozen snapshot, automatically unwrapped on the
  exposed Vue ref: index, position, viewportStart/End, duration, durationMs,
  playing, client, viewport, speed, reducedMotion, date, partial, dailyShare,
  rollingShare, and frozen counts. Date/rates/counts use floor(position).

Initial state is latest, paused, Overall, full history, 4×. No global window API
or keyboard listeners. Controls stop event bubbling to deck navigation. Native
range keyboard handling remains available. Instances have unique SVG clip and
accessibility IDs. Hidden pages and slide departure pause without auto-resume;
unmount cancels RAF and removes listeners. Reduced-motion explicit playback
steps daily using the same elapsed clock. Slidev and browser print show the last
frame of the configured client/viewport with controls hidden.

## Props and narrative

`capture?: boolean`, `title?: string`, `subtitle?: string`, `footer?: string`,
`footnote?: string`. The keynote slide supplies its concise title in markdown. Subtitle, viewport label,
snapshot footer, and stage caveat text are omitted; provenance remains accessible. An empty footer uses the source default. Keep
custom text short enough for the fixed SVG coordinates; put longer narrative
and interpretation in parent markdown/speaker notes, not this component.

Speaker-note recommendations: private aggregates require review before public
sharing; logged tool calls are neither users nor protocol messages. This is a
pinned **2026-09-15** partial snapshot, not live. Rolling7 is a count-weighted
ratio over exactly seven completed UTC dates, not a mean of daily percentages.
Changing plot viewport never changes that denominator. Purple guide segments
are illustrative reveals, not intraday estimates. Unknown/unreviewed protocols
are excluded from the rate denominator; null is unavailable, never zero. Overall
includes retained clients beyond the six focus identities.

## Browser selectors

- Root `.protocol-adoption-chart`: `data-state`, `data-position`, `data-index`,
  `data-client`, `data-viewport`, `data-partial`, `data-capture`,
  `data-reduced-motion`.
- `[data-role="stage"]`, `axes`, `lines`, `ticks`, `clock`, `client-label`,
  `viewport-label`, `daily-heading`, `daily`, `daily-date`, `partial-label`,
  `rolling`, `rolling-start`, `rolling-end`, `rolling-note`, `stage-snapshot`.
- `[data-series="daily_share"]` / `[data-series="rolling_share"]` contain
  per-anchor `g[data-index][data-partial]`, `[data-role="connection"]` and
  `circle[data-role="point"][data-index][data-field]`. Only reached points
  exist in the DOM. Use `g[data-partial="true"] circle` for the amber hollow
  point (the root also has a selected-day `data-partial` attribute).
- `[data-control="client"|"viewport"|"speed"|"scrubber"]`,
  `[data-action="play"|"reset"|"provenance"]`, `[data-role="controls"]`,
  `[data-role="status"]`. Controls/details are absent in capture/print.

## Source preservation and compromises

`data.json`, `daily.csv`, `provenance.json`, `README.md` are byte-for-byte copies
of `/home/evalstate/source/data-analysis/charts/protocol-adoption-dashboard/`.
The original README describes the standalone HTML, not this Vue integration.
No remote scan, source aggregation, denominator recalculation, or data updates
were performed. Full source provenance and an accessible selected-client raw
count table are available in the Data & provenance overlay, alongside download
links to all four copied files.

Standalone fullscreen and SVG-download buttons are intentionally omitted:
Slidev owns fullscreen and parent capture owns recording. The original
page-wide hotkeys are deliberately not ported. Controls are compact and the
source's below-page details become a scrollable overlay inside the wrapper.
No audience-facing private-data warning was added to the source SVG; the source
warning is retained in the README and the data overlay. Review before publishing.

Validation: `npm run build` and direct Vue SFC compilation passed. Isolated
Chromium harness (mocked Slidev context) checked all seven client identities,
latest paused, fractional floor/viewport, no future points, initial null rolling,
hollow partial point, deterministic re-render, speed mapping, slide departure,
and final print state. Capture at 1600×900 and controls at 980×552 were visually
reviewed; default SVG text was within canvas bounds. Parent should still verify
actual Slidev routing, deck CSS interaction, print/export and wrapper sizing.
