# Native Slidev port

`components/ToolQualityVersionChart.vue` imports only these local aggregates. The five source artifacts (`data.json`, `daily.csv`, `changes.json`, `provenance.json`, `README.md`) are copied byte-for-byte. Source README build/open instructions and provenance output hashes describe the original standalone chart, not this Vue component. No refresh, remote scan, classification, or reanalysis was performed. **Private aggregates: review before external sharing.**

## Integration

Give the component a wrapper with explicit width and height. Its root fills that wrapper; its native SVG retains the source 1600×900 viewBox (1110×540 plot), aspect ratio, colors, typography, and layer order. Use `capture` for the full uncluttered chart. Without it, compact controls and scrollable methodology share the wrapper height. No shared styles, slide content, packages, or composables are required to change.

Props: `capture?: boolean`, `title?: string`, `subtitle?: string`, `footer?: string`. Named `title`, `subtitle`, `footer` slots override their visible text (SVG text content). Prefer props for an accessible matching title. Defaults preserve the source title/subtitle; footer is empty. Long custom copy must fit the allotted SVG space. Source methodological caveats, exact counts, all observed events, all four verified changes, metadata and downloadable source files remain in the non-capture details; the SVG accessible description retains the private-sharing and non-causal warnings even in capture mode. Parent narrative should retain those caveats when presenting captured frames.

## Exposed API

- `durationMs = 33700`.
- `seek(position: number)` and `renderAt(elapsedMs: number)` both accept **source-clock milliseconds**, not a day index or normalized fraction. Finite inputs clamp to `[0, 33700]`; non-finite inputs throw. Both stop playback and deterministically set all geometry and effects. Fractional milliseconds are supported. Await Vue `nextTick()` before reading DOM or capturing.
- Clock: 0–1200 axes introduction; 1200–31200 continuous affine August 25 00:00 to September 14 00:00 UTC; 31200–33700 final hold. No event dwells. For a UTC timestamp `t`, use `1200 + (t - Date.parse('2026-08-25T00:00:00Z')) / 86400000 * 1500`.
- `play()` explicitly starts/resumes (restarts at zero if complete); `pause()` freezes all effects; `reset()` goes to source initial frame, paused. No autoplay. Optional looping begins only on explicit play.
- `state`: read-only computed, frozen snapshot with `position`, `elapsed`, `durationMs`, `playing`, `phase`, `state`, `progress`, `cursorTime`, `viewportStart`, `viewportEnd`, `rolling`, `dayIndex`, `day`, `clientVersion`, `serverVersion`, `activeChangeVersion`, and frozen `currentEvents`. Public component refs auto-unwrap the computed value. Methods return the snapshot.

Default is **latest paused**, rather than the original HTML's autoplay. Reduced motion also initializes at the end and disables guide growth/contrast and note fade during explicit playback/scrubbing. Browser print and Slidev print render the last frame regardless of the requested clock. Navigation departure, hidden document, print entry, and unmount pause/cancel playback. No keyboard listeners. Capture hides the interface, not the programmatic API.

## Semantic selectors

- Root: `[data-tool-quality-version]`, `data-state`, `data-phase`, `data-elapsed`, `data-day`, `data-rolling`, `data-cursor-time`, `data-viewport-start`, `data-viewport-end`.
- `[data-ribbon="claude|server"] .tqv-tape-segment`: `data-version`, `data-start`, `data-end`, `data-visible-start`, `data-visible-end`, `data-initial`, `data-visible`. Hidden future segments remain in DOM; filter on `data-visible="true"`. Exact start/end timestamps are milliseconds since epoch.
- `.tqv-version-marker`: `data-kind`, `data-version`, `data-time`, `data-visible`, `data-landed`; child `.tqv-version-guide` has actual SVG coordinates/opacity. Initial .12 has no marker. Old segments crossing the left edge do not invent markers.
- `[data-client-version]`, `[data-server-version]`, `[data-clock]`, `[data-rate="claude|others"]`, `[data-trace="claude|others"]`, `[data-samples="claude|others"]`, `[data-cursor]`.
- `[data-change-note]`: `data-version`, `data-visible`, logical-time `opacity`; `[data-change-headline]`, `[data-change-detail]`.
- `[data-controls]`, `[data-play]`, `[data-scrubber]`, `[data-method]`.

## Validation and intentional differences

Vue SFC script/template compilation and `npm run build` passed. An isolated native Vue/Vite browser harness with mocked Slidev context verified final rates (1.28%, 5.24%), final September 7–14 exclusive viewport, each observed change (.13/.15/.18/.19), unchanged .14 retaining the .13 note, halfway guide growth, exact six-hour .14 width (39.642857 SVG units), repeated fractional `renderAt` producing identical SVG, slide-departure pause, and print/reduced-motion final frames. Early and latest 1600×900 renders reviewed for clipping, density, contrast, and alignment. Temporary harness/screenshots removed. Parent should still test actual Slidev scaling, navigation, and multiple instances in the intended wrapper.

The standalone page's body layout/global shortcuts are not carried over. Controls are compact and methodology scrolls inside the wrapper; capture preserves the original chart geometry. Text fitting uses measured unscaled SVG Arial text after mount, retaining full versions down to 10px and hiding labels below that threshold without widening intervals. Use mounted component + `nextTick()` for captures. No animation CSS or wall-clock effects are used.
