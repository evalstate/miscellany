# Tessl MCP deck handover

Current workspace:

```text
/home/shaun/source/miscellany/presentations/2026/aidevcon-london
```

This is a Slidev conference deck. The source of truth for slide flow and
narrative is `slides.md`; global design primitives live in `style.css`; reusable
charts/diagrams live under `components/`.

## Current working state

Recent checkpoint commits:

```text
220d70d Checkpoint Tessl MCP deck refactor
9a20db1 Apply Slidev-first chart composition
```

Since those commits, additional uncommitted work has been done on the HTTP
standardization problem slide and visual-review guidance:

- added `components/HttpRouteMap.vue`;
- replaced raw route-map SVG/HTML in `slides.md` with `<HttpRouteMap />`;
- added/adjusted HTTP request-panel styling in `style.css`;
- updated `.fast-agent/agent-cards/visual-review.md` to catch connector/arrow
  anchoring and directionality problems.

Do **not** assume `slides.md` has stayed stable across turns. The user may edit
it directly; reread it before planning slide-number-specific work.

## Authoring principles now in force

See `AGENTS.md`, especially **Slidev-first authoring**.

Practical rules:

- Slide narrative belongs in `slides.md`:
  - titles;
  - subtitles;
  - rhetorical framing;
  - explanatory audience-facing copy;
  - ordering and high-level composition.
- Components own data-driven rendering, charts, diagrams, interactions, and
  chart-internal labels.
- Chart/diagram internals may own axis titles, legend labels, row labels,
  stat labels, and data-derived badges.
- If those chart-internal labels need to vary by narrative/dataset, prefer props
  over hard-coded text.
- Parent slide/wrapper controls available space; components fill that space.
- Avoid raw connector coordinates in `slides.md`. Connector-heavy diagrams
  justify Vue/SVG components because geometry is actual diagram logic.

Decision tree:

- Simple flowchart: consider Mermaid.
- Custom nodes/connectors: use a dedicated Vue/SVG component.
- Decorative line/divider: use CSS.
- Slide title/subtitle/copy: keep in `slides.md`.

## Important current slide patterns

### Shared chart slide wrapper

Several data slides use slide-owned headers plus data/rendering components:

```html
<div class="weekly-activity-slide chart-slide">
  <header class="chart-slide__header">
    <div>
      <h1>Weekly MCP activity</h1>
      <p>Initialization requests as bars · tool calls as line</p>
    </div>
  </header>
  <McpWeeklyActivityChart />
</div>
```

Shared classes in `style.css`:

- `.chart-slide`
- `.chart-slide__header`
- `.chart-slide__kicker`
- `.weekly-activity-slide`
- `.conversion-chart-slide`
- `.protocol-efficiency-slide`
- `.traffic-chart-slide`

The chart components keep data-coupled stats/labels but no longer own the slide
headline/subhead.

### HTTP Standardization: Problem

Slide location currently around `slides.md` line ~272. Current structure:

```md
# HTTP Standardization: Problem

<div class="http-standardization-problem">
  <HttpRouteMap />

  <section class="http-request-panel deck-panel">
    ...
  </section>
</div>
```

The route map was deliberately moved out of raw markdown because the previous
implementation mixed CSS-positioned nodes with hard-coded SVG line/polygon
coordinates. That was brittle and produced floating/misdirected arrows.

New component:

```text
components/HttpRouteMap.vue
```

Architecture:

- all nodes are defined in a single SVG coordinate system;
- edges reference node IDs and anchor sides;
- connector endpoints are computed from node geometry;
- SVG markers render arrowheads;
- future edits should adjust node positions, edge anchors, or offsets in the
  component data, not hand-tune SVG coordinates in `slides.md`.

Current `HttpRouteMap` still may benefit from visual polish, but it is now
maintainable. The latest rendered version shows correctly vertical region ↔
endpoint and endpoint ↔ client connectors. If improving further, prefer data
changes in the component:

- move `regionA`, `regionB`, `endpoint`, or `client` node coordinates;
- adjust `fromOffsetX` / `toOffsetX` for individual edges;
- tune SVG marker size/ref points.

Do not reintroduce raw `<line>` / `<polygon>` connector markup into `slides.md`.

## Components of note

### `components/McpWeeklyActivityChart.vue`

- Renders weekly initializations as bars and tool calls as a line.
- No slide title/subtitle; those live in `slides.md`.
- Keeps latest-week stat badge as data-coupled chart content.

### `components/McpRemoteNoFallbackChart.vue`

- Renders initializations and mcp-remote share excluding fallback checks.
- No slide title/subtitle; those live in `slides.md`.
- Keeps latest-share stat badge as data-coupled chart content.

### `components/SessionConversionChart.vue`

- Renders session conversion rate and 3-day converted-session average.
- Slide title/subtitle moved to `slides.md`.
- Keeps `overall` / `latest` stats as data-derived chart internals.

### `components/McpProtocolEfficiency.vue`

- Renders message-mix bars and tool-call callout.
- Slide kicker/title moved to `slides.md`.
- Date window remains chart/data context and is positioned into the shared header
  region.

### `components/McpRemoteTrafficChart.vue`

- Still has `title` / `subtitle` props and renders its own header.
- This is a remaining Slidev-first mismatch if those slides become actively
  edited. Consider refactoring later to use the shared `chart-slide` header
  pattern and leave only chart-internal labels/stats in the component.

### `components/McpSpecTransportTimeline.vue`, `ProtocolStack.vue`, `RemoteMcpLoadBalancer.vue`

- Custom diagram components are appropriate because they own interaction or
  diagram geometry.
- Continue parent-defined sizing via wrappers such as `.spec-timeline-diagram`,
  `.protocol-diagram`, and `.remote-mcp-diagram`.

## Data-viz workflow

Authoritative MCP stats source data is outside this deck:

```text
/home/shaun/source/hf-mcp-stats
```

Treat that repo as read-only unless explicitly told otherwise. Deck-local
`data-viz/` contains repeatable transforms and small slide-ready CSV/JSON
artifacts.

Important convention from `AGENTS.md`:

> When adding, renaming, deleting, or materially changing a script or generated
> dataset under `data-viz/`, update `data-viz/SCRIPTS.md` in the same change.

Recent data-viz additions committed in `220d70d` include generated mcp-remote
excluding-fallback datasets and scripts.

## Visual QA workflow

Preferred order:

1. Run deterministic geometry checks.
2. Fix geometry findings.
3. Render screenshots.
4. Use VLM/visual-review for qualitative issues.

Commands:

```bash
npm run build
python3 scripts/check_slide_geometry.py --range 23 --fail-on findings
python3 scripts/visual_review.py --range 23 --clean
```

For broader review:

```bash
python3 scripts/check_slide_geometry.py --range 1-24 --fail-on findings
python3 scripts/visual_review.py --range 1-24 --clean
```

Screenshots go to:

```text
reports/screenshots/
```

These are runtime artifacts; do not commit screenshots unless explicitly asked.

## Known build warnings

`npm run build` succeeds but emits upstream warnings:

```text
[INVALID_ANNOTATION] A comment "/* #__PURE__ */" in node_modules/@vueuse/core/dist/index.js ...
```

A dependency-refresh attempt did not clear this. Treat it as upstream
Rolldown/VueUse noise unless CI starts failing.

There is also a warning about `--localstorage-file` without a valid path during
Slidev export/build helpers; it is currently non-fatal.

## Current uncommitted / intentionally unstaged files

As of this handover, expect some non-deck or runtime files to remain dirty:

- sibling presentation changes under `../agentic-ai-aws/` — unrelated;
- `fastagent.jsonl` — runtime log;
- `reports/slide-geometry.json` — generated QA output;
- `reports/screenshots/*` — generated screenshots;
- `data-viz/__pycache__/` — generated Python cache;
- root-level notes such as `../../../acp.md`, `../../../core-transports.md`, etc.

The HTTP route-map work itself is currently uncommitted at the time of this
handover unless a later agent commits it. Relevant files to stage if committing:

```text
components/HttpRouteMap.vue
slides.md
style.css
.fast-agent/agent-cards/visual-review.md
handover.md
```

Before committing, rerun:

```bash
npm run build
python3 scripts/check_slide_geometry.py --range 23 --fail-on findings
```

## Suggested next actions

1. Visually review slide 23 one more time after any route-map tweaks.
2. If satisfied, commit the HTTP standardization problem slide work.
3. Consider refactoring `McpRemoteTrafficChart` away from owning title/subtitle
   if those slides need further authoring.
4. Consider turning repeated chart-slide wrapper markup into a formal
   `layouts/chart.vue` if the pattern grows.
5. Keep watching connector-heavy diagrams: use components with node/edge data,
   not raw markdown SVG coordinates.
