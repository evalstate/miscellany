# Tessl MCP deck handover

This document summarizes the current working state of the Slidev deck and the
supporting data/QA workflow.

## Project

Workspace:

```text
/home/shaun/source/miscellany/presentations/2026/tessl-mcp
```

Primary files:

- `slides.md` — deck content
- `style.css` — global visual system and slide-level sizing
- `components/*.vue` — custom diagrams/charts
- `data-viz/` — deck-local data transforms and small chart artifacts
- `scripts/` — visual review, deterministic geometry QA, and helper scripts
- `AGENTS.md` — project instructions for agents

## Recent deck additions

### Remote MCP load balancer slide

Component:

```text
components/RemoteMcpLoadBalancer.vue
```

Slide title:

```text
Remote MCP through a load balancer
```

Purpose:

- Shows MCP Client → Load Balancer → three remote MCP Servers.
- Clicking the MCP Client triggers an animated light pulse.
- Each click cycles the routed target server.

Style notes:

- Uses the deck’s dark/amber/blue design tokens.
- Animation is implemented with Vue state, SVG paths, CSS keyframes, and SVG
  `animateMotion`.

### Transport Evolution slide

Component:

```text
components/McpSpecTransportTimeline.vue
```

Slide title:

```text
Transport Evolution
```

Purpose:

- Shows MCP Specification transport evolution as an oversized Gantt/timeline.
- Spec versions are equally spaced, not wall-clock accurate.
- Dates are month-level for visual compactness:
  - `2024-11`
  - `2025-03`
  - `2025-06`
  - `2025-11`
  - `2026-07`
  - `...`
- Lanes:
  - `TRANSPORTS` — `STDIO`
  - `REMOTE TRANSPORTS` — `SSE`, `Streamable HTTP`, `Stateless HTTP`
  - `AUTH` — `OAuth AS`, `OAuth Resource Server`

Interaction:

- Hovering over a spec column lights up the corresponding column across the
  timeline.
- Hovering a date lozenge also highlights its column.

Important implementation note:

- Header/date lozenges and Gantt lanes share a common six-column grid.
- Bar heights are normalized for visual rhythm.

### MCP remote traffic data-viz slides

Component:

```text
components/McpRemoteTrafficChart.vue
```

Slides:

1. `Claude Code`
2. `Codex`
3. `Codex` with the same date range as Claude Code

Purpose:

- Blue filled background area: opaque usage index.
- Bold amber line: share of traffic using `mcp-remote`.
- Top-right badge: latest `mcp-remote` share.

Current layout approach:

- Chart slides use an absolute safe area:

  ```css
  .traffic-chart-slide {
    position: absolute !important;
    inset: 28px 42px;
  }
  ```

- Chart component uses `box-sizing: border-box`.
- X-axis labels are outside the plot but within reserved SVG bottom space.
- The deterministic geometry audit currently reports zero findings for these
  slides.

## Data-viz workflow

Authoritative source data:

```text
/home/shaun/source/hf-mcp-stats
```

Treat that repository as read-only unless explicitly instructed otherwise.
Large raw datasets should stay there. This deck’s `data-viz/` directory should
contain only repeatable transforms and small slide-ready artifacts.

Deck-local data-viz files:

- `data-viz/generate_mcp_remote_weekly.py`
- `data-viz/emit_chart_data.py`
- `data-viz/SCRIPTS.md`
- `data-viz/mcp_remote_share_weekly.json`
- `data-viz/mcp_remote_share_weekly_chart.csv`
- `data-viz/mcp_remote_share_weekly_codex_claude_code.csv`
- `data-viz/mcp_remote_share_weekly_codex_claude_code_opencode.csv`

### `data-viz/SCRIPTS.md`

This is the catalog for data scripts and generated artifacts.

Maintenance rule:

> When adding, renaming, deleting, or materially changing a script or generated
> dataset in `data-viz/`, update `data-viz/SCRIPTS.md` in the same change.

### Generic chart data emitter

Script:

```text
data-viz/emit_chart_data.py
```

Purpose:

- Converts CSV into deck-friendly JSON using schema `deck.chart-data.v1`.
- Keeps CSV parsing/filtering out of Vue components.
- Supports repeated `--where`, `--series`, and `--include` flags.

Example:

```bash
python3 data-viz/emit_chart_data.py \
  --input data-viz/mcp_remote_share_weekly_codex_claude_code.csv \
  --output data-viz/codex_mcp_remote_chart.json \
  --where client_family=Codex \
  --x week_start \
  --x-bucket week \
  --series field=usage_index_0_100,kind=area,label="Opaque usage index",axis=usage \
  --series field=mcp_remote_share_pct,kind=line,label="mcp-remote share",axis=share \
  --include week_end \
  --include total_requests \
  --include mcp_remote_requests \
  --title Codex \
  --pretty
```

## Subagents

Project-local agent cards are now intended to be tracked in the repo:

```text
.fast-agent/agent-cards/
```

`.gitignore` is configured so:

- `.fast-agent/agent-cards/**` is trackable;
- `.fast-agent` runtime state, sessions, config, etc. remain ignored.

### Data-viz subagent

Card:

```text
.fast-agent/agent-cards/data-viz.md
```

Use for:

- querying `/home/shaun/source/hf-mcp-stats`;
- answering data questions;
- creating/maintaining Python transforms under `data-viz/`;
- keeping `data-viz/SCRIPTS.md` current.

Validated successfully.

### Visual-review subagent

Card:

```text
.fast-agent/agent-cards/visual-review.md
```

Use for qualitative visual QA after deterministic geometry checks pass.

Focus:

- malformed charts/SVGs/components;
- overlap;
- poor contrast;
- awkward hierarchy/density;
- ambiguous diagrams;
- visual rhythm issues.

Validated successfully.

## QA workflow

### Build

```bash
npm run build
```

Known build warnings:

- Rolldown/VueUse `INVALID_ANNOTATION` warnings.
- These are upstream warnings and not caused by the deck changes.

### Deterministic geometry audit

Script:

```text
scripts/check_slide_geometry.py
```

NPM alias:

```bash
npm run visual:geometry -- --range 13-15
```

Fail-on-findings mode:

```bash
npm run visual:geometry -- --range 13-15 --fail-on findings
```

What it catches:

- viewport overflow;
- clipping by overflow-hidden/clip/auto/scroll ancestors;
- chart/SVG text outside visible bounds;
- container scroll overflow;
- page-level scroll overflow.

Implementation:

- Starts Slidev dev server.
- Starts headless Chrome.
- Uses Chrome DevTools Protocol via Python `websockets`.
- Evaluates DOM geometry with `getBoundingClientRect()`, `scrollWidth`,
  `clientWidth`, `scrollHeight`, and `clientHeight`.

Dependency:

```bash
python3 -m pip install websockets
```

Current status:

- `npm run visual:geometry -- --range 13-15 --fail-on findings` passes.
- `python3 scripts/check_slide_geometry.py --range 11-15` reports zero findings.

### Screenshot rendering

```bash
python3 scripts/visual_review.py --range 13-15 --clean
```

Screenshots are written to:

```text
reports/screenshots/
```

The helper tries Slidev export first. If Playwright is not installed, it falls
back to live Chrome screenshots.

### VLM visual review

```bash
python3 scripts/visual_review.py --range 13-15 --review
```

Preferred order:

1. Run deterministic geometry audit.
2. Fix all deterministic findings.
3. Render screenshots.
4. Use VLM/visual-review for qualitative issues only.

Rationale:

- VLM/manual inspection can miss low-salience clipping, especially tiny axis
  labels near the slide edge.
- Geometry audit catches measurable clipping/overflow reliably.

## Design guidance captured during work

### Voiceover vs written text

Not every caveat belongs on the slide.

Before adding labels, captions, or explanatory prose, ask whether the audience
needs it to parse the visual or whether the presenter will say it aloud.

Prefer:

- bold structural labels;
- visual hierarchy;
- clean diagrams;
- voiceover for caveats and detail.

### Visual rhythm

Repeated semantic elements should share a clear sizing/alignment system:

- bars;
- cards;
- labels;
- icons;
- timeline markers;
- diagram nodes.

Differences in size, spacing, stroke weight, or radius should communicate
hierarchy/emphasis, not accidental variation.

This came up specifically with the Gantt bars on the Transport Evolution slide.

## Current useful commands

```bash
npm run build
npm run visual:geometry -- --range 13-15 --fail-on findings
python3 scripts/visual_review.py --range 13-15 --clean
python3 scripts/visual_review.py --range 13-15 --review
```

For local editing:

```bash
npm run dev
```

For static preview:

```bash
npm run preview
```

## Notes / possible next improvements

- Consider adapting `McpRemoteTrafficChart.vue` to consume `deck.chart-data.v1`
  JSON from `emit_chart_data.py` rather than the current bespoke weekly JSON
  shape.
- Consider adding a CI-style command that runs:
  1. `npm run build`
  2. `npm run visual:geometry -- --range <important slides> --fail-on findings`
- The current data-viz slides are probably still candidates for narrative
  simplification. The top-right badge is strong; inline latest labels may be
  optional depending on taste.
- If screenshots/VLM miss something suspicious, add a deterministic geometry
  rule before relying on prompt changes alone.
