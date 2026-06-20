# Agent notes for this Slidev deck

## Goal

Assist with the deck as both source editor and visual reviewer. Prefer rendered
evidence over source-only guesses when discussing layout anomalies.

## Project basics

- Source deck: `slides.md`
- Global styles: `style.css`
- Layouts: `layouts/*.vue`
- Components: `components/*.vue`
- Static assets: `public/`
- Chrome is available at `/usr/bin/google-chrome`.
- Project-local fast-agent tooling lives under `.fast-agent/`.

The user may edit `slides.md` concurrently. Always reload/read the current
`slides.md` before planning slide changes, citing slide numbers, or applying
edits that depend on slide order/content.

## Common commands

```bash
npm run dev
npm run build
npm run preview
npm run export
npm run visual:geometry -- --range 1-5
npm run visual:export -- --range 1-5 --clean
npm run visual:review -- --range 1-5
```

Build output in `dist/` is an ES-module SPA; serve it over HTTP. Use
`npm run build:single` only when a filesystem-openable single HTML file is
needed.

## Slidev-first authoring

Prefer Slidev markdown, layouts, frontmatter, and shared CSS primitives before
creating new Vue components or component-local style systems.

Use `slides.md` for slide narrative: titles, subtitles, labels, explanatory
copy, ordering, and high-level composition. Vue components should usually own
only data-driven rendering, reusable diagrams, interaction, or small reusable
presentational primitives.

Before adding a new component, ask:

- Can this be expressed clearly in `slides.md` with an existing layout/class?
- Is this behavior/data rendering reused or complex enough to justify Vue?
- Will future wording/layout edits require opening a Vue file unnecessarily?

Before adding component-local CSS, ask:

- Is this a shared deck pattern that belongs in `style.css`?
- Can existing tokens/classes such as card styles, labels, or diagram containers be reused?
- Is the parent slide controlling size while the component fills the available region?

Prefer parent-defined dimensions and shared wrappers over component props such
as `size="sm|md|lg"` unless the variant is semantic. Keep components narrow:
render the chart/diagram/widget; let Slidev compose the slide.

## Connector-heavy diagrams

For diagrams with arrows/connectors between nodes, avoid hand-authoring raw SVG
coordinates in `slides.md` when the nodes are positioned separately with CSS.
That creates two geometry systems that are brittle to maintain.

Prefer one of these approaches:

- simple flow: use Mermaid;
- custom static/interactive visual: use a dedicated Vue/SVG component;
- decorative divider/line only: use CSS.

Connector-heavy components should use a single coordinate system. Define nodes
as data with `x`, `y`, `w`, and `h`; define edges by `from`/`to` node IDs and
anchor sides; compute SVG paths from that data. Use SVG `<marker>` arrowheads
instead of manually positioned arrowhead polygons.

## Visual inspection workflow

### fast-agent orchestration

Preferred interactive interface:

```bash
fast-agent --env .fast-agent go
```

Then request:

```text
@visual-loop review slides 1-5
@visual-loop review slide 7
```

The `visual-loop` agent runs deterministic geometry first and only proceeds to
VLM review after geometry passes, unless explicitly instructed otherwise. The
underlying helper scripts live in `.fast-agent/scripts/`.


Use deterministic geometry checks before VLM review:

```bash
uv run python .fast-agent/scripts/check_slide_geometry.py --range 1-5
uv run python .fast-agent/scripts/check_slide_geometry.py --range 13-15 --fail-on findings
```

Then export screenshots:

```bash
uv run python .fast-agent/scripts/visual_review.py --range 1-5 --clean
```

For model-assisted visual QA:

```bash
uv run python .fast-agent/scripts/visual_review.py --range 1-5 --review
```

Screenshots are written under:

```text
reports/screenshots/
```

For ad hoc model-assisted review, use the project-local visual-review subagent:

```text
.fast-agent/agent-cards/visual-review.md
```

Preferred review order:

1. run deterministic geometry checks;
2. fix all geometry findings;
3. render screenshots;
4. use VLM/visual-review only for qualitative defects such as hierarchy,
   density, malformed charts, poor contrast, ambiguous flow, or visual rhythm.

## What to look for visually

This is a presentation: prefer generous sizing, strong hierarchy, and carefully
intentional negative space. Do not optimize for document-like density.

Assume some information will be delivered as voiceover. Before adding labels,
captions, caveats, or explanatory prose to a slide, ask whether the text is
needed for the audience to parse the visual, or whether it should instead be
spoken by the presenter. Prefer bold structural labels and visual hierarchy over
small explanatory copy.

Watch for:

- clipped or offscreen content;
- broken Mermaid diagrams;
- unreadable contrast or type size;
- inline code, badge, or pill styling with poor contrast or inconsistent theme fit;
- awkward spacing/alignment;
- content that is too small, dense, or cramped for presentation use;
- negative space that looks accidental rather than deliberate;
- diagram connectors whose endpoints or arrowheads imply the wrong flow.
