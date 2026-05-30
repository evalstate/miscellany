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

## Common commands

```bash
npm run dev
npm run build
npm run preview
npm run export
```

Build output in `dist/` is an ES-module SPA; serve it over HTTP. Use
`npm run build:single` only when a filesystem-openable single HTML file is
needed.

## Visual inspection workflow

Use the helper first:

```bash
python3 scripts/visual_review.py --range 3
python3 scripts/visual_review.py --range 1-5 --clean
```

Screenshots are written under:

```text
reports/screenshots/
```

For model-assisted visual QA:

```bash
python3 scripts/visual_review.py --range 1-5 --review
```

The helper uses Slidev PNG export with Chrome. If an issue only appears in live
dev mode, run `npm run dev` and use direct Chrome screenshots against
`http://127.0.0.1:3030/` or `http://127.0.0.1:3030/print`.

## What to look for visually

This is a conference presentation: prefer generous sizing, strong hierarchy,
and carefully intentional negative space. Do not optimize for document-like
density.

Assume some information will be delivered as voiceover. Before adding labels,
captions, caveats, or explanatory prose to a slide, ask whether the text is
needed for the audience to parse the visual, or whether it should instead be
spoken by the presenter. Prefer bold structural labels and visual hierarchy over
small explanatory copy.

- clipped or offscreen content;
- broken Mermaid diagrams;
- unreadable contrast or type size;
- inline code, badge, or pill styling with poor contrast or inconsistent theme fit
  — e.g. light default inline-code pills on the dark amber deck;
- awkward spacing/alignment;
- content that is too small, dense, or cramped for presentation use;
- negative space that looks accidental rather than deliberate;
- reusable diagram components should be geometrically pleasing at any rendered
  size: the outer container controls scale, internal padding is proportional,
  section labels are legible, and cards should not be much taller than their
  icon/title content requires;
- avoid reusable diagram APIs such as `size="sm|md|lg"` when the component can
  instead fill a parent-defined region. Prefer a slide/layout wrapper that sets
  available width/height, then use proportional CSS variables, `clamp()`, and
  container query units inside the component;
- labeled containers should give their section labels enough size and breathing
  room to read as structural labels, not tiny captions squeezed into the border;
- icon/title cards should read as a coherent unit, not as an icon stranded at
  the top with a label stranded at the bottom and a large empty middle;
- custom Vue component rendering defects;
- title/content hierarchy problems;
- code blocks or terminal snippets overflowing;
- export-only differences from dev rendering.

When giving feedback, cite the slide number and make recommendations actionable.
