# September AGNTCON — Transport

This is the **HF/yellow transport deck**. Edit `slides.md` here for this presentation.
The separate **white/purple keynote** lives in `../sep-agntcon-keynote/`.
Neither deck has runtime imports from the other; July remains an untouched archive.

## Preview and build

```sh
npm ci
CI=1 NO_COLOR=1 npx slidev slides.md --port 3031 </dev/null
npm run build
```

Preview: http://localhost:3031/

## Current slides

- 1: Speaker intro
- 2–5: HF-styled static and animated Legacy/Modern MCP comparisons
- 6–7: HF MCP Server topology variants
- 8–16: Existing elicitation storyboard frames (unchanged)
- 17–20: HTTP routing storyboard

The purple communications diagrams, message ratio, adoption charts and tool-quality
chart were moved to the keynote, with their components, data, tests and capture tools.
For those exports, use `../sep-agntcon-keynote/recordings/`, not this deck.

## Editing

- `slides.md`: audience content, ordering, speaker notes.
- `style.css`: HF theme, diagrams and static storyboard styling.
- `components/CommunicationTraffic.vue`: existing HF animation.
- `components/CapabilityIcon.vue`: capability glyphs.
- `composables/useTimedStoryboard.ts`: shared animation helper, independently copied.

`npm run build:single` creates a standalone build; `npm run export` exports PDF.
Read `AGENTS.md` before changing protocol behaviour. Existing specification-source
verification caveats and distinctions around modern retry parameters remain in notes.

Pre-split source backup: `.deck-backups/sep-transport-before-split.tgz` (local/ignored).
No slides were dropped from this deck except the six purple visuals moved to keynote.
