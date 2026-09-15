# September AGNTCON — Keynote

This is the **white/purple keynote deck**. Edit `slides.md` here for keynote content
and ordering. The separate **HF/yellow transport deck** is `../sep-agntcon/`.
Both decks are self-contained; no runtime imports link them.

Your eight draft prose slides (with obvious typos corrected), notes, `PLAN.md` and `speaking-notes.md` are preserved.
The prose now uses the same white/purple family as the approved visuals. The old
midnight/gold experiment is retired. The preferred speaker intro and animated-fan
HF server topology are restored at the start.

## Preview and build

```sh
npm ci
CI=1 NO_COLOR=1 npx slidev slides.md --port 3030 </dev/null
npm run build
npm run test:charts
```

Preview: http://localhost:3030/ (tests accept `SLIDEV_URL` to override).

## Current order

| Slide | Content |
| --- | --- |
| 1 | Shaun Smith — original speaker intro format |
| 2 | Hugging Face MCP Server — Clients → HTTP Router, animated GPU fan |
| 3 | In the Beginning — draft prose |
| 4 | MCP Communications — original layout/icons, slowed pulses, 5s highlights |
| 5 | Streamable HTTP — draft prose |
| 6 | Practical Challenges — draft prose |
| 7 | 1 tool call / 73 other messages — lighter greys, shortened footer |
| 8 | Observability — draft prose |
| 9 | Tool Error Rate — approved 6% plot ceiling |
| 10 | Stateless (2026-07-28) — draft prose |
| 11 | 2026-07-28 MCP Communications — static crossed-out Roots/Sampling |
| 12 | Migration Progress — draft prose |
| 13 | Protocol Adoption — rolling-led reveal, larger figures, final stars |
| 14 | Six-client contact sheet |
| 15 | Deployment Tips — draft prose |
| 16 | Beyond the transport — draft prose |

## Exports and data

- `npm run capture:charts`: deterministic MP4 + poster exports.
- `npm run export:comparison`: contact-sheet PNG/SVG.
- `npm run export`: PDF; `npm run build:single`: standalone HTML build.
- See `RECORDING.md` for capture options and Google Slides/PowerPoint guidance.
- **Approved existing exports:** `recordings/README.md` lists canonical files.
  Earlier variants remain available; videos are local, ignored and not uploaded.
- Source aggregates and provenance moved unchanged to `data/`. Private review data:
  review before public sharing. Fixed-five exclusions and window caveats remain
  in the relevant source files and speaker notes. No data refresh during the split.

Add prose slides with a Markdown heading and three or four bullets between `---`
separators. Keep narrative in `slides.md`, shared appearance in `style.css`, and
reusable visuals in `components/`. Read `AGENTS.md` before editing protocol claims.

Pre-split source backup: `.deck-backups/sep-keynote-before-split.tgz` (local/ignored).
