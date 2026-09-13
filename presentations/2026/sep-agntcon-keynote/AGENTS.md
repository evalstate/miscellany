# Agent notes

This is a Slidev conference deck.

- Reload `slides.md` before edits that depend on slide order or content.
- Prefer Slidev markdown and existing layouts before adding Vue components.
- Keep audience-facing narrative in `slides.md`.
- Use components for reusable, interactive, or data-driven visuals.
- Put shared visual patterns and tokens in `style.css`.
- Let slide wrappers control component dimensions.
- Run `npm run build` after structural changes.
- Review rendered slides for clipping, density, contrast, and alignment.
- When starting the Slidev dev server from an agent, detach stdin to prevent
  terminal CPR/raw-keyboard issues: `CI=1 NO_COLOR=1 npx slidev slides.md </dev/null`.
- Do not commit `node_modules/`, `dist/`, `dist-single/`, reports, or PDFs.

## Keynote readability — a primary requirement

- The presenter strongly dislikes small fonts and unnecessary tiny detail.
  Design for delegates at the back of the room, not close-up laptop viewing.
- Use the full slide canvas for large diagrams, charts, and text, with safe
  margins. Do not confine the main content to small panels or oversized chrome.
- Voiceover-led slides: remove nonessential text and put supporting detail in
  speaker notes. Simplify or split content instead of reducing font sizes.
- Chart labels and legends are audience content too: keep them large, sparse,
  and readable. Avoid tiny footnotes, badges, and decorative microcopy.
- Use subtle, deliberately chosen colours with robust contrast. Muted palettes
  must not mean faint text or nearly invisible chart lines. Plan for washed-out
  projectors and varying display technology, screen size, and viewing distance.
- Target at least 4.5:1 text contrast and 3:1 for essential graphical elements;
  treat these as a floor, not a guarantee of room-scale legibility. Never rely
  only on colour to distinguish routes, series, or states.
- Review rendered slides for distance readability as well as clipping. Check
  essential labels and marks under reduced contrast, not just on a good monitor.
- The copied reference slides are not a typography or density standard for the
  keynote. Reuse only what serves these readability requirements.
- Planning is intentionally rough for now; data and content will be supplied
  later. “Tasks for model cache control” means keeping models warm.

## Animated storyboards

- Model narrative animations as typed semantic frames. Put state such as
  `phase`, `hold`, `flash`, and direction on frames instead of deriving the
  story from hard-coded frame indexes.
- Use `composables/useTimedStoryboard.ts` for linear timed sequences that need
  restart, looping, animation keys, and timer cleanup. Keep routing, scrubbing,
  or data-specific behavior in the owning component.
- Prefer a manually triggered first run for talk timing. If a sequence loops,
  begin looping only after that first audience-triggered activation.
- Verify protocol method names, direction, lifecycle state, and the illustrated
  specification version against the local protocol checkout before editing.
- Test animations by waiting for semantic DOM states (labels/classes), not only
  by taking screenshots after guessed delays.
- Slidev scales its logical slide canvas to the presentation viewport. Validate
  rendered or computed geometry before using CSS pixel values as screen-space
  measurements.

## This workspace

- This is the standalone September AGNTCON keynote deck. Edit the root `slides.md`.
- `../sep-agntcon` is the source deck; `../july-release-party` is the historical archive. Do not edit either as part of this deck.
- `PLAN.md` holds the rough 10-minute keynote outline. `slides.md` contains seven large-type section placeholders matching it; timings are in speaker notes.
- The original reference storyboards remain in `../sep-agntcon`; they are not part of the active keynote deck.
- Animated charts are requested for the planned client migration section. The current placeholders are static and contain no data; keep other visuals static unless further animation is requested.
- The previously referenced protocol checkout is `/home/ssmith/source/modelcontextprotocol` (unavailable when this deck was copied). Locate and verify sources before developing protocol claims or behaviour.
- `composables/useTimedStoryboard.ts` is available for future animation work; the current slides do not use it.
