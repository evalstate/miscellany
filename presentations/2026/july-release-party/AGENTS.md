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
