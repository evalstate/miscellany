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
