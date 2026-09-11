# September AGNTCON

Standalone Slidev deck, starting with the Modern Elicitations storyboard.
Copied from `../july-release-party/ideas/modern-elicitations` on 11 September 2026.
The July workspace remains the historical archive; this deck has no imports or
runtime dependencies on it.

## Work here

```sh
cd ../sep-agntcon
npm ci
npm run dev
```

For an agent-managed dev server:

```sh
CI=1 NO_COLOR=1 npx slidev slides.md </dev/null
```

- `npm run build` — static site in `dist/`, suitable for static hosting.
- `npm run build:single` — single-file build in `dist-single/`.
- `npm run export` — `sep-agntcon.pdf` (requires Slidev's browser export tooling).
- `npm run format` — format deck source and shared helpers.

## Starting storyboard

Ten static frames; advance manually. Dialog buttons are illustrations.

- **1–5: Old way** — start call, ask permission, show dialog while the Server
  stays prominently OPEN / WAITING, return Yes, finish.
- **6–10: Modern way** — start call, prepare question, deliver the yellow
  question card above the Client dialog, return context + Yes, finish.
- Compare **3 and 8**, then **4 and 9** for the central contrast.

## Files

- `slides.md` — audience narrative, visuals and speaker notes.
- `style.css` — standalone deck foundations and storyboard styles only.
- `AGENTS.md` — editing, review and protocol-verification instructions.
- `composables/useTimedStoryboard.ts` — helper retained for possible future animation.
- `setup/routes.ts` — Slidev print route support.
- `vite.config.ts` — optional single-file build support.

No July-specific components, media, alternate sketches or generated outputs were
carried over. The existing Slidev toolchain and lockfile were retained.

Speaker notes distinguish conceptual “Question context + Answer” from MRTR's
original parameters, keyed inputResponses and unchanged opaque requestState when
provided. Protocol source: `/home/ssmith/source/modelcontextprotocol`, legacy
2025-11-25 elicitation and 2026-07-28 MRTR.
