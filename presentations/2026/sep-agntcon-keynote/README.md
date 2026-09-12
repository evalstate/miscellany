# September AGNTCON Keynote

Independent Slidev deck copied from `../sep-agntcon` on 11 September 2026.
The source deck and July archive are unchanged. No runtime imports depend on them.

## Work here

```sh
cd ../sep-agntcon-keynote
npm ci
npm run dev -- --port 3031
```

For an agent-managed dev server:

```sh
CI=1 NO_COLOR=1 npx slidev slides.md --port 3031 </dev/null
```

## Talk plan

See `PLAN.md` for the rough 10-minute outline and keynote readability priorities.
The copied slides are sketching references only, not the planned keynote content.

## Reference content

- Slides 1–5: legacy elicitation storyboard.
- Slides 6–10: modern elicitation storyboard.
- Slides 11–14: HTTP inspection, tool routing to inference/API/sandbox servers,
  and optional tool-defined routing keys.

All frames advance manually. Dialog buttons are illustrations.
Edit `slides.md` for narrative and speaker notes; `style.css` for shared visuals.
See `AGENTS.md` for editing and protocol-verification guidance. HTTP labels are
conceptual; verify exact protocol syntax against an available source checkout.

## Commands

- `npm run build` — static site in `dist/`.
- `npm run build:single` — single-file build in `dist-single/`.
- `npm run export` — `sep-agntcon-keynote.pdf` (requires browser export tooling).
- `npm run format` — format source files.

The toolchain, lockfile, route setup, and optional storyboard composable are
retained. Dependencies, build outputs, and agent runtime files were not copied.
