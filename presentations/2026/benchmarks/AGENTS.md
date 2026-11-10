# Agent notes

This is a Slidev conference deck using the fast-agent "Forward" design system.

- Read `design/README.md` (brand rules) and `design/CHARACTER-DIRECTION.md`
  before designing slides. Visual reference: `design/fast-agent-system/brand-overview.html`
  and `guidelines/*.html` (for example, colour contrast pairs). Key rules:
  - Ivory ground and petrol ink. Keylines are 2px, never 1px.
  - Amber means fast-agent only. Use `--amber-ink` for amber-family text.
  - Max one burst, one tilt and one shout (Ultra) per slide. Data, headlines
    and buttons never tilt.
  - No gradients, blur, soft shadows, left-border accent cards, emoji,
    stat pills or hype verbs (supercharge, unleash, seamless...).
  - Every number carries its receipt on the same slide: n, date, model,
    harness version. Round down. Use `<Receipt>`.
  - British English, sentence case, "fast-agent" lowercase and hyphenated.
  - One mascot per slide, outside chart/data areas. Keep her approved identity.
    Don't generate or redraw her.
- **No eyebrows.** No kicker, overline or small uppercase label above a heading,
  on slides, dividers or cards. They waste space, nobody reads them, and they are
  an LLM tell. If a card needs a title, use a real heading (`###`). The layouts
  and Card deliberately have no eyebrow/kicker props: don't add them back.
- Use the design tokens (`var(--petrol)` etc.) rather than raw hex values.
- Prefer Slidev markdown and the existing layouts before adding Vue components.
- Put shared visual patterns in `style.css`; keep narrative in `slides.md`.
- Don't give a layout and a component the same name. A layout named `x.vue`
  that uses `<X>` resolves to itself and recurses.
- Run `npm run build` after structural changes. Review the rendered slides for
  clipping, density, contrast and alignment.
- Start the dev server detached from stdin:
  `CI=1 NO_COLOR=1 npx slidev slides.md </dev/null`.
- Don't commit `node_modules/`, `dist/`, `dist-single/`, reports or PDFs.

## Talk source material (atif-scan research)

The talk content comes from atif-scan research done in other herdr-core
sessions (`hc ls`: tabs `atif-talk-prep`, `publish`, `atif-scan`). The
narrative draft is `talk.md`. Read the source material and don't invent
numbers: every figure must trace back to these files.

- Tool: `~/source/atif-scan` (README, SECURITY.md). Session history is in
  `.fast-agent/sessions/`. Talk prep is `2611080938-be1TZH`, and the DeepSWE
  calibration work is `2611090825-lPkYHu`.
- Reviews: `~/.cache/atif-scan/reviews/`. Open each folder's `index.html` or README:
  - `talk-hunt-2026-11-08/`: start with `SUMMARY.md`, which has the headline
    stories by category: network evasion, sandbox probing, side channels,
    tampering and concealment, awareness, credentials and hosts. Then read
    `stories/` (40 verified cards), `round2/SUMMARY.md` (find-me,
    fingerprints, footprint, oddities), `strategies/` (OCR vs vision, the
    ASCII renderer), `defective-task/` (the TB-lite answer download on a
    broken task), `census/CENSUS.md` (corpus size, tokens and tool calls) and
    `rerun/DELTA.md` (Datacurve effort grid).
  - `lure-timeline-2026-11-06/` (install lure, see `talk-notes.md`),
    `model-calls-2026-11-07/`, `env-dump-2026-11-07/`,
    `lab-host-timeline-2026-11-07/`, `corpus-2026-11-08/` (181,732-trace scan)
    and `tb21-network/` (which TB 2.1 tasks need network egress).
- DeepSWE calibration, 9 Nov, still in progress: `~/.cache/atif-scan/bundles/deepswe-{upstream-audit,guided-random50,patch-reassurance,crossmodel100}-20261109/`.
- Public, shareable: HF Spaces `evalstate/{tb21-annotated,tb21-submissions-annotated,deepswe-annotated,tb21-fast-agent-annotated}`
  (atif-scan b344b40).
- Everything under `~/.cache/atif-scan` is private. Slides may only use
  masked quotes. Never show raw trace paths, run IDs, credentials or
  unmasked trace text. Step numbers are ATIF `step_id`, not the zero-based
  `usteps` index. Present findings as reviewed candidates, not verdicts, and
  keep the caveats each report gives.
