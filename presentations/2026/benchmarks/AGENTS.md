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
