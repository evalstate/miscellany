# Benchmarks (working title)

Slidev deck using the new **fast-agent "Forward" design system**: atomic-age
salesmanship with lab-grade receipts. Tooling is the same as `../sep-agntcon`.

## Preview and build

```sh
npm ci
CI=1 NO_COLOR=1 npx slidev slides.md --port 3030 </dev/null
npm run build          # dist/
npm run build:single   # dist-single/index.html, standalone
npm run export         # PDF
```

Restart the dev server after you edit `index.html` (font links are read at startup).

## Where things are

| Path | What |
| --- | --- |
| `slides.md` | The deck: intro, then 4 parts (Why / What / Planning and running / Reading), each opened by a `section` divider |
| `analysis/` | Source analyses and chart generators (TB 4 subset calibration, Claude Code and Fable 5 harness charts). See `analysis/README.md` |
| `kit.md` | Demo slides for every layout and component. Run with `npx slidev kit.md` |
| `style.css` | Brand base styles (type, lists, links, code, tables) on top of the tokens |
| `index.html` | Fonts merged into `<head>`: Fraunces, Ultra, Figtree, DM Mono |
| `design/tokens/` | Tokens copied verbatim from the design system |
| `design/README.md` | Design system rules. **Read this first.** |
| `design/fast-agent-system/` | Full design-system reference (without `uploads/`). Open `brand-overview.html` for the sample cards, `guidelines/*.html` for each rule, and `components/` and `ui_kits/` for the React references |
| `design/CHARACTER-DIRECTION.md` | Rules for the presenter character |
| `design/design-system-notes.md` | Copy of `~/design-system.md` (sources, palette, website decisions) |
| `public/fa/` | Brand assets copied from the website (`docs/docs/assets/forward/assets/`), a superset of the design-system assets |
| `public/fa/illustration/poses/` | Approved presenter poses A–F cut from `presenter-approved-poses.png` (transparent PNG) |

Assets are served from `/fa/...`, for example `<img src="/fa/icon-tile.svg">`.

## Layouts

- `cover`: headline and `::meta::` slot on the left, the animated homepage splash
  (burst plus typed tagline) on the right. Frontmatter: `tagline`.
- `default`: ivory page with a flat footer showing the tile
  mark and page number (`footer: false` hides the footer).
- `section`: petrol divider. `number` puts an Ultra numeral in a score burst.
- `presenter`: content plus the presenter. Frontmatter: `pose` (A–F), `side`,
  `height`. She stands where her gesture points at the content:
  B and D on the right, C and F on the left.

For a petrol ground on any layout, add `class: fa-dark`.

## Components

- `<Burst shape="capsule|star|score" tone="orange|amber|petrol|teal" :size :tilt>`:
  the one shout per slide. Ultra lettering, pops in, rests at −8°.
- `<Sticker tone="paper|amber|petrol">`: tilted label with a solid offset shadow.
- `<Card tone="paper|ivory|deep|amber|orange|petrol">`: editorial card with a 2px keyline. Title it with `###`.
- `<Receipt n date model harness source mark>`: the disclaimer that travels with a
  number. Missing fields show "TBC".
- `<Terminal :lines="['$ command', 'output']">`: petrol console with the ❯ prompt.
- `<PresenterPose pose="B" :height="420" />`: the character by herself.
- `<VerifierDesigns :phase="$clicks" />`: TB 2.1 / DeepSWE 1.1 / TB 4 verifier lanes across Setup → Agent time → Verifier. The slide sets `clicks: 3`. Lane content is data in the component.
- `<Wordmark tagline :size :dark>`: the stacked fast-/agent splash.
- CSS helpers: `.fa-mark` (amber highlighter; `.fa-mark--orange`),
  `.fa-grid-2`, `.fa-grid-3`, `.fa-grid-4`, `.fa-compact` (denser cards and tables), `.fa-draft` (placeholder text), and `.fa-us` on a table cell to highlight the fast-agent row.

## Presenter poses

A: board · B: open palm · C: pointing · D: folder · E: leaning on a sign · F: palm up.
These are cut from the approved sheet without redrawing. Don't use
`presenter-reference-palm.png`, whose proportions were rejected. The single-pose PNGs
(`presenter*.png`, `pointer`, `sitter`, `bust-wink`) come from the original design
system. The `-install-*`, `-reference-*` and `space-age-terminal*` files are website
iteration artwork. `space-age-terminal-paddles.png` is the current one.

## Design source

`~/Documents/Codex/2026-09-27/referenced-chatgpt-conversation-this-is-an/work/fast-agent-design/`
(branch `docs/forward-redesign`). Re-copy `docs/docs/assets/forward/` if the
website artwork changes.
