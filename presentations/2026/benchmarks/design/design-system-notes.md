# Fast-Agent design system

## Locations

The supplied design system, copied into the working repository:

`/home/evalstate/Documents/Codex/2026-09-27/referenced-chatgpt-conversation-this-is-an/work/fast-agent-design/design/fast-agent-system/`

Working repository:

`/home/evalstate/Documents/Codex/2026-09-27/referenced-chatgpt-conversation-this-is-an/work/fast-agent-design/`

Design branch: `docs/forward-redesign`. The working checkout contains ongoing, uncommitted design refinements; it is the place to continue this work.

Local Zensical preview: http://127.0.0.1:8001/ (available while the preview server is running).

## Website implementation

Paths below are relative to the working repository above:

- `docs/docs/stylesheets/forward.css`: website styling, responsive layouts, atomic bursts, splash animation, and terminal lamp animation.
- `docs/docs/assets/forward/assets/`: brand artwork and assets, including bursts, sparkles, and illustrations.
- `docs/docs/index.md`: homepage layout, content, links, and illustration markup.
- `docs/overrides/main.html`: template customization and font loading.
- `docs/zensical.toml`: Zensical configuration, navigation, and stylesheet registration.

## Visual direction

Clean, optimistic, retro-modern 1950s/1960s atomic-age design. The universe is grown-up and naively perfect: sophisticated mid-century commercial illustration and space-age industrial design, with a Jetsons/Flintstones-era sensibility rather than Futurama's tone. Avoid childish toy proportions, cynical sci-fi, distressed textures, and generic futuristic neon.

Use bold editorial cards, oversized expressive typography, round serrated starbursts (the supplied asset calls one a “score”), restrained atomic sparkles, hard outlines, and offset print-like shadows. Keep layouts compact and functional.

### Palette

- Ivory: `#FFF7E8`
- Paper: `#FFFCF4`
- Petrol: `#082C34`
- Secondary petrol: `#11414B`
- Amber/yellow: `#FFB52E`
- Orange: `#F45125`
- Teal: `#277C80`

### Typography and motion

Fraunces for expressive headlines, Ultra for selected burst lettering, Figtree for body copy, and DM Mono for commands.

The homepage wordmark is stacked as “fast-” / “agent”, slightly spilling beyond a round yellow burst. Dark mode uses ivory lettering with a petrol outline/shadow. The burst expands and rotates before the tagline reveals smoothly; its block cursor blinks briefly and fades. Hover motion should feel tactile without snapping during the introduction. Respect reduced-motion preferences.

## Approved illustration direction

The blonde presenter in a teal suit is a consistent recurring character. Keep her approved face, hair, and proportions across poses; do not reinvent her between assets.

- `docs/docs/assets/forward/assets/illustration/presenter-approved-poses.png`: approved character pose sheet.
- The benchmarks card uses the presenter holding an illustrative generic Pareto chart, not actual plotted benchmark data.
- `docs/docs/assets/forward/assets/illustration/space-age-terminal-paddles.png`: current compact workstation artwork for the Get started card. It has a CRT on the left, 18 amber lamps on the right, and ivory/orange PDP-inspired paddle switches below. The lamps have a separate animated SVG overlay in the homepage markup.

The workstation replaces the presenter in the installation card. Its real command and copy button are HTML controls below the illustration. The copy button is orange, and “Requires uv” appears in a small yellow burst.

## Homepage structure

Desktop top row: 40% brand splash, 30% Get started, 30% benchmarks. Smaller screens reflow the cards.

Benchmarks remains a separate top-level navigation destination. The viewer is a separate follow-up workstream; the homepage work should preserve existing routes and functionality where practical.

The supplied design system is the original reference. The current website files also capture decisions made during the subsequent interactive design iterations.
