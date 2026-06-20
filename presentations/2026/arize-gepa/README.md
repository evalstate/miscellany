# Arize GEPA presentation

Slidev deck scaffold with project-local fast-agent visual review tooling.

## Run while editing

```bash
npm install
npm run dev
```

Live vertical stack view is available during dev at:

```text
http://localhost:3030/print
```

## fast-agent visual loop

From the deck root:

```bash
fast-agent --env .fast-agent go
```

Then ask the project-local orchestration agent, for example:

```text
@visual-loop review slides 1-5
@visual-loop review slide 7
```

The `visual-loop` agent runs geometry first, exports screenshots, then invokes
the `visual-review` VLM pass with the PNGs attached.

## Visual loop

Run deterministic layout checks first:

```bash
npm run visual:geometry -- --range 1-5
```

Export screenshots:

```bash
npm run visual:export -- --range 1-5 --clean
```

Ask fast-agent/VLM for screenshot review:

```bash
npm run visual:review -- --range 1-5
```

Screenshots and review artifacts are written under `reports/`.

## Build

```bash
npm run build
npm run preview
```

Single-file build:

```bash
npm run build:single
xdg-open dist-single/index.html
```
