# July Release Party

A Slidev presentation based on the Tessl MCP deck's tooling and visual language,
with independent content and assets.

## Edit

```bash
npm install
npm run dev
```

Open `http://localhost:3030/print` for a live vertical slide stack.

## Build

```bash
npm run build
npm run preview
```

The regular build is an ES-module SPA and must be served over HTTP.

## Export

```bash
npm run export
```

## Single-file build

```bash
npm run build:single
xdg-open dist-single/index.html
```

## Structure

- `slides.md` — presentation narrative and composition
- `style.css` — global tokens and shared presentation styles
- `layouts/` — reusable Slidev layouts
- `components/` — data-driven or reusable visual components
- `public/` — static images, video, and other assets

