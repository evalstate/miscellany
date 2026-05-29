# Tessl MCP presentation

Fresh Slidev starter deck.

## Run while editing

```bash
cd tessl-mcp
npm install
npm run dev
```


## Live vertical stack view

Slidev v52 does not enable the built-in `/print` route during dev by default. This deck adds it back via `setup/routes.ts`, so you can use a Marp-like vertical scroll preview while editing.

```bash
npm run dev
```

Then open this in a browser or VS Code Simple Browser:

```text
http://localhost:3030/print
```


## Build and preview the static output

```bash
npm run build
npm run preview
```

Do not treat `dist/index.html` like a Marp standalone HTML file. Slidev builds an ES-module SPA, so serve `dist/` over HTTP. The build script uses `--base ./ --router-mode hash` so the output is safe to host from a subdirectory.


## Build a single HTML file for filesystem viewing

```bash
npm run build:single
xdg-open dist-single/index.html
```

This creates a large, self-contained `dist-single/index.html` by inlining the Slidev JS/CSS bundle. This is the closest equivalent to a Marp-style HTML file that can be opened from the filesystem.

Caveats:

- The file is much larger than the normal build.
- Browser/security behaviour around `file://` can still vary. Tested with Chrome.
- External embeds, iframes, remote fonts, videos, and large assets may still need special handling.

## Export

```bash
npm run export
```

PDF export may need Playwright/browser setup depending on your environment.

## Files

- `slides.md` — main deck content
- `style.css` — global design tokens and base slide styling
- `layouts/` — reusable slide layouts
- `components/` — reusable content components
- `public/` — images, videos, demos, and other static assets
