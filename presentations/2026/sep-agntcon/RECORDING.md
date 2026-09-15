# Native charts and video capture

The adoption and tool-quality slides are native Vue/SVG charts, not embedded HTML pages or videos.
Their unchanged source aggregates, CSVs, change notes and provenance are in
`data/protocol-adoption/` and `data/tool-quality-version/`. The regular slide controls
include data/provenance download links and count tables. The source HTML remains untouched.

**Private aggregates: review before external sharing.** These are pinned historical
snapshots with different coverage windows and metrics, not live data. No upload,
remote scan or refresh is performed by this workflow. Keep the interpretation and
non-causal caveats from the speaker notes when reusing these charts.

## Capture

Requirements: `npm install`, `npx playwright install chromium`, `ffmpeg`, `ffprobe`.
Start the deck using `CI=1 NO_COLOR=1 npx slidev slides.md --port 3031 </dev/null`.

```sh
npm run capture:charts
# Selected dashboard variant:
npm run capture:charts -- --chart=protocol-adoption --client=chat-ui-mcp --viewport=seven
# Smaller review copy:
npm run capture:charts -- --chart=tool-quality-version --width=1280 --fps=30
```

Other options: `--base=http://localhost:3031`, `--out=recordings`, `--chart=all`.
Slide positions are discovered from `slides.md`, not hard-coded. The default adoption
video is Overall / full history at 4× (7.5 seconds of motion, 11 seconds including holds).
Use `--speed=1` for the original 30-second motion, or another positive speed. Use the regular chart selector to inspect identities;
`configure` uses exact data IDs, not display labels. Record each desired client/view
as its own clip. No clip is uploaded or inserted into a remote presentation automatically.

Outputs are local and gitignored:
- 1920×1080, 30 FPS **H.264 MP4**, yuv420p, faststart, silent, no controls/cursor.
- Final-frame **PNG poster**.
- `.capture.json` with frame count, timing, settings, hashes and ffprobe validation.

Capture seeks every frame to an exact timestamp before taking a browser screenshot;
it does not screen-record a best-effort live animation. So rendering can take longer
than the video duration without dropping frames or changing the animation timing.
Each clip has a one-second opening hold and 2.5-second closing hold in addition to
any holds intrinsic to the source animation. Versions/guides/line reveals retain
source timing. Do not edit the deck while capture is running.

For a clean interactive preview, append `?capture=1` to the chart's slide URL. This
also enables `window.__deckCapture` for local automation. Normal pages expose no
recording bridge. SVG font sizes/opacity use inline styles to avoid Slidev's UnoCSS
attributify rules reinterpreting SVG presentation attributes.

## Use in other presentation tools

**PowerPoint:** Insert → Video → This Device. Select the MP4, set playback on click
or automatically, and use the matching PNG as a poster if desired. The clip has no
loop seam; enable looping only if an end-to-start jump is acceptable.

**Google Slides:** after reviewing the private-data warning, upload the MP4 to your
Google Drive, then Insert → Video → Google Drive. Set playback behaviour in Format
options. Wait for Drive processing and test permissions using the presenting account.
Google Slides video playback generally depends on access to Drive/network resources.

The videos intentionally lose chart scrubbing/client selection. The Slidev originals
and accessible source data remain editable for regenerating different clips.

## Six-client contact sheet

The native static comparison follows Protocol Adoption. Its source is the completed
September 14 reference snapshot (rates require ≥100 valid calls), NOT the dashboard's
partial September 15 snapshot (no support threshold). Reference percentages are
preserved, not recalculated. Source files and normalization script live in
`data/client-adoption-comparison/`. The exported PNG/SVG are in `recordings/`.
