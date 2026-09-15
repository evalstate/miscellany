> This workflow now belongs to `sep-agntcon-keynote`. Run commands from that directory.
> Approved exports and older alternatives are identified in `recordings/README.md`.
> Export metadata from before the split retains its original slide numbers and hashes.

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
Start the deck using `CI=1 NO_COLOR=1 npx slidev slides.md --port 3030 </dev/null`.

```sh
npm run capture:charts
# Message-ratio animation only:
npm run capture:charts -- --chart=legacy-message-ratio
# Selected dashboard variant:
npm run capture:charts -- --chart=protocol-adoption --client=chat-ui-mcp --viewport=seven
# Smaller review copy:
npm run capture:charts -- --chart=tool-quality-version --width=1280 --fps=30
```

Other options: `--base=http://localhost:3030`, `--out=recordings`, `--chart=all`.
Slide positions are discovered from `slides.md`, not hard-coded. The default adoption
video is Overall / full history at 4× (7.5 seconds of motion, 11 seconds including holds).
Use `--speed=1` for the original 30-second motion, or another positive speed. Use the regular chart selector to inspect identities;
`configure` uses exact data IDs, not display labels. Record each desired client/view
as its own clip. No clip is uploaded or inserted into a remote presentation automatically.

The message-ratio slide uses its original source design: Arial, white background,
purple tool square, grey message categories, and the large uppercase headline. Its source
sequence lasts 11.58 seconds; with opening/closing holds and frame rounding, the
30 FPS export lasts 15.1 seconds. Capture seeks cell arrivals and paused flash/fade
animations deterministically. The final frame shows 74 equal squares: one tool
call and 73 other messages (27 initialization, 39 listing, seven other). This is a
rounded aggregate ratio, not a literal message sequence; provenance is in
`data/legacy-message-ratio.provenance.json`.

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

For a Google Slides quality check, wait for Drive video processing, then test in
presentation mode at the intended screen resolution. Compare a paused closing
frame against the matching `-poster.png` reference at the same displayed size,
looking especially at small version labels and thin lines. Drive/Slides playback
may use a processed rendition rather than the original file; the local MP4 is
H.264 CRF 18 at 1080p/30 FPS. A sharp local file does not guarantee identical
streamed playback. No Drive upload or cloud playback validation is performed here.

The videos intentionally lose chart scrubbing/client selection. The Slidev originals
and accessible source data remain editable for regenerating different clips.

## Six-client contact sheet

The native static comparison follows Protocol Adoption. Its source is the completed
September 14 reference snapshot (rates require ≥100 valid calls), NOT the dashboard's
partial September 15 snapshot (no support threshold). Reference percentages are
preserved, not recalculated. Source files and normalization script live in
`data/client-adoption-comparison/`. The exported PNG/SVG are in `recordings/`.

## Large-room legacy protocol video

`npm run capture:charts -- --chart=legacy-protocol-video` captures the new white /
purple / slate diagram (slide discovered from markdown). Existing Legacy/Modern
slides remain unchanged. Single filled-path arrows have no separate heads or labels.
Six independent activations alternate direction. Each travels for 1,275 ms, then
holds the receiver/capability highlight for 5,000 ms, with gentle 250 ms edge fades.
The last highlight remains selected. Native sequence: 37.65 seconds; 30 FPS MP4
with standard opening/closing holds: about 41.17 seconds. Playback is manually
started and one-shot by default; optional looping begins only after activation.

This is a conceptual illustration, not telemetry or a literal protocol trace.
`data/legacy-protocol-video.provenance.json` records its source and timing.
