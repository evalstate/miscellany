> Historical restoration record from before the deck split. These chart components,
> data and exports are now owned by `sep-agntcon-keynote`; use its README for current
> slide numbers and preview URLs. No data was refreshed during the split.

# Source-design restoration

Scope: `sep-agntcon` only. The keynote experiment is independent and was not edited.

The pre-HF native-chart baseline is commit `5d21514`. Earlier local build logs
also describe white 16:9 canvases, Arial, purple daily/Claude series, slate
comparison series and amber server-version indicators. The restored components
use that baseline, checked against the original HTML and saved previews in
`/home/evalstate/source/data-analysis/charts/`.

- `legacy-message-animation`: large uppercase heading, #4531ba tool square,
  original grey categories, square cells and original geometry. Source footnote
  positioning restored (5.2% from bottom). Native playback stays manually started,
  unlike the standalone HTML's autoplay. Deterministic capture support is retained.
- `tool-quality-version-focus`: authoritative design is `version-focus-v8`.
  `index.html` mtime is 2026-09-15 17:35:02 +02:00; browser-validation report is
  17:35:10 +02:00. Preceding top-ribbon v7 was 17:08:07, drops v6 16:55:29,
  clean v5 16:42:25, feedback v4 15:56:21. These are design/file timestamps,
  not telemetry freshness. Original black title/date restored as well.
- Adoption dashboard and completed six-client contact sheet: restore the native
  pre-HF styling; preserve the requested title and adoption's default 4× speed.

## Quality data freshness

Latest compatible local publication is
`summaries/telemetry-fixed-five-20260914/series-0-4/`, run date 2026-09-14.
Chart observations span August 25–September 13. The last rendered daily anchor is
2026-09-13T12:00:00Z (illustrative noon, not a last-event timestamp). The animation
boundary at September 14 00:00 UTC is exclusive and is not September 14 data.

Newer client-protocol/protocol-overall publications contain no tool-quality
classifications; none of the inspected local sources extends this quality series.
No remote refresh or reanalysis was performed. Private aggregates still require
review before public sharing.

The following source and deck files are byte-identical (SHA-256):

| File | SHA-256 |
| --- | --- |
| data.json | c1574c5d06eb8e7a7c57e5d1be3f7245380c127e2c6b5e2d42e4f9afbccf0178 |
| daily.csv | e8941d02ceeada0958f157a429bd29d3887b1208b6224b75afc3b038c4c6cf27 |
| changes.json | ac2f64cd4c60e6b9351d3ed6d7b1805a981c5c9ec651116b94dba6f5f807dd93 |
| provenance.json | bf9ba7ddbed2ce8395fec29c9d35433b245ec86548075fe7abba789df893180a |

## Review and exports

Review chart slides 22–25 on port 3031 (shifted by the added legacy-video slide 6). `npm run test:charts` exercises capture seeking,
count boundaries, timing, controls, data and print; `npm run build` validates the
deck. Videos and posters in `recordings/` have now been regenerated from the
restored design: message ratio, adoption at 4×, and tool quality. The six-client
PNG/SVG contact sheet is also refreshed. The older adoption 1× clip is not part of
this refreshed set. All generated files remain local and gitignored.
