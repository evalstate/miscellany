# Native comparison integration

Use `ClientAdoptionComparison` inside a parent-sized wrapper. Its root and SVG fill
that wrapper; the logical canvas is 1600×900. Prefer a 16:9 wrapper. `title` owns
the SVG heading (default: `2026-07-28 Protocol Adoption (Tool Calls)`); audience
narrative belongs in parent markdown. `capture` hides all HTML controls/overlays.
There is no animation, clock, navigation handler or global listener. Slidev
`useNav().isPrintMode`, `/print` routes and print CSS hide controls.

Export the native `[data-role="comparison-svg"]` SVG, not the entire component.
Styles needed by SVG text/marks are inline or presentational SVG attributes;
font size and opacity specifically use inline styles to avoid Uno attributify.

## Data fidelity and parent speaker-note recommendations

- This is the reference completed-day July 28–September 14 snapshot, **not** the
  current September 15 dashboard. Last7 is September 8–14 inclusive.
- Unlike ProtocolAdoptionChart's unsuppressed dashboard rates, supplied source
  rates require at least 100 valid calls. Strict trailing-seven and last7 also
  require seven calendar dates. Blank values remain null and interrupt paths.
- Values come directly from `daily_share`, `trailing7_share` and summary
  `last7_share`, never recomputed from counts. Multiplication by 100 only formats
  percentages. Source counts are unsuppressed. Volume geometry alone is scaled.
- Source order and exact identities are retained. No merging chat-ui-intern into
  chat-ui-mcp; no merging codex-mcp-client into openai-mcp (Codex).
- Mini bars use all-protocol daily counts, each with its own maximum: activity
  patterns, **not relative popularity**. Unknown/unreviewed protocols contribute
  volume but not rate denominators. Fixed five excluded; missing hashes retained.
- Private review aggregates: review before public sharing. Self-reported client
  identities are not users or evidence of individual product migrations. Padded
  partitions do not establish complete real-world traffic. See original README
  and provenance, retained unchanged, for full source/coverage warnings.
- Last7 in panel order: **95.2%, 99.3%, 100.0%, 28.4%, 15.1%, 0.0%**.

## Reproduction and selectors

Run `python data/client-adoption-comparison/generate.py`. Uses Python's standard
CSV parser (quoted cells/newlines supported). Verifies source SHA256 hashes,
identity order, calendar coverage, supplied null policies and summary/stats
agreement. Emits `normalized.json`; does not alter the five copied artifacts.
Original README describes the source's cyan/lime rendering, not this adaptation.

- Root: `[data-chart="client-adoption-comparison"]`, with `data-through` and
  `data-min-valid-calls`.
- Panels: `[data-role="client-panel"][data-client="…"]` in source order.
- KPI: `[data-role="last7"]` with raw supplied `data-value`.
- Series: `[data-series="daily_share"]`, `[data-series="trailing7_share"]`.
- Points: `[data-role="rate-point"]`, `data-date`, raw `data-value`.
- Activity: `[data-role="activity"]`, `data-scale="independent"`, `data-max`;
  child rectangles retain `data-date` and `data-count`.
- Controls: `[data-action="provenance"]`, `[data-role="data-overlay"]`,
  `[data-control="table-client"]`, `a[download]`.

Validation: deck `npm run build` passed. Isolated Vue/Vite browser render inspected
at 1600×900: all six KPIs, 48px title, no canvas-clipped text, six downloads,
49 daily table rows/client, and print-media hiding checked. The isolated harness
mocked Slidev navigation; it did not modify or mount into slides.md.
