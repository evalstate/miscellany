# Native comparison integration

`ClientAdoptionComparison` fills its parent; SVG canvas is 1600×900. Keep a 16:9
wrapper. White/purple styles, typography, six-panel order and independent activity
scales are unchanged. `title` controls the heading; `capture` hides controls.
Print mode and print CSS hide controls and overlays. No animation or listeners.

Pinned data now covers completed July 28–September 15 calendar days. Last7 is
September 9–15. Both daily and rolling series include September 15 with ordinary
filled points and bars. Final tick: Sep 15. No partial-day markers or caveats.
Ingestion can still be revised; completed calendar coverage is not final ingestion.

The component formats supplied fractions only; the reproducible aggregate adapter
and display normalization live in generate.py (see README). Daily CSV is
unsuppressed; normalized daily/rolling rates require ≥100 valid calls and rolling
rates require seven calendar dates. Null breaks paths. Counts remain available.
Overall is retained in data but not drawn. Exact identities are not merged:
chat-ui-mcp is not chat-ui-intern; codex-mcp-client is not openai-mcp (Codex).
Independent all-protocol activity strips show patterns, not relative popularity.

Last7 in panel order: **95.3%, 99.5%, 100.0%, 32.6%, 15.3%, 0.0%**.

Selectors: `[data-role="comparison-svg"]`, `[data-role="client-panel"]` with
`data-client`, `[data-role="last7"]` with raw `data-value`, daily_share and
trailing7_share `data-series`, rate-point with `data-date` and `data-value`,
activity with independent `data-scale` and `data-max`. Bars retain date/count.
Controls: provenance action, data-overlay, table-client and download links.

Parent speaker notes, capture metadata and full chart tests now use the same completed-day snapshot.
No reference renderings are required or copied.

Validation: generator passed publication/hash/reconciliation and exact KPI checks;
second regeneration was byte-identical for all five generated artifacts without
PDF/rendered inputs. `npm run build` passed. Targeted Playwright checks passed for
all six KPIs, every plotted daily/rolling value and activity count, September 15
endpoints, absence of partial markers, text within SVG bounds, 50-row overlay,
six downloads and print hiding. Rendered 1600px contact sheet reviewed: original
palette/hierarchy/alignment preserved, no clipping. The repository-wide chart test suite was updated for this completed snapshot and passed.
