# Client migration contact sheet — completed September 15

Verified aggregate source: `data-analysis/summaries/client-protocol-20260916-v1`,
revision `cbbf32b5f6bf8f3d73fcc9f76ba91166644fd0c3`.
July 28–September 15, 2026 UTC: completed calendar days. Source last modified
`2026-09-16T00:28:15+00:00`. Last7 is **September 9–15** inclusive.
Completed days do not imply final ingestion; late arrivals may revise counts.

| Identity | Last7 modern / valid calls | Share |
|---|---:|---:|
| Overall | 199797 / 393660 | 50.7536960829% |
| claude-code | 68118 / 71512 | 95.2539433941% |
| Anthropic/ClaudeAI | 40194 / 40408 | 99.4704019006% |
| chat-ui-mcp | 41687 / 41687 | 100.0000000000% |
| openai-mcp | 26783 / 82112 | 32.6176441933% |
| openai-mcp (Codex) | 2507 / 16341 | 15.3417783489% |
| codex-mcp-client | 0 / 1309 | 0.0000000000% |

Exact integers and full-precision rates are in `stats.json` and `day_on_day.csv`.
Overall includes every retained client, not just the six panels. Modern means
2026-07-28 protocol; denominator is reviewed-valid calls. Unknown/unreviewed
protocols contribute only to activity. Fixed five excluded across all identities
and versions; missing hashes retained. Counts are not users or causal migration
evidence. Private aggregate review material: review before public sharing.

## Reproduce offline

```sh
python data/client-adoption-comparison/generate.py
# Optional relocation: --analysis-root /path/to/data-analysis
```

Requires only Python and pinned local aggregate/text inputs, not PDFs, rendered
charts, matplotlib, network access or raw logs. The adapter verifies pinned input
and builder SHA256s, invokes `plot_client_migration_focus.prepare` for COMPLETE,
manifest/output hashes, coverage, fixed-five audit and client/overall cube
reconciliation, then uses `build_protocol_adoption_dashboard.series_data` and
`plot_client_migration_snapshot.comparisons`. It never invokes the snapshot
exporter or its partial-only README/provenance generation.

The completed adapter preserves the full original manifest (including partition
hashes and cohort policy), COMPLETE hash, all input/builder hashes and its own
hash in `provenance.json`. It regenerates daily.csv, day_on_day.csv, stats.json,
provenance.json and normalized.json deterministically. Rates are count weighted;
normalization copies supplied rates with ≥100-valid-call display suppression.
Rolling rates require exactly seven calendar dates. Null is not zero.
Verification includes publication-report daily and last7 values for all seven
identities, exact rolling bounds and 50 dates per identity.
