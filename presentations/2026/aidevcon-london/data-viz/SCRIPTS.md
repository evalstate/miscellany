# Data-viz scripts and generated artifacts

This directory contains presentation-specific data transforms and small derived
datasets used by the Slidev deck.

## Authoritative source data

Primary source repository:

```text
/home/shaun/source/hf-mcp-stats
```

Treat that repository as read-only unless explicitly instructed otherwise. Keep
large raw inputs there; write only slide-ready summaries or generated chart data
in this deck's `data-viz/` directory.

## Maintenance rule

When adding, renaming, deleting, or materially changing a script or generated
dataset in this directory, update this catalog in the same change.

## Scripts

### `emit_chart_data.py`

Generic CSV-to-chart-JSON emitter for simple deck charts.

- Preferred language: Python
- Inputs: any small/medium CSV with an x-axis column and one or more numeric
  series columns.
- Output: JSON using schema `deck.chart-data.v1`.
- Purpose:
  - keep CSV parsing and filtering out of Vue slide components;
  - create a consistent, deck-friendly data shape for line, area, and bar
    charts;
  - support repeated `--where`, `--series`, and `--include` flags.
- Example:

  ```bash
  python3 data-viz/emit_chart_data.py \
    --input data-viz/mcp_remote_share_weekly_codex_claude_code.csv \
    --output data-viz/codex_mcp_remote_chart.json \
    --where client_family=Codex \
    --x week_start \
    --x-bucket week \
    --series field=usage_index_0_100,kind=area,label="Opaque usage index",axis=usage \
    --series field=mcp_remote_share_pct,kind=line,label="mcp-remote share",axis=share \
    --include week_end \
    --include total_requests \
    --include mcp_remote_requests \
    --title Codex \
    --pretty
  ```

### `generate_mcp_remote_weekly.py`

Generates weekly `mcp-remote` share and opaque usage-index metrics by client
family.

- Preferred language: Python
- Inputs: Hugging Face MCP transport metric snapshots under the source stats
  repository's dated `data/YYYY-MM/DD/*.json` tree, when present.
- Outputs:
  - `mcp_remote_share_weekly_chart.csv`
  - `mcp_remote_share_weekly_codex_claude_code.csv`
  - `mcp_remote_share_weekly_codex_claude_code_opencode.csv`
- Notes:
  - Counters are cumulative within a server startup session.
  - The script computes positive per-snapshot deltas keyed by `startupTime`
    before aggregating into weekly buckets.
  - `usage_index_0_100` is intentionally opaque and presentation-oriented.


### `generate_mcp_remote_clients_excluding_fallback.py`

Generates weekly distinct-client `mcp-remote` share excluding fallback-check
clients.

- Preferred language: Python
- Inputs: Hugging Face MCP transport metric snapshots under:
  - `/home/shaun/source/hf-mcp-stats/data`
- Output:
  - `mcp_remote_clients_excluding_fallback_weekly.json`
- Notes:
  - Defaults to the same 2025-06-09 through 2026-05-31 range as `mcp_weekly_init_tool_calls.json`.
  - Counts distinct `(client.name, client.version)` identities seen per week.
  - Excludes clients whose name or version contains `fallback`.

### `generate_mcp_remote_share_excluding_fallback.py`

Generates weekly `mcp-remote` request-share metrics across all client traffic,
excluding fallback-check clients.

- Preferred language: Python
- Inputs: Hugging Face MCP transport metric snapshots under:
  - `/home/shaun/source/hf-mcp-stats/data`
- Outputs:
  - `mcp_remote_share_excluding_fallback_weekly.csv`
  - `mcp_remote_share_excluding_fallback_weekly.json`
- Notes:
  - Defaults to the latest local six-week window used for protocol-efficiency
    analysis: 2026-04-19 through 2026-05-31.
  - Counters are cumulative within a server startup session.
  - The script computes positive per-snapshot deltas keyed by `startupTime`.
  - Excludes clients whose name or version contains `fallback`, including
    `mcp-remote-fallback-test`.
  - `mcp_remote_share_pct` is request share, not distinct-user or
    distinct-install share.

### `generate_mcp_weekly_activity.py`

Generates weekly MCP initialization-request counts and summed tool-call counts.

- Preferred language: Python
- Inputs: Hugging Face MCP transport metric snapshots under the source stats
  repository's dated `data/YYYY-MM/DD/*.json` tree.
- Output:
  - `mcp_weekly_init_tool_calls.json`
- Notes:
  - Method counters are cumulative within a server startup session.
  - The script computes positive per-snapshot deltas keyed by `startupTime` and
    method name before aggregating into Monday-aligned weekly buckets.
  - `init_requests` is the delta of `methods[].count` where
    `method == "initialize"`.
  - `tool_calls` is the sum of deltas for methods whose name starts with
    `tools/call`.
  - First and latest weeks are marked `partial_week`; the first week can include
    pre-series counter activity from a server that started before collection.

### `generate_session_conversion_daily.py`

Generates slide-ready daily session-conversion JSON from the refreshed
last-8-weeks conversion-analysis CSV.

- Preferred language: Python
- Input:
  - `session_conversion_daily_2026-04-07_to_2026-06-01.csv`
- Output:
  - `session_conversion_daily.json`
- Notes:
  - Session creation is the first `initialize` row per `clientSessionId` in the
    upstream analysis.
  - Conversion means at least one joined query row for the same
    `clientSessionId`; query success is not required.
  - `converted_sessions_3d_avg` is a trailing 3-day rolling average.

## Generated datasets

### `mcp_remote_share_weekly.json`

Small JSON dataset generated from
`mcp_remote_share_weekly_codex_claude_code.csv` for the Slidev Vue chart
component `McpRemoteTrafficChart.vue`.

Contains weekly rows for:

- `Claude Code`
- `Codex`

Fields:

- `week_start`
- `week_end`
- `client_family`
- `mcp_remote_share_pct`
- `usage_index_0_100`
- `mcp_remote_requests`
- `total_requests`

### `mcp_remote_share_weekly_codex_claude_code.csv`

Weekly CSV used as the source for the deck-local JSON chart artifact. Contains
Codex and Claude Code only.

### `mcp_remote_share_weekly_codex_claude_code_opencode.csv`

Weekly CSV variant including OpenCode.

### `mcp_remote_share_weekly_chart.csv`

Compact weekly CSV with chart-facing columns.

### `mcp_remote_share_excluding_fallback_weekly.csv`

Weekly CSV for all-client `mcp-remote` request share after excluding fallback
checks.

### `mcp_remote_share_excluding_fallback_weekly.json`

Small JSON dataset for `McpRemoteNoFallbackChart.vue`.

Fields:

- `week_start`
- `week_end`
- `mcp_remote_share_pct`
- `usage_index_0_100`
- `mcp_remote_requests`
- `total_requests`
- `fallback_excluded_requests`

### `mcp_weekly_init_tool_calls.json`

Small JSON dataset for `McpWeeklyActivityChart.vue`.

Contains one row per Monday-aligned week from the first available transport
metrics snapshot through the latest available snapshot.

Fields:

- `week_start`
- `week_end`
- `iso_week`
- `init_requests`
- `tool_calls`
- `snapshot_count`
- `partial_week`

### Session conversion analysis files

Compact copies of the refreshed conversion-analysis outputs from:

```text
/home/shaun/temp/hf-mcp-logs/outputs/last8w_2026-06-01
```

Copied files:

- `session_conversion_daily_2026-04-07_to_2026-06-01.csv`
- `client_session_conversion_2026-04-07_to_2026-06-01.csv`
- `client_version_session_conversion_2026-04-07_to_2026-06-01.csv`
- `session_conversion_analysis_2026-04-07_to_2026-06-01.json`

Large parquet extracts are intentionally kept outside the deck.

### `session_conversion_daily.json`

Small JSON dataset for `SessionConversionChart.vue`.

Contains one row per actual event date from 2026-04-07 through 2026-05-31.

Fields:

- `day`
- `sessions`
- `converted_sessions`
- `unconverted_sessions`
- `conversion_rate_pct`
- `converted_sessions_3d_avg`
- `matched_tool_calls`

## Subagent

Use the project-local fast-agent card below for data questions and script
maintenance:

```text
.fast-agent/agent-cards/data-viz.md
```

The subagent is responsible for querying `/home/shaun/source/hf-mcp-stats`,
creating repeatable Python analyses, and keeping this catalog current.
