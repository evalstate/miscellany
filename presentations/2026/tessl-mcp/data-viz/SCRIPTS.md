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

## Subagent

Use the project-local fast-agent card below for data questions and script
maintenance:

```text
.fast-agent/agent-cards/data-viz.md
```

The subagent is responsible for querying `/home/shaun/source/hf-mcp-stats`,
creating repeatable Python analyses, and keeping this catalog current.
