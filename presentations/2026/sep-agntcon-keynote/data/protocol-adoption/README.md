# Protocol Adoption · 2026-07-28

Open index.html directly with file://; no server, network or autoplay. Private aggregates: review before public sharing.

Static snapshot, not live. Snapshot cutoff is source snapshot publication time, not a guarantee that all events were ingested. Logged tool calls, not protocol messages or users. Fixed five excluded across all clients/versions; missing hashes retained. Unknown/unreviewed protocols excluded from the denominator.

Source: evalstate/hf-mcp-logs @ 1b5b83fbc3b4be2870345edf498b2644167caf18; window 2026-07-28–2026-09-15 UTC.
Partial day: 2026-09-15. Publication cutoff: 2026-09-15T16:06:46+00:00. Revision discovered: 2026-09-15T16:12:21.881654+00:00.

Purple is selected-day tool-call share; slate is strict count-weighted rolling7. The source is the queries tool-operation log, not protocol-message counters. Reducer counts query rows excluding initialize/session_delete, failures included, no deduplication. The five hashes are removed across all identities before aggregation. Only recognized protocol versions enter the rate denominator; unknown/unreviewed versions remain in count tables. Ratios are 0..1 in JSON/CSV, percentages on stage. No sample threshold. Null is unavailable, not zero. A partial day uses the preceding seven completed calendar dates. The plot viewport selector changes only the visible dates, never the rolling metric. Straight daily connections are visual guides, not intraday estimates; playback progressively reveals straight connections as illustration only; numerical rates are unchanged. KPIs/date use floor(position), the last actually reached daily anchor. No connections bridge nulls or calendar gaps; no future point markers appear. Amber hollow point/dotted connection marks actual partial data.

Overall includes every retained client, including Other/unknown identities. Focus labels preserve exact identities: Hugging Face Chat UI = chat-ui-mcp (not chat-ui-intern); Codex CLI = codex-mcp-client (not openai-mcp (Codex)). No version/event evidence is provided or invented.

Schema v1: data.json has protocol, dates, series (id, label, rows), provenance. Each row has date, daily_share, rolling_share, rolling_start, rolling_end, partial, counts (all_calls, valid_calls, modern_calls, unknown_or_unreviewed_calls, rolling_valid_calls, rolling_modern_calls). daily.csv flattens rows with client identities. Provenance includes coverage, counting, cohort policy, input/script hashes and snapshot metadata.

Controls: native keyboard-accessible buttons/selectors; scrubber arrow/Home/End keys. Space on page toggles playback; arrows on page seek. Reset pauses at first day; default is latest day paused. Playback uses one requestAnimationFrame clock, 30 seconds for the full series at 1×, with a continuously translating sliding viewport. Pause freezes the exact fractional position; resume does not jump. Hidden pages pause. Reduced-motion preference never triggers autoplay; explicit play steps daily without smooth motion. No independent CSS animation clock is used. SVG export captures current stage. API: window.protocolAdoption.seek(index) (pauses, accepts fractional day indices), play(), pause(), state, data. state is a read-only snapshot with index (floor), position (fractional), viewportStart/viewportEnd (day indices), duration (30000 ms at 1×), playing, client, viewport, speed, reducedMotion. Scrubber step is .001 day; play from the end replays.

Rebuild from repository root:
```
.venv/bin/python scripts/build_protocol_adoption_dashboard.py --source summaries/client-protocol-20260915-current-v1 --output charts/protocol-adoption-NEW
```
