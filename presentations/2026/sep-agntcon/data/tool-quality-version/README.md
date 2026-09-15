# Version-focused top ribbons

Open **index.html** directly, offline. Separate iteration; earlier slides untouched.

Two time-aligned ribbons above the chart show Claude Code and MCP Server versions. Each observed version introduction drops a dashed line from its ribbon's bottom edge, at the exact segment start. There are **no floating flags, lollipop heads, or bottom tapes**. Current versions appear in 32px type at the right of their ribbon headers, directly above their matching timelines. Bottom-right version readouts are removed; historical ribbon type increases from 18px to 22px where it fits. The space is used for a larger tool-change panel with 36px headline and 25px detail. Plot stays 1110 × 540.

The line grows downward over 450 logical milliseconds, then settles in contrast over 1.2 seconds. MCP lines are heavier than client lines; rate traces render above them. The other ribbon can visually occlude a line passing behind it. Pause freezes all effects; no event dwells. Reduced motion shows completed strokes without effects.

Ribbon intervals stop at the animated clock and clip to the seven-day viewport. No future versions are shown. Full version names shrink to fit narrow intervals where readable (minimum 10px), otherwise the exact duration remains unlabelled with a tooltip; current versions remain readable in the headers. The six-hour .14/.15 boundary is not widened or merged. The initial .12 ribbon is not a fabricated introduction event. An old segment entering the left edge does not generate a new line.

Four verified tool-description/argument-schema notes appear at observed .13, .15, .18 and .19; .14 is unchanged. .17 guidance is represented at observed .18, not an invented .17 rollout. The latest change persists until superseded.

## Data and freshness

Same verified Aug25–Sep13 disjoint Claude/non-Claude published client cells and exact CSV/JSON bytes. Metric: exclusive tool_quality failed calls / selected calls, not all failures or operations. Fixed five excluded; missing hashes retained; suppressed cells are not zero. Published client-cell coverage ~98–99%. Daily noon UTC anchors, illustrative linear connections; numerical rates use the last revealed observation. Client flags are daily usage leaders, not verified release dates; server flags are first-observed build hours. Edits and error rates are not causally attributed.

Latest compatible local quality publication ends **Sep13**: summaries/telemetry-fixed-five-20260914/series-0-4. Newer local client-protocol-20260915-v1 and protocol-overall-20260915 publications cover Sep14 but do not contain the tool-quality classifications needed here. A separately approved remote telemetry refresh is needed to extend this chart. No remote scan or publication was performed.

Rebuild: `.venv/bin/python scripts/build_tool_quality_version_focus.py`.
Browser check: `NODE_PATH=PATH_TO_PLAYWRIGHT node scripts/check_tool_quality_version_focus.cjs`.
See provenance.json, changes.json, browser_validation.json and screenshots. Private aggregate outputs: review before external sharing.
