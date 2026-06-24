#!/usr/bin/env python3
"""Generate weekly MCP initialize and tools/call activity.

Reads Hugging Face MCP transport metric snapshots from the source stats repo.
Method counters are cumulative within a server startup session, so this computes
positive per-snapshot deltas keyed by startupTime + method before aggregating
into Monday-aligned weekly buckets.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path("/home/shaun/source/hf-mcp-stats")
DATA = SOURCE / "data"
OUT = ROOT / "mcp_weekly_init_tool_calls.json"
CURRENT_RE = re.compile(r'"currentTime"\s*:\s*"([^"]+)"')
STARTUP_RE = re.compile(r'"startupTime"\s*:\s*"([^"]*)"')
METHOD_RE = re.compile(r'"method"\s*:\s*"(initialize|tools/call[^"]*)"\s*,\s*"count"\s*:\s*(\d+)')


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def snapshots() -> list[tuple[datetime, str, dict[str, int]]]:
    rows = []
    for path in sorted(DATA.glob("20*/*/transport-metrics-*.json")):
        try:
            text = path.read_text()
        except Exception:
            continue
        current_match = CURRENT_RE.search(text)
        if not current_match or '"methods"' not in text:
            continue
        current_time = parse_ts(current_match.group(1))
        startup_match = STARTUP_RE.search(text)
        methods: dict[str, int] = {}
        for name, count in METHOD_RE.findall(text):
            methods[name] = methods.get(name, 0) + int(count)
        if current_time and methods:
            rows.append((current_time, startup_match.group(1) if startup_match else "", methods))
    return sorted(rows)


def generate() -> dict:
    previous_by_startup: dict[str, dict[str, int]] = {}
    weekly = defaultdict(lambda: {"init_requests": 0, "tool_calls": 0, "snapshot_count": 0})
    first_seen: date | None = None
    last_seen: date | None = None

    for current_time, startup_time, current in snapshots():
        first_seen = current_time.date() if first_seen is None else min(first_seen, current_time.date())
        last_seen = current_time.date() if last_seen is None else max(last_seen, current_time.date())

        previous = previous_by_startup.get(startup_time)
        if previous is None:
            deltas = current
        else:
            deltas = {name: max(0, count - previous.get(name, 0)) for name, count in current.items()}

        bucket = weekly[week_start(current_time.date())]
        bucket["snapshot_count"] += 1
        bucket["init_requests"] += deltas.get("initialize", 0)
        bucket["tool_calls"] += sum(count for name, count in deltas.items() if name.startswith("tools/call"))

        previous_by_startup[startup_time] = current

    if first_seen is None or last_seen is None:
        rows = []
    else:
        rows = []
        for week in sorted(weekly):
            rows.append(
                {
                    "week_start": week.isoformat(),
                    "week_end": (week + timedelta(days=6)).isoformat(),
                    "iso_week": f"{week.isocalendar().year}-W{week.isocalendar().week:02d}",
                    "init_requests": weekly[week]["init_requests"],
                    "tool_calls": weekly[week]["tool_calls"],
                    "snapshot_count": weekly[week]["snapshot_count"],
                    "partial_week": week <= first_seen <= week + timedelta(days=6)
                    or week <= last_seen <= week + timedelta(days=6),
                }
            )

    return {
        "schema": "deck.chart-data.v1",
        "title": "Weekly MCP initialization requests and tool calls",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo": str(SOURCE),
            "files": "data/YYYY-MM/DD/transport-metrics-*.json",
            "fields": ["currentTime", "startupTime", "methods[].method", "methods[].count"],
        },
        "definitions": {
            "week_start": "Monday-aligned week in UTC, assigned from snapshot currentTime.",
            "init_requests": "Delta of methods[].count where method == 'initialize'.",
            "tool_calls": "Sum of deltas for methods whose name starts with 'tools/call'.",
            "restart_handling": "Counters are differenced within each startupTime + method stream.",
        },
        "series": [
            {"key": "init_requests", "label": "Initializations", "kind": "bar", "axis": "init"},
            {"key": "tool_calls", "label": "Tool calls", "kind": "line", "axis": "tools"},
        ],
        "rows": rows,
    }


def main() -> None:
    data = generate()
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT} ({len(data['rows'])} rows)")


if __name__ == "__main__":
    main()
