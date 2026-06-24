#!/usr/bin/env python3
"""Generate weekly mcp-remote share and opaque usage indices by client family.

Reads Hugging Face MCP transport metric snapshots from data/YYYY-MM/DD/*.json.
Counters are cumulative within a server startup session, so this computes positive
per-snapshot deltas keyed by startupTime before aggregating into weekly buckets.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

FAMILIES = {
    "claude-code": "Claude Code",
    "codex-mcp-client": "Codex",
    "opencode": "OpenCode",
}

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHART_OUT = ROOT / "mcp_remote_share_weekly_chart.csv"
FULL_OUT = ROOT / "mcp_remote_share_weekly_codex_claude_code_opencode.csv"


def parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def client_family(name: str) -> str:
    # Normalize e.g. "codex-mcp-client (via mcp-remote 0.1.37)" -> "codex-mcp-client".
    return re.sub(r"\s*\(via mcp-remote [^)]+\)\s*", "", name, flags=re.I).lower()


def is_mcp_remote(name: str, version: str) -> bool:
    return "via mcp-remote" in f"{name} {version}".lower()


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())  # Monday-aligned weeks.


def snapshots():
    paths = sorted(DATA.glob("20*/*/transport-metrics-*.json"))
    rows = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        current_time = parse_ts(data.get("currentTime"))
        if current_time:
            rows.append((current_time, data.get("startupTime") or "", data))
    return sorted(rows)


def generate() -> list[dict[str, object]]:
    previous_by_startup: dict[str, dict[tuple[str, str], int]] = {}
    daily = defaultdict(lambda: defaultdict(lambda: {"remote": 0, "total": 0}))

    for current_time, startup_time, data in snapshots():
        current: dict[tuple[str, str], int] = {}
        for client in data.get("clients") or []:
            key = (client.get("name") or "", client.get("version") or "unknown")
            current[key] = current.get(key, 0) + int(client.get("requestCount") or 0)

        previous = previous_by_startup.get(startup_time)
        if previous is None:
            deltas = current
        else:
            deltas = {key: max(0, value - previous.get(key, 0)) for key, value in current.items()}

        day = current_time.date()
        for (name, version), delta in deltas.items():
            if delta <= 0:
                continue
            family_key = client_family(name)
            if family_key not in FAMILIES:
                continue
            bucket = daily[day][family_key]
            bucket["total"] += delta
            if is_mcp_remote(name, version):
                bucket["remote"] += delta

        previous_by_startup[startup_time] = current

    weekly = defaultdict(lambda: defaultdict(lambda: {"remote": 0, "total": 0}))
    for day, families in daily.items():
        week = week_start(day)
        for family_key, values in families.items():
            weekly[week][family_key]["remote"] += values["remote"]
            weekly[week][family_key]["total"] += values["total"]

    max_total = {
        family_key: max((weekly[week][family_key]["total"] for week in weekly), default=0)
        for family_key in FAMILIES
    }

    rows = []
    for week in sorted(weekly):
        for family_key, label in FAMILIES.items():
            values = weekly[week][family_key]
            total = values["total"]
            if total == 0:
                continue
            remote = values["remote"]
            rows.append(
                {
                    "week_start": week.isoformat(),
                    "week_end": (week + timedelta(days=6)).isoformat(),
                    "client_family": label,
                    "mcp_remote_share_pct": round(100 * remote / total, 4),
                    "usage_index_0_100": round(100 * total / max_total[family_key], 4),
                    "mcp_remote_requests": remote,
                    "total_requests": total,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    rows = generate()
    chart_fields = ["week_start", "week_end", "client_family", "mcp_remote_share_pct", "usage_index_0_100"]
    full_fields = chart_fields + ["mcp_remote_requests", "total_requests"]
    write_csv(CHART_OUT, rows, chart_fields)
    write_csv(FULL_OUT, rows, full_fields)
    print(f"wrote {CHART_OUT} ({sum(1 for _ in CHART_OUT.open()) - 1} rows)")
    print(f"wrote {FULL_OUT} ({sum(1 for _ in FULL_OUT.open()) - 1} rows)")


if __name__ == "__main__":
    main()
