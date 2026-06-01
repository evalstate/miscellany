#!/usr/bin/env python3
"""Generate weekly distinct-client mcp-remote share excluding fallback checks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_SOURCE = Path("/home/shaun/source/hf-mcp-stats/data")
DEFAULT_JSON = Path("data-viz/mcp_remote_clients_excluding_fallback_weekly.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--start", default="2025-06-09")
    parser.add_argument("--end", default="2026-05-31")
    return parser.parse_args()


def parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def is_mcp_remote(name: str, version: str) -> bool:
    return "via mcp-remote" in f"{name} {version}".lower()


def is_fallback_check(name: str, version: str) -> bool:
    return "fallback" in f"{name} {version}".lower()


def generate(source: Path, start: date, end: date) -> list[dict[str, object]]:
    weekly = defaultdict(lambda: {"remote_clients": set(), "total_clients": set(), "fallback_clients": set()})

    for month_dir in sorted(source.glob("20*")):
        for day_dir in sorted(month_dir.glob("*")):
            try:
                day = date.fromisoformat(f"{month_dir.name}-{day_dir.name}")
            except ValueError:
                continue
            if day < start or day > end:
                continue
            week = week_start(day)
            for path in sorted(day_dir.glob("transport-metrics-*.json")):
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                if not parse_ts(data.get("currentTime")):
                    continue
                for client in data.get("clients") or []:
                    name = client.get("name") or ""
                    version = client.get("version") or "unknown"
                    key = (name, version)
                    if is_fallback_check(name, version):
                        weekly[week]["fallback_clients"].add(key)
                        continue
                    weekly[week]["total_clients"].add(key)
                    if is_mcp_remote(name, version):
                        weekly[week]["remote_clients"].add(key)

    rows = []
    for week in sorted(weekly):
        values = weekly[week]
        total = len(values["total_clients"])
        remote = len(values["remote_clients"])
        if total == 0:
            continue
        rows.append(
            {
                "week_start": week.isoformat(),
                "week_end": (week + timedelta(days=6)).isoformat(),
                "mcp_remote_client_share_pct": round(100 * remote / total, 4),
                "mcp_remote_clients": remote,
                "total_clients": total,
                "fallback_clients_excluded": len(values["fallback_clients"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    rows = generate(args.source, start, end)
    payload = {
        "schema": "deck.mcp-remote-clients-excluding-fallback-weekly.v1",
        "source": str(args.source),
        "window": {"start": args.start, "end": args.end},
        "filter": "exclude clients whose name or version contains 'fallback'",
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
