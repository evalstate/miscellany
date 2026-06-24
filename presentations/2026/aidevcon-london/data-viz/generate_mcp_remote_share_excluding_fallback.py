#!/usr/bin/env python3
"""Generate weekly mcp-remote traffic share excluding fallback-check clients."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_SOURCE = Path("/home/shaun/source/hf-mcp-stats/data")
DEFAULT_CSV = Path("data-viz/mcp_remote_share_excluding_fallback_weekly.csv")
DEFAULT_JSON = Path("data-viz/mcp_remote_share_excluding_fallback_weekly.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
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
    text = f"{name} {version}".lower()
    return "fallback" in text


def parse_snapshot(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    current_time = parse_ts(data.get("currentTime"))
    if not current_time:
        return None
    current: dict[tuple[str, str], int] = {}
    for client in data.get("clients") or []:
        key = (client.get("name") or "", client.get("version") or "unknown")
        current[key] = current.get(key, 0) + int(client.get("requestCount") or 0)
    return current_time, data.get("startupTime") or "", current


def snapshot_paths(source: Path, start: date, end: date) -> list[Path]:
    paths = []
    for month_dir in sorted(source.glob("20*")):
        for day_dir in sorted(month_dir.glob("*")):
            try:
                day = date.fromisoformat(f"{month_dir.name}-{day_dir.name}")
            except ValueError:
                continue
            if start <= day <= end:
                paths.extend(sorted(day_dir.glob("transport-metrics-*.json")))
    return paths


def snapshots(source: Path, start: date, end: date):
    paths = snapshot_paths(source, start, end)
    rows = []
    with ProcessPoolExecutor() as pool:
        for row in pool.map(parse_snapshot, paths, chunksize=64):
            if row is not None:
                rows.append(row)
    return sorted(rows)


def generate(source: Path, start: date, end: date) -> list[dict[str, object]]:
    previous_by_startup: dict[str, dict[tuple[str, str], int]] = {}
    daily = defaultdict(lambda: {"remote": 0, "total": 0, "fallback_excluded": 0})

    for current_time, startup_time, current in snapshots(source, start, end):
        previous = previous_by_startup.get(startup_time)
        if previous is None:
            deltas = current
        else:
            deltas = {key: max(0, value - previous.get(key, 0)) for key, value in current.items()}

        bucket = daily[current_time.date()]
        for (name, version), delta in deltas.items():
            if delta <= 0:
                continue
            if is_fallback_check(name, version):
                bucket["fallback_excluded"] += delta
                continue
            bucket["total"] += delta
            if is_mcp_remote(name, version):
                bucket["remote"] += delta

        previous_by_startup[startup_time] = current

    weekly = defaultdict(lambda: {"remote": 0, "total": 0, "fallback_excluded": 0})
    for day, values in daily.items():
        bucket = weekly[week_start(day)]
        for key, value in values.items():
            bucket[key] += value

    max_total = max((values["total"] for values in weekly.values()), default=1)
    rows = []
    for week in sorted(weekly):
        values = weekly[week]
        total = values["total"]
        if total <= 0:
            continue
        remote = values["remote"]
        rows.append(
            {
                "week_start": week.isoformat(),
                "week_end": (week + timedelta(days=6)).isoformat(),
                "mcp_remote_share_pct": round(100 * remote / total, 4),
                "usage_index_0_100": round(100 * total / max_total, 4),
                "mcp_remote_requests": remote,
                "total_requests": total,
                "fallback_excluded_requests": values["fallback_excluded"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "week_start",
        "week_end",
        "mcp_remote_share_pct",
        "usage_index_0_100",
        "mcp_remote_requests",
        "total_requests",
        "fallback_excluded_requests",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    rows = generate(args.source, start, end)
    write_csv(args.csv, rows)
    payload = {
        "schema": "deck.mcp-remote-share-excluding-fallback.v1",
        "source": str(args.source),
        "window": {"start": args.start, "end": args.end},
        "filter": "exclude clients whose name or version contains 'fallback'",
        "rows": rows,
    }
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.csv} ({len(rows)} rows)")
    print(f"wrote {args.json} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
