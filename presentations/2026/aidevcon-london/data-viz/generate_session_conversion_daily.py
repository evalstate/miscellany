#!/usr/bin/env python3
"""Emit slide-ready daily session conversion data with a 3-day rolling series."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data-viz/session_conversion_daily_2026-04-07_to_2026-06-01.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data-viz/session_conversion_daily.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    rolling = deque(maxlen=3)

    with args.input.open(newline="") as handle:
        for row in sorted(csv.DictReader(handle), key=lambda item: item["day"]):
            converted = int(row["converted_sessions"])
            rolling.append(converted)
            rows.append(
                {
                    "day": row["day"],
                    "sessions": int(row["sessions"]),
                    "converted_sessions": converted,
                    "unconverted_sessions": int(row["unconverted_sessions"]),
                    "conversion_rate_pct": float(row["conversion_rate_pct"]),
                    "converted_sessions_3d_avg": round(sum(rolling) / len(rolling), 1),
                    "matched_tool_calls": int(row["matched_tool_calls"]),
                }
            )

    payload = {
        "schema": "deck.session-conversion-daily.v1",
        "source": str(args.input),
        "window": {"start": rows[0]["day"], "end": rows[-1]["day"]},
        "rolling_window_days": 3,
        "rows": rows,
    }

    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
