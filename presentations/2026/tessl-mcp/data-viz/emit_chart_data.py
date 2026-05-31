#!/usr/bin/env python3
"""Emit small, deck-friendly JSON datasets for simple charts.

This is intentionally lightweight: it turns a CSV into a normalized JSON shape
that Vue/SVG chart components can consume without embedding CSV parsing or data
massaging logic in the presentation layer.

Example:

    python3 data-viz/emit_chart_data.py \
      --input data-viz/mcp_remote_share_weekly_codex_claude_code.csv \
      --output data-viz/codex_mcp_remote_chart.json \
      --where client_family=Codex \
      --x week_start \
      --series field=usage_index_0_100,kind=area,label="Opaque usage index",axis=usage \
      --series field=mcp_remote_share_pct,kind=line,label="mcp-remote share",axis=share \
      --include week_end --include total_requests --include mcp_remote_requests \
      --title Codex
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_key_value(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} expects FIELD=VALUE, got {value!r}")
    key, raw = value.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"{option} has an empty field name")
    return key, raw


def parse_series(value: str) -> dict[str, str]:
    """Parse `field=...,kind=...,label=...,axis=...`.

    Keep this deliberately simple; labels that need commas should be avoided for
    slide labels anyway.
    """

    result: dict[str, str] = {}
    for part in value.split(","):
        key, raw = parse_key_value(part, option="--series")
        result[key] = raw

    if "field" not in result:
        raise argparse.ArgumentTypeError("--series requires field=...")

    result.setdefault("kind", "line")
    result.setdefault("label", result["field"])
    result.setdefault("axis", result["field"])
    return result


def coerce_value(value: str) -> Any:
    if value == "":
        return None
    try:
        if "." not in value:
            return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def filter_rows(rows: list[dict[str, str]], filters: list[tuple[str, str]]) -> list[dict[str, str]]:
    if not filters:
        return rows
    return [row for row in rows if all(row.get(field) == expected for field, expected in filters)]


def build_chart(
    rows: list[dict[str, str]],
    *,
    x_field: str,
    x_type: str,
    x_bucket: str | None,
    series_specs: list[dict[str, str]],
    include_fields: list[str],
    title: str | None,
    subtitle: str | None,
    source: Path,
    filters: list[tuple[str, str]],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row.get(x_field, ""))

    points = []
    for row in rows:
        point: dict[str, Any] = {"x": row.get(x_field)}
        for field in include_fields:
            if field in row:
                point[field] = coerce_value(row[field])
        points.append(point)

    series = []
    for spec in series_specs:
        field = spec["field"]
        series.append(
            {
                "id": spec.get("id", field),
                "label": spec["label"],
                "kind": spec["kind"],
                "axis": spec["axis"],
                "field": field,
                "data": [
                    {
                        "x": row.get(x_field),
                        "y": coerce_value(row.get(field, "")),
                    }
                    for row in rows
                ],
            }
        )

    return {
        "schema": "deck.chart-data.v1",
        "meta": {
            "title": title,
            "subtitle": subtitle,
            "source": str(source),
            "filters": [{"field": field, "value": value} for field, value in filters],
            "row_count": len(rows),
        },
        "x": {
            "field": x_field,
            "type": x_type,
            "bucket": x_bucket,
        },
        "points": points,
        "series": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--x", default="date", help="X-axis field")
    parser.add_argument("--x-type", default="date", help="X-axis type metadata")
    parser.add_argument("--x-bucket", default=None, help="Optional bucket metadata, e.g. day/week/month")
    parser.add_argument("--where", action="append", default=[], help="Filter as FIELD=VALUE; may be repeated")
    parser.add_argument(
        "--series",
        action="append",
        type=parse_series,
        required=True,
        help="Series spec: field=FIELD,kind=line|area|bar,label=LABEL,axis=AXIS",
    )
    parser.add_argument("--include", action="append", default=[], help="Extra point field to include; may be repeated")
    parser.add_argument("--title", default=None)
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    filters = [parse_key_value(value, option="--where") for value in args.where]
    rows = filter_rows(load_rows(args.input), filters)

    chart = build_chart(
        rows,
        x_field=args.x,
        x_type=args.x_type,
        x_bucket=args.x_bucket,
        series_specs=args.series,
        include_fields=args.include,
        title=args.title,
        subtitle=args.subtitle,
        source=args.input,
        filters=filters,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(chart, handle, indent=2 if args.pretty else None)
        handle.write("\n")

    print(f"wrote {args.output} ({chart['meta']['row_count']} rows)")


if __name__ == "__main__":
    main()
