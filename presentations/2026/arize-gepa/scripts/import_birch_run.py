#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a simple-gepa Birch run into the Slidev deck.")
    parser.add_argument("run_dir", type=Path, help="Path to runs/birch/<run-name>")
    parser.add_argument("--asset-dir", default="gepa-birch", help="Directory under public/ for screenshots")
    parser.add_argument("--clean", action="store_true", help="Remove existing imported screenshots first")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not (run_dir / "candidates").exists():
        raise SystemExit(f"not a Birch run directory: {run_dir}")

    asset_root = ROOT / "public" / args.asset_dir
    if args.clean and asset_root.exists():
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)

    items = []
    for score_path in sorted((run_dir / "candidates").glob("candidate-*/score.json")):
        candidate_dir = score_path.parent
        candidate_id = candidate_dir.name
        candidate = _read_json(candidate_dir / "candidate.json")
        score = _read_json(score_path)
        side = score.get("side_info") or score
        scores = side.get("scores") or {}
        summary = side.get("summary") or {}
        turns = [
            turn
            for generation in side.get("generation", [])
            for turn in ((generation.get("usage") or {}).get("turns") or [])
        ]

        image_name = f"{candidate_id}-deep.png"
        screenshot = candidate_dir / "reports" / "screenshots" / "numeric-data-deep.png"
        if screenshot.exists():
            shutil.copy2(screenshot, asset_root / image_name)

        items.append({
            "id": candidate_id,
            "iteration": int(candidate_id.split("-")[-1]),
            "image": f"{args.asset_dir}/{image_name}",
            "score": scores.get("gepa_score", 0),
            "generation": scores.get("generation", 0),
            "checker": scores.get("checker", 0),
            "hygiene": scores.get("hygiene", 0),
            "skillLengthScore": scores.get("skill_length_score", 0),
            "skillLines": summary.get("skill_lines") or _line_count(candidate.get("SKILL.md")),
            "recipeLines": _line_count(candidate.get("recipes/numeric-data.md")),
            "toolCalls": sum(int(turn.get("tool_calls") or 0) for turn in turns),
            "turns": len(turns),
            "totalTokens": sum(int(turn.get("total_tokens") or 0) for turn in turns),
            "checkerFailures": summary.get("checker_failures", 0),
            "checkerWarnings": summary.get("checker_warnings", 0),
            "missingCssArtifacts": summary.get("missing_birch_css_artifacts", 0),
            "scoreCap": summary.get("score_cap", 1),
        })

    data = {"run": run_dir.name, "source": str(run_dir), "items": items}
    (ROOT / "components" / "gepaBirchRun.ts").write_text(
        "export const gepaBirchRun = " + json.dumps(data, indent=2) + " as const\n",
        encoding="utf-8",
    )
    (asset_root / "run.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(items)} candidates from {run_dir.name}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_count(text: str | None) -> int:
    return len((text or "").splitlines())


if __name__ == "__main__":
    main()
