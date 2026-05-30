#!/usr/bin/env python3
"""Export Slidev slides to PNG and optionally ask fast-agent for visual QA.

Typical usage:

  python3 scripts/visual_review.py --range 3
  python3 scripts/visual_review.py --range 3 --review

The script keeps screenshots under reports/screenshots/ so they can be attached
to chat or reviewed in batch. It intentionally uses Slidev's own PNG exporter
first; use direct Chrome screenshots only when diagnosing dev-only behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "screenshots"
FAST_AGENT_ENV = ROOT / ".fast-agent"
FAST_AGENT_CONFIG = FAST_AGENT_ENV / "fast-agent.yaml"
DEFAULT_MODEL = "codexresponses.gpt-5.5"

PROMPT = """You are reviewing rendered Slidev deck screenshots.

Look for actionable visual defects only. Focus on:
- clipped, cropped, or offscreen content;
- broken custom Vue components;
- unreadable contrast or type size;
- inline code/badge/pill styling that has poor contrast or clashes with the theme;
- awkward alignment, spacing, or visual hierarchy;
- content that is too small, dense, or cramped for a conference presentation;
- negative space that looks accidental rather than intentional;
- reusable diagram components whose containers do not fit their content
  proportionally: section labels that are too small, container padding that is
  too tight, cards with large unused middles, or card height much larger than
  icon/title content requires;
- labeled containers where the section label lacks enough visual weight or
  breathing room to read as structure, instead appearing like a tiny caption
  squeezed into the border;
- icon/title cards where the icon and label feel disconnected instead of
  arranged as one coherent unit;
- Mermaid/code block rendering problems;
- slide content that feels too sparse/dense for presentation use;
- anything that differs from likely author intent.

Do not nitpick taste. Give concise, actionable feedback. Refer to each
screenshot by file name.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", default="slides.md", help="Slidev entry markdown")
    parser.add_argument("--range", dest="page_range", help='Slide range, e.g. "3" or "2-5"')
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Screenshot output directory")
    parser.add_argument("--wait", type=int, default=800, help="Extra wait in ms before export")
    parser.add_argument("--wait-until", default="networkidle", choices=["networkidle", "load", "domcontentloaded", "none"])
    parser.add_argument("--scale", type=float, default=None, help="Slidev PNG export scale")
    parser.add_argument("--executable-path", default=detect_chrome(), help="Browser executable for Slidev export")
    parser.add_argument("--clean", action="store_true", help="Remove old PNGs in --out before exporting")
    parser.add_argument(
        "--fallback-live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If Slidev export is unavailable, start a dev server and screenshot with Chrome",
    )
    parser.add_argument("--port", type=int, default=0, help="Port for --fallback-live screenshots; 0 selects a free port")
    parser.add_argument("--viewport", default="1280,720", help="Chrome screenshot viewport WIDTH,HEIGHT")
    parser.add_argument("--review", action="store_true", help="Run fast-agent vision review over exported PNGs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="fast-agent model for --review")
    parser.add_argument("--name", default="slidev-vision-reviewer", help="fast-agent agent name")
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in out_dir.glob("*.png"):
            old.unlink()

    try:
        exported = export_pngs(args, out_dir)
    except subprocess.CalledProcessError as exc:
        if not args.fallback_live:
            raise
        print(f"Slidev export failed ({exc.returncode}); falling back to live Chrome screenshots.", file=sys.stderr)
        exported = live_chrome_screenshots(args, out_dir)
    if not exported:
        print("No PNG files found after export.", file=sys.stderr)
        return 2

    print("Exported screenshots:")
    for path in exported:
        print(f"  {path.relative_to(ROOT)}")

    if args.review:
        review_with_fast_agent(exported, args.model, args.name)

    return 0


def detect_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    return ""


def export_pngs(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    before = set(out_dir.glob("*.png"))
    cmd = [
        "npx",
        "slidev",
        "export",
        args.entry,
        "--format",
        "png",
        "--output",
        str(out_dir),
        "--wait",
        str(args.wait),
        "--wait-until",
        args.wait_until,
    ]
    if args.page_range:
        cmd.extend(["--range", args.page_range])
    if args.scale is not None:
        cmd.extend(["--scale", str(args.scale)])
    if args.executable_path:
        cmd.extend(["--executable-path", args.executable_path])

    subprocess.run(cmd, cwd=ROOT, check=True)
    after = set(out_dir.glob("*.png"))
    new_files = sorted(after - before)
    return new_files or sorted(after)


def live_chrome_screenshots(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    chrome = args.executable_path or detect_chrome()
    if not chrome:
        raise SystemExit("No Chrome/Chromium executable found for live screenshots.")
    pages = parse_page_range(args.page_range)
    if not pages:
        raise SystemExit("Live screenshot fallback needs --range, e.g. --range 3 or --range 1-5.")

    port = args.port or find_free_port()
    proc = subprocess.Popen(
        [
            "npx",
            "slidev",
            args.entry,
            "--port",
            str(port),
            "--log",
            "silent",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        wait_for_port("127.0.0.1", port, timeout=30)
        if proc.poll() is not None:
            raise SystemExit(f"Slidev server exited early with {proc.returncode}")
        time.sleep(max(0, args.wait) / 1000)
        exported: list[Path] = []
        for page in pages:
            output = out_dir / f"slide-{page:02d}.png"
            url = f"http://127.0.0.1:{port}/{page}"
            cmd = [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                f"--window-size={args.viewport}",
                f"--virtual-time-budget={max(1000, args.wait + 2200)}",
                f"--screenshot={output}",
                url,
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            exported.append(output)
        return exported
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)


def parse_page_range(value: str | None) -> list[int]:
    if not value:
        return []
    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return pages


def wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise SystemExit(f"Timed out waiting for {host}:{port}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def review_with_fast_agent(images: list[Path], model: str, name: str) -> None:
    if not FAST_AGENT_CONFIG.exists():
        raise SystemExit(f"fast-agent config not found: {FAST_AGENT_CONFIG}")

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    prompt_path = reports_dir / "slidev-vision-prompt.txt"
    result_path = reports_dir / "slidev-vision-review.json"
    prompt_path.write_text(
        PROMPT + "\n\nAttached screenshots:\n" + "\n".join(f"- {p.name}" for p in images) + "\n",
        encoding="utf-8",
    )

    schema_path = reports_dir / "slidev-vision-schema.json"
    schema_path.write_text(json.dumps(vision_schema()), encoding="utf-8")

    cmd = [
        "fast-agent",
        "--no-update-check",
        "--env",
        str(FAST_AGENT_ENV),
        "go",
        "--config-path",
        str(FAST_AGENT_CONFIG),
        "--name",
        name,
        "--model",
        model,
        "--quiet",
        "--no-shell",
        "--prompt-file",
        str(prompt_path),
        "--json-schema",
        str(schema_path),
        "--results",
        str(result_path),
    ]
    for image in images:
        cmd.extend(["--attach", str(image)])

    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        raise SystemExit(proc.returncode)

    print("\nVision review:")
    print(proc.stdout.strip())
    print(f"\nSaved result: {result_path.relative_to(ROOT)}")


def vision_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "screenshots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "file": {"type": "string"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "severity": {"type": "string", "enum": ["ok", "note", "warn", "fail"]},
                                    "issue": {"type": "string"},
                                    "recommendation": {"type": "string"},
                                },
                                "required": ["severity", "issue", "recommendation"],
                            },
                        },
                    },
                    "required": ["file", "findings"],
                },
            }
        },
        "required": ["screenshots"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
