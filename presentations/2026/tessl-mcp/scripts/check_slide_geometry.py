#!/usr/bin/env python3
"""Deterministic geometry audit for rendered Slidev slides.

This is a lightweight, Birch-inspired smoke checker for layout failures that
vision review can miss:

- text or SVG labels outside the viewport,
- text/SVG labels clipped by an overflow-hidden ancestor,
- chart/diagram containers whose scroll size exceeds their client size,
- whole-page scroll overflow at the slide viewport.

It intentionally does not judge taste. It reports measurable geometry symptoms.

Examples:

  python3 scripts/check_slide_geometry.py --range 13-15
  python3 scripts/check_slide_geometry.py --range 13-15 --fail-on findings
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:  # pragma: no cover - environment/setup error
    raise SystemExit(
        "scripts/check_slide_geometry.py requires the Python package 'websockets'. "
        "Install it with: python3 -m pip install websockets"
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWPORT = "1280,720"


AUDIT_JS = r"""
(() => {
  const tolerance = 2;
  const maxFindings = 160;

  function roundRect(r) {
    return {
      x: Math.round(r.left),
      y: Math.round(r.top),
      width: Math.round(r.width),
      height: Math.round(r.height),
      left: Math.round(r.left),
      top: Math.round(r.top),
      right: Math.round(r.right),
      bottom: Math.round(r.bottom),
    };
  }

  function path(el) {
    const parts = [];
    while (el && el.nodeType === 1 && parts.length < 6) {
      let part = el.tagName.toLowerCase();
      if (el.id) part += "#" + el.id;
      const cls = typeof el.className === "string"
        ? el.className.trim().split(/\s+/).filter(Boolean).slice(0, 4)
        : [];
      if (cls.length) part += "." + cls.join(".");
      parts.unshift(part);
      el = el.parentElement;
    }
    return parts.join(" > ");
  }

  function text(el) {
    return (el.textContent || "").replace(/\s+/g, " ").trim().slice(0, 140);
  }

  function visible(el) {
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden" || Number(cs.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0.5 && r.height > 0.5;
  }

  function clips(cs) {
    return /(hidden|clip|auto|scroll)/.test(cs.overflow + " " + cs.overflowX + " " + cs.overflowY);
  }

  function clippingAncestor(el) {
    let p = el.parentElement;
    while (p && p !== document.documentElement) {
      const cs = getComputedStyle(p);
      if (clips(cs)) return p;
      p = p.parentElement;
    }
    return null;
  }

  function overlaps(a, b) {
    return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
  }

  function push(findings, finding) {
    if (findings.length < maxFindings) findings.push(finding);
  }

  const viewport = {
    left: 0,
    top: 0,
    right: window.innerWidth,
    bottom: window.innerHeight,
    width: window.innerWidth,
    height: window.innerHeight,
  };

  const findings = [];
  const selector = [
    "h1", "h2", "h3", "p", "li", "code", "pre", "button",
    ".kicker", ".deck-panel",
    ".traffic-chart", ".traffic-chart-slide", ".traffic-chart__badge",
    ".traffic-chart__axis-title", ".traffic-chart__usage-label",
    ".traffic-chart__x-axis text", ".traffic-chart__y-axis text", ".traffic-chart__latest text",
    ".remote-mcp__node", ".remote-mcp__legend",
    ".spec-timeline__tick", ".spec-timeline__bar", ".spec-timeline__lane-label",
    "svg text"
  ].join(",");

  const elements = Array.from(document.querySelectorAll(selector)).filter(visible);
  const seen = new Set();

  elements.forEach((el) => {
    if (seen.has(el)) return;
    seen.add(el);

    const r = el.getBoundingClientRect();
    const off = {
      left: Math.max(0, viewport.left - r.left),
      right: Math.max(0, r.right - viewport.right),
      top: Math.max(0, viewport.top - r.top),
      bottom: Math.max(0, r.bottom - viewport.bottom),
    };

    if (off.left > tolerance || off.right > tolerance || off.top > tolerance || off.bottom > tolerance) {
      push(findings, {
        kind: "viewport_overflow",
        selector: path(el),
        text: text(el),
        rect: roundRect(r),
        overflow_px: {
          left: Math.round(off.left),
          right: Math.round(off.right),
          top: Math.round(off.top),
          bottom: Math.round(off.bottom),
        },
      });
    }

    const ancestor = clippingAncestor(el);
    if (ancestor) {
      const cr = ancestor.getBoundingClientRect();
      if (overlaps(r, viewport) && cr.width > 0 && cr.height > 0) {
        const clipped = {
          left: Math.max(0, cr.left - r.left),
          right: Math.max(0, r.right - cr.right),
          top: Math.max(0, cr.top - r.top),
          bottom: Math.max(0, r.bottom - cr.bottom),
        };
        if (clipped.left > tolerance || clipped.right > tolerance || clipped.top > tolerance || clipped.bottom > tolerance) {
          push(findings, {
            kind: "clipping_ancestor_overflow",
            selector: path(el),
            container: path(ancestor),
            text: text(el),
            rect: roundRect(r),
            containerRect: roundRect(cr),
            overflow_px: {
              left: Math.round(clipped.left),
              right: Math.round(clipped.right),
              top: Math.round(clipped.top),
              bottom: Math.round(clipped.bottom),
            },
          });
        }
      }
    }
  });

  const containerSelector = [
    ".slidev-layout",
    ".traffic-chart-slide",
    ".traffic-chart",
    ".remote-mcp",
    ".spec-timeline",
    ".protocol-stack",
    ".mermaid"
  ].join(",");

  Array.from(document.querySelectorAll(containerSelector)).filter(visible).forEach((el) => {
    const overX = el.scrollWidth - el.clientWidth;
    const overY = el.scrollHeight - el.clientHeight;
    if (overX > tolerance || overY > tolerance) {
      push(findings, {
        kind: "container_scroll_overflow",
        selector: path(el),
        text: text(el),
        rect: roundRect(el.getBoundingClientRect()),
        scrollWidth: el.scrollWidth,
        clientWidth: el.clientWidth,
        scrollHeight: el.scrollHeight,
        clientHeight: el.clientHeight,
        overflow_px: {
          right: Math.round(Math.max(0, overX)),
          bottom: Math.round(Math.max(0, overY)),
        },
      });
    }
  });

  const pageOverflowX = Math.round(document.documentElement.scrollWidth - document.documentElement.clientWidth);
  const pageOverflowY = Math.round(document.documentElement.scrollHeight - document.documentElement.clientHeight);
  if (pageOverflowX > tolerance || pageOverflowY > tolerance) {
    push(findings, {
      kind: "page_scroll_overflow",
      selector: "document.documentElement",
      overflow_px: {
        right: Math.max(0, pageOverflowX),
        bottom: Math.max(0, pageOverflowY),
      },
      scrollWidth: document.documentElement.scrollWidth,
      clientWidth: document.documentElement.clientWidth,
      scrollHeight: document.documentElement.scrollHeight,
      clientHeight: document.documentElement.clientHeight,
    });
  }

  return {
    url: location.href,
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio,
    },
    finding_count: findings.length,
    findings,
  };
})()
"""


@dataclass
class ChromeProcess:
    proc: subprocess.Popen[str]
    user_data_dir: tempfile.TemporaryDirectory[str]
    debugging_port: int


class CDPClient:
    def __init__(self, websocket_url: str) -> None:
        self.websocket_url = websocket_url
        self.next_id = 0
        self.ws: Any = None

    async def __aenter__(self) -> "CDPClient":
        self.ws = await websockets.connect(self.websocket_url, max_size=16_000_000)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.ws.close()

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        msg_id = self.next_id
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = await self.ws.recv()
            message = json.loads(raw)
            if message.get("id") != msg_id:
                continue
            if "error" in message:
                raise RuntimeError(f"CDP {method} failed: {message['error']}")
            return message.get("result", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", default="slides.md")
    parser.add_argument("--range", dest="page_range", required=True, help='Slide range, e.g. "3" or "2-5"')
    parser.add_argument("--viewport", default=DEFAULT_VIEWPORT, help="Viewport WIDTH,HEIGHT")
    parser.add_argument("--port", type=int, default=0, help="Slidev dev server port; 0 selects a free port")
    parser.add_argument("--chrome-port", type=int, default=0, help="Chrome remote debugging port; 0 selects a free port")
    parser.add_argument("--chrome", default=detect_chrome(), help="Chrome/Chromium executable")
    parser.add_argument("--wait", type=float, default=1.2, help="Seconds to wait after navigation before auditing")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "slide-geometry.json")
    parser.add_argument("--fail-on", choices=["never", "findings"], default="never")
    args = parser.parse_args()

    pages = parse_page_range(args.page_range)
    if not pages:
        raise SystemExit("--range did not resolve to any slide numbers")
    if not args.chrome:
        raise SystemExit("No Chrome/Chromium executable found")

    slidev_port = args.port or find_free_port()
    chrome_port = args.chrome_port or find_free_port()
    width, height = parse_viewport(args.viewport)

    slidev_proc = start_slidev(args.entry, slidev_port)
    chrome: ChromeProcess | None = None
    try:
        wait_for_port("127.0.0.1", slidev_port, timeout=35)
        chrome = start_chrome(args.chrome, chrome_port, width, height)
        wait_for_port("127.0.0.1", chrome_port, timeout=20)
        results = asyncio.run(run_audits(chrome_port, slidev_port, pages, args.wait))
    finally:
        terminate_process(slidev_proc)
        if chrome is not None:
            terminate_process(chrome.proc)
            chrome.user_data_dir.cleanup()

    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"slides": results}, indent=2) + "\n", encoding="utf-8")

    total = sum(item["finding_count"] for item in results)
    print(f"Wrote {args.out.relative_to(ROOT)}")
    if total:
        print(f"Geometry findings: {total}")
        for item in results:
            if item["finding_count"]:
                print(f"  slide {item['page']}: {item['finding_count']}")
                for finding in item["findings"][:6]:
                    text = f" — {finding.get('text')}" if finding.get("text") else ""
                    print(f"    {finding['kind']}: {finding.get('selector')}{text}")
    else:
        print("Geometry findings: 0")

    if args.fail_on == "findings" and total:
        return 1
    return 0


def detect_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    return ""


def parse_page_range(value: str) -> list[int]:
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


def parse_viewport(value: str) -> tuple[int, int]:
    width_s, height_s = value.split(",", 1)
    return int(width_s), int(height_s)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise SystemExit(f"Timed out waiting for {host}:{port}")


def start_slidev(entry: str, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["npx", "slidev", entry, "--port", str(port), "--log", "silent"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )


def start_chrome(chrome: str, port: int, width: int, height: int) -> ChromeProcess:
    user_data_dir = tempfile.TemporaryDirectory(prefix="slidev-geometry-chrome-")
    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir.name}",
            f"--window-size={width},{height}",
            "about:blank",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    return ChromeProcess(proc=proc, user_data_dir=user_data_dir, debugging_port=port)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def json_get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def json_put(url: str) -> Any:
    request = urllib.request.Request(url, method="PUT")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


async def run_audits(chrome_port: int, slidev_port: int, pages: list[int], wait: float) -> list[dict[str, Any]]:
    target = json_put(f"http://127.0.0.1:{chrome_port}/json/new?about:blank")
    websocket_url = target["webSocketDebuggerUrl"]
    results: list[dict[str, Any]] = []
    async with CDPClient(websocket_url) as cdp:
        await cdp.call("Page.enable")
        await cdp.call("Runtime.enable")
        for page in pages:
            url = f"http://127.0.0.1:{slidev_port}/{page}"
            await cdp.call("Page.navigate", {"url": url})
            await asyncio.sleep(wait)
            result = await cdp.call(
                "Runtime.evaluate",
                {
                    "expression": AUDIT_JS,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
            value = result.get("result", {}).get("value", {})
            value["page"] = page
            results.append(value)
    return results


if __name__ == "__main__":
    raise SystemExit(main())
