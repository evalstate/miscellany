"""Build data/chess-intro.json for the opening animation (components/ChessAsciiIntro.vue).

Every character grid is copied from an agent's tool output in a TB 2.1
chess-best-move trace, after atif-scan's masking. Nothing is redrawn: the only
things added are layout facts read from each agent's own script (how many
columns, and which part of the 80px square it sampled).

Trials are selected from the private strategies review by harness, model,
step and trace length, so no run IDs or trace paths are stored here or in the
output. The board image is regenerated from the task's public generator and
checked against two traces (see `verify`).

Run:
  uv run --project ~/source/atif-scan --with pillow==11.2.1 \
    python analysis/chess-intro/extract.py --font /path/to/NotoSansSymbols2-Regular.ttf
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import sys
from pathlib import Path

from PIL import Image

from atif_scan.evidence.cite import mask

HERE = Path(__file__).resolve().parent
DECK = HERE.parent.parent
ROWS = Path.home() / ".cache/atif-scan/reviews/talk-hunt-2026-11-08/strategies/asciiart_rows.json"
TASK = Path.home() / "source/terminal-bench-2-1/tasks/chess-best-move"
FEN = "r1bq1r2/1p3pp1/p1n1p3/3nPkbP/8/P1N5/1P2QPP1/R1B1K2R"

# kind "board": the whole image as one grid. kind "squares": labelled crops.
# region = the part of the 80px square the script sampled (x0, y0, x1, y1).
SOURCES = [
    dict(id="ds4f-board", harness="fast-agent", model_name="deepseek-v4-flash", version="0.9.30", step=4, find_step=True,
         model="DeepSeek V4 Flash (max)", kind="board", cols=128, rows=128, ramp=" .:-=+*#%@", dark_is="light"),
    dict(id="glm52-board", harness="fast-agent", model_name="zai-org/GLM-5.2:fireworks-ai", step=6,
         model="GLM-5.2", kind="board", cols=80, rows=40, ramp=" .:-=+*#%@"),
    dict(id="gem31-board", harness="terminus-2", model_name="gemini/gemini-3.1-pro-preview", step=4,
         model="Gemini 3.1 Pro preview", kind="board", cols=80, rows=40, ramp=" .:-=+*#%@"),
    dict(id="ds4f", harness="fast-agent", model_name="deepseek-v4-flash", version="0.9.30", step=22, find_step=True,
         model="DeepSeek V4 Flash (max)", kind="squares", header=r"^--- ([a-h][1-8]) .*---$", cols=40, rows=40, region=[0, 0, 80, 80]),
    dict(id="grok46", harness="nano-grok-build", model_name="grok-4.6", step=11,
         model="Grok 4.6", kind="squares", header=r"^={50} ([a-h][1-8])$", cols=40, region=[0, None, 80, None], trimmed=True),
    dict(id="claude47", harness="terminus-2", model_name="anthropic/claude-opus-4-7", step=19,
         model="Claude Opus 4.7", kind="squares", header=r"^([a-h][1-8]): \(mask_pixels=\d+\)$", cols=40, rows=40, region=[0, 0, 80, 80]),
    dict(id="gpt55", harness="terminus-2", model_name="openai/gpt-5.5", step=6,
         model="GPT-5.5", kind="squares", header=r"^--- ([a-h][1-8]) bg=.*---$", cols=32, rows=32, region=[0, 0, 80, 80]),
    dict(id="gem3", harness="terminus-2", model_name="gemini/gemini-3-pro-preview", step=8,
         model="Gemini 3 Pro preview", kind="squares", header=r"^Piece at ([a-h][1-8]) \(", cols=30, rows=30, region=[10, 10, 70, 70]),
    dict(id="glm51", harness="claude-code", model_name="glm-5.1", step=26,
         model="GLM-5.1", kind="squares", header=r"^=== ([a-h][1-8]) \(col=", cols=60, rows=35, region=[10, 5, 70, 75]),
    dict(id="gpt55-agav", harness="agav", model_name="gpt-5.5", step=8,
         model="GPT-5.5", kind="squares", header=r"^SQUARE ([a-h][1-8])$", cols=24, rows=24, region=[0, 0, 80, 80]),
    dict(id="ds4pro", harness="fast-agent", model_name="deepseek.deepseek-v4-pro?reasoning=max&temperature=1.0&top_p=0.95", step=9,
         model="DeepSeek V4 Pro (max)", kind="squares", header=r"^==== ([a-h][1-8]) \(row", cols=38, rows=38, region=[2, 2, 78, 78]),
    dict(id="grok47", harness="fast-agent", model_name="grok-4.7", step=23,
         model="Grok 4.7", kind="squares", header=r"^=== [bw] ([a-h][1-8]) ===$", cols=40, rows=40, region=[0, 0, 80, 80]),
]

ART = re.compile(r"^[ .:\-=+*#%@oO]*$")
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def fmt_date(iso: str) -> str:
    y, m, d = iso[:10].split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]} {y}"


def load_trial(src: dict, rows: list[dict]) -> tuple[dict, dict]:
    hits = []
    for r in rows:
        if r["task"] != "chess-best-move" or r["harness"] != src["harness"]:
            continue
        if not src.get("find_step") and (r["art_step"] != src["step"] or r["art_lines"] < 12):
            continue
        d = json.loads(Path(r["path"]).read_text())
        a = d.get("agent", {})
        if a.get("model_name") != src["model_name"] or (src.get("version") and a.get("version") != src["version"]):
            continue
        if src.get("find_step") and not (r["reward"] == 1.0 and r["nsteps"] == 35):
            continue  # the featured DeepSeek run already used in the deck (35 steps, reward 1)
        hits.append((r, d))
    if len(hits) != 1:
        sys.exit(f"{src['id']}: expected one trial, found {len(hits)}")
    return hits[0]


def step_text(d: dict, step_id: int) -> tuple[str, str | None]:
    for s in d["steps"]:
        if s.get("step_id") == step_id:
            parts = []
            for res in (s.get("observation") or {}).get("results") or []:
                c = res.get("content")
                parts.append(c if isinstance(c, str) else json.dumps(c))
            return mask("\n".join(parts)), s.get("timestamp")
    sys.exit(f"step {step_id} missing")


def trial_date(r: dict, d: dict, ts: str | None) -> str:
    if ts:
        return fmt_date(ts)
    res = Path(r["path"]).with_name("result.json")
    if res.exists():
        j = json.loads(res.read_text())
        for k in ("started_at", "finished_at"):
            if j.get(k):
                return fmt_date(j[k])
    stamps = re.findall(r"202\d-\d\d-\d\dT", Path(r["path"]).read_text())
    if stamps:
        return fmt_date(min(stamps))
    sys.exit(f"no date for {r['harness']}")


def pad(lines: list[str], cols: int) -> list[str]:
    out = []
    for ln in lines:
        ln = ln.rstrip("\r")
        if len(ln) > cols or not ART.match(ln):
            raise ValueError(f"not art: {ln[:60]!r}")
        out.append(ln.ljust(cols))
    return out


def board_block(text: str, src: dict) -> tuple[list[str], list[int] | None]:
    """The grid, plus [first, last) rows the harness cut from the middle of the output.

    fast-agent caps long tool output and drops the middle. Rows it dropped (or left
    partial) stay blank here: the agent never saw them, so we don't restore them."""
    runs: list[list[str]] = [[]]
    for ln in text.split("\n") + [""]:
        if len(ln.rstrip()) == src["cols"] and ART.match(ln):
            runs[-1].append(ln)
        elif runs[-1]:
            runs.append([])
    runs = [r for r in runs if len(r) >= 8]
    got = sum(map(len, runs))
    if got == src["rows"] and len(runs) == 1:
        return runs[0], None
    if len(runs) == 2 and "Output truncated" in text and got < src["rows"]:
        gap = [len(runs[0]), src["rows"] - len(runs[1])]
        return runs[0] + [" " * src["cols"]] * (gap[1] - gap[0]) + runs[1], gap
    sys.exit(f"{src['id']}: {got} rows in {len(runs)} runs, expected {src['rows']}")


def first_piece_row(img: Image.Image, sq: str) -> int:
    """First sampled row (step 2) of a square holding a non-board pixel."""
    c, r = ord(sq[0]) - 97, 8 - int(sq[1])
    def is_board(p):
        return all(abs(p[i] - v) < 15 for i, v in enumerate((240, 217, 181))) or all(abs(p[i] - v) < 15 for i, v in enumerate((181, 136, 99)))
    for y in range(0, 80, 2):
        if any(not is_board(img.getpixel((c * 80 + x, r * 80 + y))) for x in range(0, 80, 2)):
            return y
    return 0


def square_blocks(text: str, src: dict, img: Image.Image) -> dict[str, dict]:
    head = re.compile(src["header"])
    lines = text.split("\n")
    out: dict[str, dict] = {}
    i = 0
    while i < len(lines):
        m = head.match(lines[i].rstrip())
        i += 1
        if not m:
            continue
        body = []
        while i < len(lines) and not head.match(lines[i].rstrip()):
            body.append(lines[i])
            i += 1
        sq = m.group(1)
        if sq in out:
            continue  # some agents print the same square twice; keep the first
        region = list(src["region"])
        try:
            if src.get("trimmed"):
                while body and not body[-1].strip():
                    body.pop()
                grid = pad(body, src["cols"])
                y0 = first_piece_row(img, sq)
                region[1], region[3] = y0, y0 + 2 * len(grid)
            else:
                grid = pad(body[: src["rows"]], src["cols"])
                if len(grid) != src["rows"]:
                    continue
        except ValueError:
            continue  # terminal scrollback cut this block; skip it, never patch it
        out[sq] = dict(cols=src["cols"], rows=len(grid), region=region, lines=grid)
    return out


def regenerate(font: Path, out: Path) -> Image.Image:
    code = (TASK / "environment/make.py").read_text().replace("/fonts/noto.ttf", str(font))
    tmp = HERE / "_make.py"
    tmp.write_text(code)
    try:
        import os
        cwd = os.getcwd()
        os.chdir(out.parent)
        runpy.run_path(str(tmp), run_name="__main__")
        os.chdir(cwd)
    finally:
        tmp.unlink()
    return Image.open(out).convert("RGB")


def verify(img: Image.Image, boards: dict[str, list[str]]) -> dict[str, str]:
    """Re-run two agents' own renderers on the regenerated PNG; compare with their output."""
    res = {}
    g = img.convert("L")
    chars = " .:-=+*#%@"
    im2 = g.resize((128, 128)); pix = im2.load()
    mine = ["".join(chars[pix[x, y] * len(chars) // 256] for x in range(128)) for y in range(128)]
    seen = [(A, B) for A, B in zip(mine, boards["ds4f-board"]) if B.strip()]
    same = sum(a == b for A, B in seen for a, b in zip(A, B))
    res["ds4f-board"] = f"{same}/{len(seen) * 128} characters identical"
    sm = g.resize((80, 40)); pix = sm.load()
    mine = ["".join(chars[min(len(chars) - 1, pix[x, y] * len(chars) // 256)] for x in range(80)) for y in range(40)]
    same = sum(a == b for A, B in zip(mine, boards["glm52-board"]) for a, b in zip(A, B))
    res["glm52-board"] = f"{same}/{80 * 40} characters identical"
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", type=Path, required=True)
    args = ap.parse_args()

    img = regenerate(args.font, DECK / "public/chess/chess_board.png")
    rows = json.loads(ROWS.read_text())
    sources, boards, squares, gaps = {}, {}, {}, {}
    for src in SOURCES:
        r, d = load_trial(src, rows)
        text, ts = step_text(d, src["step"])
        a = d["agent"]
        sources[src["id"]] = dict(
            harness=a["name"], version=a["version"], model=src["model"], step=src["step"],
            date=trial_date(r, d, ts), reward=int(r["reward"] or 0), ramp=src.get("ramp"),
        )
        if src["kind"] == "board":
            boards[src["id"]], gaps[src["id"]] = board_block(text, src)
        else:
            for sq, block in square_blocks(text, src, img).items():
                squares.setdefault(sq, {})[src["id"]] = block
        print(src["id"], sources[src["id"]], "squares:" if src["kind"] != "board" else "",
              " ".join(sq for sq, v in squares.items() if src["id"] in v))

    check = verify(img, boards)
    print("verify:", check)
    out = dict(
        task="TB 2.1 chess-best-move", fen=FEN, image="chess/chess_board.png", verify=check,
        sources=sources,
        boards={k: dict(cols=len(v[0]), rows=len(v), gap=gaps[k], lines=v) for k, v in boards.items()},
        squares=squares,
    )
    dest = DECK / "data/chess-intro.json"
    dest.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", dest, dest.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
