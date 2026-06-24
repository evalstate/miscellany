#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLISH = ROOT / "publish"
MANIFEST = PUBLISH / "manifest.json"
SITE = PUBLISH / "site"
CURRENT_SPACE_INFO = PUBLISH / "current-space-info.json"

EXCLUDE_NAMES = {
    ".git",
    ".fast-agent",
    "node_modules",
    ".venv",
    "reports",
    "eval-cards",
    "fastagent.secrets.yaml",
}



def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_root() -> Path:
    root = git_output("rev-parse", "--show-toplevel")
    return Path(root).resolve() if root else ROOT.resolve()


def git_status_paths() -> set[str]:
    out = git_output("status", "--porcelain", "-z")
    if not out:
        return set()
    paths: set[str] = set()
    parts = out.split("\0")
    for part in parts:
        if not part:
            continue
        # porcelain entries are "XY path". Rename records may include a second path.
        value = part[3:] if len(part) > 3 else part
        if " -> " in value:
            paths.update(value.split(" -> ", 1))
        else:
            paths.add(value)
    return paths


def is_dirty_under(source: Path, dirty_paths: set[str], repo_root: Path) -> bool:
    try:
        source_rel = source.resolve().relative_to(repo_root).as_posix().rstrip("/")
    except ValueError:
        return False
    return any(path == source_rel or path.startswith(source_rel + "/") or source_rel.startswith(path.rstrip("/") + "/") for path in dirty_paths)


def source_provenance(item: dict, dirty_paths: set[str], git_commit: str | None, space_info: dict | None, repo_root: Path) -> dict:
    source = ROOT / item["source"]
    provenance_source = ROOT / item.get("provenanceSource", item["source"])
    source_str = item["source"]
    prov = {
        "source": source_str,
        "provenanceSource": item.get("provenanceSource", item["source"]),
        "publishPath": item["publishPath"],
        "entry": item["entry"],
        "status": item.get("status"),
    }
    if source_str.startswith("publish/current-space/"):
        prov.update({
            "kind": "mirrored-space",
            "space": space_info.get("id") if space_info else None,
            "spaceSha": space_info.get("sha") if space_info else None,
            "spaceLastModified": space_info.get("last_modified") if space_info else None,
        })
    else:
        prov.update({
            "kind": "local-git",
            "gitCommit": git_commit,
            "dirty": is_dirty_under(provenance_source, dirty_paths, repo_root),
        })
    return prov

def copy_path(src: Path, dst: Path) -> None:
    if src.name in EXCLUDE_NAMES:
        return
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore_names, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ignore_names(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDE_NAMES or name.endswith(":Zone.Identifier")}


def clean_site() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)


def selected_items(manifest: dict, include_candidates: bool) -> list[dict]:
    statuses = {"published"} | ({"candidate"} if include_candidates else set())
    return [item for item in manifest["items"] if item.get("status") in statuses]


def stage_items(items: list[dict]) -> None:
    missing: list[str] = []
    for item in items:
        source = ROOT / item["source"]
        if not source.exists():
            hint = item.get("optionalSourceHint", "")
            missing.append(f"{item['title']}: {source.relative_to(ROOT)}" + (f"\n  {hint}" if hint else ""))
            continue

        dest = SITE / item["publishPath"]
        dest.mkdir(parents=True, exist_ok=True)
        for part in item.get("include", ["."]):
            src = source if part == "." else source / part
            if not src.exists():
                missing.append(f"{item['title']}: missing {src.relative_to(ROOT)}")
                continue
            copy_path(src, dest / src.name if part != "." else dest)

        entry = dest / item["entry"]
        if not entry.exists():
            missing.append(f"{item['title']}: missing entry {entry.relative_to(ROOT)}")

    if missing:
        raise SystemExit("Missing publish inputs:\n- " + "\n- ".join(missing))


def display_date(item: dict) -> str:
    return item["date"]


def display_event(item: dict) -> str:
    location = item["location"].strip()
    return f"{item['event']} · {location}" if location else item["event"]


def sorted_items(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (item.get("sortDate", "0000-00-00"), item.get("title", "")),
        reverse=True,
    )


def render_index(manifest: dict, items: list[dict], build_meta: dict) -> str:
    title = html.escape(manifest.get("title", "Presentations"))
    ordered = sorted_items(items)
    years = sorted({int(item["year"]) for item in ordered}, reverse=True)

    commit = build_meta.get("gitCommit") or "unknown"
    short_commit = commit[:12] if commit != "unknown" else commit
    dirty_label = " · local changes present" if build_meta.get("gitDirty") else ""
    built_at = html.escape(build_meta.get("generatedAt", ""))

    latest = ordered[0] if ordered else None
    latest_title = html.escape(latest["title"]) if latest else ""
    latest_event = html.escape(display_event(latest)) if latest else ""
    latest_path = html.escape(f"{latest['publishPath'].rstrip('/')}/{latest['entry']}") if latest else "#"
    latest_date = html.escape(display_date(latest)) if latest else ""
    year_nav = "".join(f'<a href="#y{year}">{year}</a>' for year in years)

    sections = []
    for year in years:
        cards = []
        for idx, item in enumerate([i for i in ordered if int(i["year"]) == year], 1):
            item_title = html.escape(item["title"])
            event_label = html.escape(display_event(item))
            date = html.escape(display_date(item))
            sort_date = html.escape(item.get("sortDate", ""))
            path = html.escape(f"{item['publishPath'].rstrip('/')}/{item['entry']}")
            event = item.get("eventUrl")
            event_link = f'<a class="pill ghost" href="{html.escape(event)}">Event</a>' if event else ""
            note = item.get("note")
            note_html = f'<p class="note">{html.escape(note)}</p>' if note else ""
            cards.append(f"""
        <article class="talk-card" style="--i:{idx}">
          <a class="card-main" href="{path}">
            <span class="stamp"><time datetime="{sort_date}">{date}</time></span>
            <span class="talk-copy">
              <span class="event-line">{event_label}</span>
              <h3>{item_title}</h3>
            </span>
          </a>
          <div class="card-actions">
            {event_link}
            <a class="pill" href="{path}">View deck</a>
            <a class="arrow" href="{path}" aria-label="Open {item_title}">↗</a>
          </div>
          {note_html}
        </article>""")
        sections.append(f"""
    <section class="year-section" id="y{year}">
      <div class="year-rail"><span>{year}</span></div>
      <div class="year-content">
        <h2>{year}</h2>
        <div class="talk-grid">{''.join(cards)}
        </div>
      </div>
    </section>""")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --s-0: #0b0c0f;
      --s-1: #14161b;
      --s-2: #1c1f26;
      --s-3: #262a33;
      --b-1: #262a33;
      --b-2: #3a3f4a;
      --b-3: #555b68;
      --fg-0: #f0ece2;
      --fg-1: #b9b3a5;
      --fg-2: #7d786d;
      --fg-3: #4b483f;
      --a-0: #f5a400;
      --a-hi: #ffc649;
      --a-dim: #8a5d00;
      --a-bg: rgba(245, 164, 0, .08);
      --a-line: rgba(245, 164, 0, .28);
      --ok: #6ad19c;
      --info: #6aa3f7;
      --no: #f06b5a;
      --mono: "JetBrains Mono", "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --shadow-terminal: 0 18px 44px rgba(0,0,0,.34), 0 0 36px rgba(245,164,0,.08);
      font-family: var(--mono);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; background: var(--s-0); }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--fg-0);
      background:
        radial-gradient(circle at 78% 8%, rgba(245,164,0,.12), transparent 28rem),
        radial-gradient(circle at 12% 18%, rgba(106,163,247,.08), transparent 24rem),
        linear-gradient(rgba(245,164,0,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(245,164,0,.028) 1px, transparent 1px),
        var(--s-0);
      background-size: auto, auto, 44px 44px, 44px 44px, auto;
      font-family: var(--mono);
      font-size: 14.5px;
      line-height: 1.65;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      font-feature-settings: "ss02" on, "calt" on;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(180deg, transparent 0 50%, rgba(255,255,255,.018) 50% 100%),
        radial-gradient(circle at 50% 0, transparent 0, rgba(0,0,0,.45) 100%);
      background-size: 100% 4px, auto;
      opacity: .55;
      z-index: 0;
    }}
    ::selection {{ background: var(--a-0); color: #0b0c0f; }}
    a {{
      color: var(--a-0);
      text-decoration: none;
      border-bottom: 1px solid var(--a-line);
      transition: color 180ms cubic-bezier(.2,0,0,1), border-color 180ms cubic-bezier(.2,0,0,1), transform 180ms cubic-bezier(.2,0,0,1);
    }}
    a:hover {{ color: var(--a-hi); border-color: var(--a-0); }}
    main {{
      position: relative;
      z-index: 1;
      width: min(1180px, calc(100% - 2rem));
      margin: 0 auto;
      padding: 2rem 0 4rem;
    }}
    .hero {{
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 22rem;
      gap: clamp(1.5rem, 4vw, 4rem);
      align-items: end;
      min-height: 24rem;
      padding: clamp(2rem, 7vw, 5.5rem) 0 2rem;
      border-bottom: 1px solid var(--b-1);
    }}
    .hero::after {{
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: -1px;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--a-0), transparent);
      opacity: .5;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: .55rem;
      margin: 0 0 1rem;
      color: var(--fg-2);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .18em;
      text-transform: uppercase;
    }}
    .eyebrow::before {{ content: "❯"; color: var(--a-0); }}
    h1 {{
      max-width: 12ch;
      margin: 0;
      color: var(--fg-0);
      font-size: clamp(3rem, 9vw, 7.2rem);
      font-weight: 650;
      line-height: .92;
      letter-spacing: -.075em;
      text-shadow: 0 0 28px rgba(245,164,0,.08);
    }}
    .lede {{
      max-width: 58rem;
      margin: 1.25rem 0 0;
      color: var(--fg-1);
      font-size: clamp(.98rem, 1.55vw, 1.16rem);
      line-height: 1.7;
    }}
    .hero-card {{
      position: relative;
      padding: 1rem;
      background: linear-gradient(180deg, rgba(245,164,0,.055), transparent 44%), var(--s-1);
      border: 1px solid var(--b-2);
      border-radius: 2px;
      box-shadow: inset 0 -2px 0 var(--b-2), var(--shadow-terminal);
    }}
    .hero-card::before {{
      content: "latest";
      position: absolute;
      top: -1px;
      right: -1px;
      padding: .28rem .55rem;
      color: #0b0c0f;
      background: var(--a-0);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .16em;
      text-transform: uppercase;
    }}
    .hero-card time {{
      display: block;
      margin-bottom: .55rem;
      color: var(--fg-2);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    .latest-event {{
      display: block;
      margin-bottom: .35rem;
      color: var(--a-0);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    .hero-card strong {{
      display: block;
      color: var(--fg-0);
      font-size: 1.1rem;
      line-height: 1.28;
      font-weight: 650;
    }}
    .hero-card a {{
      display: inline-flex;
      margin-top: 1rem;
      color: var(--a-0);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .1em;
      text-transform: uppercase;
    }}
    .meta-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: .7rem;
      align-items: center;
      justify-content: space-between;
      padding: .9rem 0;
      border-bottom: 1px solid var(--b-1);
      color: var(--fg-2);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    .year-nav {{ display: flex; gap: .45rem; }}
    .year-nav a {{
      padding: .32rem .55rem;
      color: var(--fg-1);
      background: var(--s-1);
      border: 1px solid var(--b-1);
      border-radius: 2px;
    }}
    .year-nav a:hover {{ color: var(--fg-0); border-color: var(--a-line); box-shadow: inset 0 -2px 0 var(--a-0); }}
    .year-section {{
      display: grid;
      grid-template-columns: 6.25rem minmax(0, 1fr);
      gap: clamp(1rem, 3vw, 2.5rem);
      padding: 2rem 0;
      border-bottom: 1px solid var(--b-1);
    }}
    .year-rail span {{
      position: sticky;
      top: 1rem;
      display: inline-block;
      color: var(--a-0);
      font-size: clamp(1.8rem, 4.2vw, 3rem);
      font-weight: 700;
      letter-spacing: -.08em;
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      text-shadow: 0 0 18px rgba(245,164,0,.16);
    }}
    h2 {{
      margin: 0 0 1rem;
      color: var(--fg-2);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .22em;
      text-transform: uppercase;
    }}
    h2::before {{ content: "// "; color: var(--a-0); }}
    .talk-grid {{ display: grid; gap: .75rem; }}
    .talk-card {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: .8rem 1rem;
      padding: .95rem 1rem;
      background: color-mix(in srgb, var(--s-1) 90%, transparent);
      border: 1px solid var(--b-1);
      border-radius: 2px;
      box-shadow: inset 0 -1px 0 rgba(245,164,0,.06);
      transition: transform 180ms cubic-bezier(.2,0,0,1), background 180ms cubic-bezier(.2,0,0,1), border-color 180ms cubic-bezier(.2,0,0,1), box-shadow 180ms cubic-bezier(.2,0,0,1);
      animation: rise .42s both cubic-bezier(.16,1,.3,1);
      animation-delay: calc(var(--i) * 30ms);
    }}
    .talk-card:hover {{
      transform: translateY(-1px);
      background: var(--s-2);
      border-color: var(--b-3);
      box-shadow: inset 0 -2px 0 var(--a-0), 0 10px 30px rgba(0,0,0,.22);
    }}
    .card-main {{
      display: grid;
      grid-template-columns: 7.2rem minmax(0, 1fr);
      gap: 1rem;
      align-items: baseline;
      border-bottom: 0;
    }}
    .talk-copy {{
      display: grid;
      gap: .28rem;
      min-width: 0;
    }}
    .event-line {{
      color: var(--a-0);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    .stamp {{
      color: var(--fg-2);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    h3 {{
      margin: 0;
      color: var(--fg-0);
      font-size: clamp(1rem, 1.75vw, 1.45rem);
      font-weight: 620;
      line-height: 1.22;
      letter-spacing: -.035em;
    }}
    .arrow {{
      display: inline-grid;
      place-items: center;
      width: 2.25rem;
      min-width: 2.25rem;
      align-self: stretch;
      color: var(--a-0);
      background: rgba(245,164,0,.06);
      border: 1px solid var(--a-line);
      border-radius: 2px;
      box-shadow: inset 0 -2px 0 rgba(245,164,0,.18);
      font-size: 1rem;
      line-height: 1;
      transform: translateX(0);
      transition: transform 180ms cubic-bezier(.2,0,0,1), background 180ms cubic-bezier(.2,0,0,1), border-color 180ms cubic-bezier(.2,0,0,1), color 180ms cubic-bezier(.2,0,0,1), box-shadow 180ms cubic-bezier(.2,0,0,1);
    }}
    .arrow:hover, .talk-card:hover .arrow {{
      color: #0b0c0f;
      background: var(--a-0);
      border-color: var(--a-0);
      box-shadow: inset 0 -2px 0 rgba(0,0,0,.18), 0 0 18px rgba(245,164,0,.16);
      transform: translateX(.08rem);
    }}
    .card-actions {{ display: flex; gap: .45rem; justify-content: end; align-items: stretch; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      white-space: nowrap;
      padding: .44rem .66rem;
      color: var(--fg-0);
      background: var(--s-1);
      border: 1px solid var(--b-2);
      border-radius: 2px;
      box-shadow: inset 0 -2px 0 var(--b-2);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .1em;
      text-transform: uppercase;
    }}
    .pill:hover {{ color: var(--fg-0); background: var(--s-2); border-color: var(--b-3); box-shadow: inset 0 -2px 0 var(--a-0); }}
    .pill.ghost {{ color: var(--fg-1); background: transparent; border-color: var(--b-1); box-shadow: none; }}
    .note {{
      grid-column: 1 / -1;
      margin: -.15rem 0 0 8.2rem;
      max-width: 58rem;
      color: var(--fg-2);
      font-size: .88rem;
      line-height: 1.55;
    }}
    footer {{
      display: flex;
      flex-wrap: wrap;
      gap: .5rem;
      justify-content: space-between;
      margin: 2.25rem 0 0;
      padding-top: 1rem;
      border-top: 1px solid var(--b-1);
      color: var(--fg-2);
      font-size: 11px;
      font-weight: 650;
      letter-spacing: .04em;
    }}
    footer code {{ color: var(--fg-1); }}
    footer a {{ color: var(--a-0); font-weight: 900; }}
    @keyframes rise {{ from {{ opacity: 0; transform: translateY(.45rem); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @media (max-width: 760px) {{
      main {{ width: min(100% - 1rem, 1180px); padding-top: .5rem; }}
      .hero {{ grid-template-columns: 1fr; min-height: auto; }}
      .year-section {{ grid-template-columns: 1fr; }}
      .year-rail span {{ position: static; writing-mode: horizontal-tb; transform: none; }}
      .talk-card {{ grid-template-columns: 1fr; }}
      .card-main {{ grid-template-columns: 1fr; }}
      .stamp {{ grid-column: 1; }}
      .talk-copy {{ grid-column: 1; }}
      .card-actions {{ justify-content: start; }}
      .note {{ margin-left: 0; }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="hero">
      <div>
        <p class="eyebrow">evalstate archive department</p>
        <h1>{title}</h1>
        <p class="lede">A maintained shelf of selected talks, demos, and conference decks. Hosted by Hugging Face Spaces.</p>
      </div>
      <aside class="hero-card">
        <time>{latest_date}</time>
        <span class="latest-event">{latest_event}</span>
        <strong>{latest_title}</strong>
        <a href="{latest_path}">Open newest deck ↗</a>
      </aside>
    </header>
    <div class="meta-bar">
      <span>{len(ordered)} decks · newest first</span>
      <nav class="year-nav" aria-label="Years">{year_nav}</nav>
    </div>
    {''.join(sections)}
    <footer>
      <span>Built from <code>evalstate/miscellany@{html.escape(short_commit)}</code>{dirty_label}</span>
      <span>{built_at} · <a href="publish-manifest.json">provenance</a></span>
    </footer>
  </main>
</body>
</html>
"""


def write_space_readme(manifest: dict) -> None:
    (SITE / "README.md").write_text(
        f"""---
title: {manifest.get('title', 'Presentations')}
emoji: 🎤
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
---

Generated static presentation index. Source and deploy scripts live in `publish/`.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static Space publish tree.")
    parser.add_argument("--include-candidates", action="store_true", help="Also stage manifest items with status=candidate.")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = selected_items(manifest, args.include_candidates)
    git_commit = git_output("rev-parse", "HEAD")
    repo_root = git_root()
    dirty_paths = git_status_paths()
    space_info = json.loads(CURRENT_SPACE_INFO.read_text(encoding="utf-8")) if CURRENT_SPACE_INFO.exists() else None
    build_meta = {
        "generatedAt": dt.datetime.now(dt.UTC).isoformat(),
        "gitCommit": git_commit,
        "gitDirty": bool(dirty_paths),
    }
    clean_site()
    stage_items(items)
    provenance = {
        "title": manifest.get("title"),
        "space": manifest.get("space"),
        "build": build_meta,
        "items": [source_provenance(item, dirty_paths, git_commit, space_info, repo_root) for item in items],
    }
    (SITE / "publish-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (SITE / "index.html").write_text(render_index(manifest, items, build_meta), encoding="utf-8")
    write_space_readme(manifest)
    print(f"Built {SITE.relative_to(ROOT)} with {len(items)} presentations")


if __name__ == "__main__":
    main()
