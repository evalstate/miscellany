"""Write the public copy of the deck: slides.public.md and pages/*.public.md.

The published deck has no speaker notes (they hold source references and review detail that
stay private) and stops at the Questions slide: the spare slides after it are left out.
Run by `npm run build:public`; the output files are git-ignored.
"""
import re
from pathlib import Path

DECK = Path(__file__).resolve().parent.parent
NOTES = re.compile(r"<!--.*?-->\n?", re.S)
SPARES = "---\nlayout: section\n---\n\n# Spare slides"

text = (DECK / "slides.md").read_text()
assert text.count(SPARES) == 1, "spare-slides divider not found exactly once"
text = text[: text.index(SPARES)]
assert "# Questions?" in text
for page in sorted(set(re.findall(r"^src: \./(pages/[\w-]+)\.md$", text, re.M))):
    public = NOTES.sub("", (DECK / f"{page}.md").read_text())
    (DECK / f"{page}.public.md").write_text(public)
    text = text.replace(f"src: ./{page}.md", f"src: ./{page}.public.md")
text = NOTES.sub("", text).rstrip("\n") + "\n"
assert "<!--" not in text
(DECK / "slides.public.md").write_text(text)
print("slides.public.md:", text.count("\n# "), "headed slides")
