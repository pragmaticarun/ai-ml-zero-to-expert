#!/usr/bin/env python3
"""Check relative markdown links under the course root. Exit 1 if any target is missing."""
from pathlib import Path
import re, sys
root = Path(__file__).resolve().parents[1]
broken = []
checked = 0
for f in list(root.glob("*.md")) + list((root/"docs").glob("*.md")) + list((root/"weeks").glob("*.md")) + list((root/"data").glob("*.md")) + list((root/"code/capstone/out").glob("*.md")):
    if not f.is_file():
        continue
    text = f.read_text(errors="replace")
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        url = m.group(2).strip()
        if url.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = url.split("#")[0].split("?")[0]
        if not path:
            continue
        checked += 1
        target = (f.parent / path).resolve()
        if not target.exists():
            broken.append(f"{f.relative_to(root)} -> {url}")
print(f"checked {checked} relative links; broken {len(broken)}")
for b in broken:
    print("BROKEN", b)
sys.exit(1 if broken else 0)
