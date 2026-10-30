# Course website (static)

A static site that needs no server: it works offline from `file://` and on GitHub Pages, uses no CDN, and is written in vanilla JS. Your progress is stored in your browser's localStorage under the key `aiml-course-v1`. Use **Settings → Export / Import JSON** to back it up or move it to another browser.

## Build
```bash
.venv/bin/python site/build.py      # needs: markdown, pygments
```
The build regenerates `site/days`, `site/weeks`, `site/exams`, the top-level pages, `assets/data.js`, `assets/search.js` and `assets/pygments.css` from `SYLLABUS.md`, `weeks/*.md`, `review-schedule.csv` and the exam files. Rerun it whenever the course files change.

## Open
- To use it offline, open `site/index.html` in a browser.
- To serve it locally, run `python3 -m http.server 8000` from the **repo root**, then go to http://localhost:8000/site/

## GitHub Pages
Publish the **repo root**, not `site/`, so the relative `../code/…` and `../weeks/…` links keep working. In the repo settings, go to Pages and set the source to the branch `main`, folder `/ (root)`. The site is then at `https://<user>.github.io/<repo>/site/`.

## Features
- **Dashboard:** days done of 180, hours of ~1080, current month and week, streak, estimated finish, a 180-cell heatmap and a Continue button.
- **Day pages:** 5 checkable blocks (the day completes automatically when all 5 are checked, or you can toggle it manually), notes, hours, prev/next, and links to the code and the week file.
- **Week quizzes and exams:** reveal the answer and mark yourself right or wrong. Scores are saved. Only questions and answers that exist in the source files are shown.
- **Review:** what's due today (from completion dates and review-schedule.csv), plus flashcards scheduled with SM-2 (Again/Hard/Good/Easy).
- **Syllabus:** a filterable table and full-text search.
- **Settings:** start date (planned dates in Asia/Kolkata, with behind/ahead flags), Export/Import, and Reset (asks for confirmation).
- **Display and keyboard:** dark/light mode, `[` and `]` for prev/next day, and Space plus 1–4 in the flashcard reviewer.
- Mermaid diagrams appear as labelled source code. Raw HTML in the markdown is escaped.

## Tests
```bash
node --check site/assets/core.js site/assets/app.js
node --test site/tests/core.test.js
python3 -m http.server 8765 &   # from repo root
bash site/tests/smoke.sh       # every page/asset 200 + all relative links exist
```
`site/tests/seed.html` fills localStorage with sample progress for manual testing in a browser.
