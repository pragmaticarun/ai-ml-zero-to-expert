# Course website (static)

A static site that needs no server: it works offline from `file://` and on GitHub Pages, uses no CDN, and is written in vanilla JS. Your progress is stored in your browser's localStorage under the key `aiml-course-v2`. If the older `aiml-course-v1` data is present, it is migrated automatically and the old key is left in place as a backup. You can also sync progress between computers through a private GitHub Gist (see below). **Export / Import JSON** still works, and it accepts both old and new export files.

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

## Sync across computers (optional, private GitHub Gist)
1. On https://pragmaticarun.github.io/ai-ml-zero-to-expert/site/settings.html#sync, follow the link to create a **classic** token with **only the `gist` scope**: https://github.com/settings/tokens/new?scopes=gist&description=aiml-course-progress
2. Paste the token and press **Connect**. The site looks for a secret gist containing `aiml-course-progress.json` and reuses it. If there isn't one, it creates one (`public: false`). You can instead enter an existing gist id.
3. Repeat on each computer, using the same GitHub account. The header shows the sync state on every page: Synced, Pending, Syncing, Offline or Sync error.

**How it works:**
- Sync runs about 2.5 s after each change, when a page loads, when you return to the tab, when the browser comes back online, and when you press **Sync now**.
- Every item carries its own `updatedAt` timestamp: each block, the hours, the notes, the completion flag, each quiz answer and mark, each card's SM-2 state, and each setting. The newest timestamp wins for each item. Deletions and unchecks are recorded as timestamped tombstones, so they win too.
- Writes use `If-Match` with the gist's ETag. If GitHub reports a conflict (409/412), the site re-fetches, merges again and retries.
- Errors are handled as follows:
  - 401 clears the token and asks for it again.
  - 403 or a rate limit shows a message and retries later.
  - 404 offers to recreate the gist.
  - Offline changes are kept locally and retried.

**Privacy:**
- The token is stored only in this browser (`aiml-sync-token`) and is sent only to `api.github.com`. It never appears in exports, the gist, URLs or logs, and the input field is cleared once you connect.
- The gist is secret, but **anyone with its URL can read it**.
- **Disconnect** removes the token and gist id from this browser only.
- Revoke the token at https://github.com/settings/tokens.
- Clocks that are badly wrong on one computer can make an older edit win.

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
node --test site/tests/*.test.js   # core, merge, sync (mocked Gist API)
python3 -m http.server 8765 &   # from repo root
bash site/tests/smoke.sh       # every page/asset 200 + all relative links exist
```
`site/tests/browser_test.py` is an optional Firefox test (Playwright) that drives two simulated computers against a mocked GitHub API. To run it: `pip install playwright && playwright install firefox`. Asset URLs carry `?v=` + `VERSION` from `build.py`, so bump `VERSION` there to refresh browser caches.

`site/tests/seed.html` fills localStorage with sample progress for manual testing in a browser.
