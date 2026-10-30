#!/usr/bin/env python3
"""Build the static course site from the Markdown in ../weeks, ../SYLLABUS.md,
../flashcards and ../review-schedule.csv. Run: python site/build.py
Needs: markdown, pygments (see requirements.txt). No network, no CDN."""
import csv, glob, html, json, os, posixpath, re, sys
from pathlib import Path
import markdown
from pygments.formatters import HtmlFormatter

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
BLOCKS = ["Concept lesson", "Worked example", "Coding lab", "Practice problems", "Review"]
HEAD_RE = re.compile(r'^(#{1,6})\s+(.*?)\s*#*\s*$')
DAY_RE = re.compile(r'^##\s+Day\s+(\d+)\s*[:\-—]\s*(.*)$')
ANS_RE = re.compile(r'answers?\b', re.I)

# ---------- safe markdown ----------
def make_md():
    md = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists", "codehilite", "toc"],
                           extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite"},
                                              "toc": {"permalink": False}})
    # Disable raw HTML: block HTML and inline HTML become escaped text.
    for name in ("html_block",):
        if name in md.preprocessors: md.preprocessors.deregister(name)
    for name in ("html",):
        if name in md.inlinePatterns: md.inlinePatterns.deregister(name)
    return md
MD = make_md()
SAFE_SCHEMES = ("http://", "https://", "mailto:", "#", "../", "./")

def fix_urls(htm, base_dir, depth):
    """Rewrite relative href/src (relative to base_dir in repo) for a page `depth` dirs below site/."""
    up = "../" * (depth + 1)
    def rep(m):
        attr, q, url = m.group(1), m.group(2), html.unescape(m.group(3))
        u = url.strip()
        low = u.lower()
        if low.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        if ":" in u.split("/")[0]:  # javascript:, data:, etc.
            return f'{attr}={q}#{q}'
        target = posixpath.normpath(posixpath.join(base_dir, u)) if not u.startswith("/") else u.lstrip("/")
        return f'{attr}={q}{html.escape(up + target)}{q}'
    return re.sub(r'(href|src)=(["\'])(.*?)\2', rep, htm)

def render(text, base_dir="weeks", depth=1):
    MD.reset()
    # mermaid fences -> labelled plain code blocks (readable source; no JS renderer vendored)
    text = re.sub(r'```mermaid\s*\n', '```text\n%%MERMAID%%\n', text)
    out = MD.convert(text)
    out = re.sub(r'<div class="codehilite">(<pre>(?:<span></span>)?<code>)%%MERMAID%%\n?',
                 r'<div class="codehilite mermaid-src" data-label="Diagram (Mermaid source)">\1', out)
    out = out.replace('%%MERMAID%%', '')
    return fix_urls(out, base_dir, depth)

def plain(text):
    t = re.sub(r'```.*?```', ' ', text, flags=re.S)
    t = re.sub(r'[`*_>#|\[\]()]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

# ---------- parsing ----------
def headings(lines):
    out = []
    infence = False
    for i, l in enumerate(lines):
        if l.lstrip().startswith("```"): infence = not infence; continue
        if infence: continue
        m = HEAD_RE.match(l)
        if m: out.append((i, len(m.group(1)), m.group(2)))
    return out

def numbered(lines):
    """Top-level numbered items -> {n: text}. Continuation lines are appended."""
    items, cur = {}, None
    infence = False
    for l in lines:
        if l.lstrip().startswith("```"): infence = not infence
        m = None if infence else re.match(r'^(\d+)\.\s+(.*)$', l)
        if m:
            cur = int(m.group(1)); items[cur] = m.group(2); continue
        if cur is not None:
            if not infence and HEAD_RE.match(l): cur = None; continue
            items[cur] += "\n" + l
    return {k: v.strip() for k, v in items.items()}

def split_inline(text):
    m = re.search(r'\*\*\s*Answer\s*:?\s*\*\*\s*:?', text)
    if not m: return text.strip(), None
    return text[:m.start()].strip(), text[m.end():].strip()

def parse_qa(lines, hs, start_idx, bound):
    """start_idx: index into hs of the start heading; bound: line index limit."""
    i0, lvl, _ = hs[start_idx]
    end_q = bound
    for (j, l2, t2) in hs[start_idx + 1:]:
        if j >= bound: break
        if l2 <= lvl: end_q = j; break
    region = lines[i0 + 1:end_q]
    # answers heading inside region?
    sub = [(j, l2, t2) for (j, l2, t2) in hs if i0 < j < end_q and ANS_RE.search(t2) and l2 > lvl]
    if sub:
        aj, al, _ = sub[0]
        qlines = lines[i0 + 1:aj]
        aend = end_q
        for (j, l2, t2) in hs:
            if j > aj and l2 <= al: aend = min(aend, j); break
        alines = lines[aj + 1:aend]
    else:
        qlines = region
        alines = []
        later = [(j, l2, t2) for (j, l2, t2) in hs if end_q <= j < bound and ANS_RE.search(t2) and not DAY_RE.match("#" * l2 + " " + t2)]
        if later:
            aj, al, _ = later[0]
            aend = bound
            for (j, l2, t2) in hs:
                if j > aj and l2 <= al: aend = j; break
            alines = lines[aj + 1:aend]
    qs = numbered(qlines)
    ans = numbered(alines)
    out = []
    for n in sorted(qs):
        q, a = split_inline(qs[n])
        if a is None: a = ans.get(n)
        out.append({"n": n, "q": q, "a": a})
    return out

def parse_heading_qa(text):
    """'## Q01. question' / '## 1. (M1) question' followed by **Answer:** text."""
    out = []
    parts = re.split(r'^##\s+', text, flags=re.M)[1:]
    for p in parts:
        head, _, body = p.partition("\n")
        m = re.match(r'Q?(\d+)\.\s*(.*)', head)
        if not m: continue
        q, a = split_inline(body)
        out.append({"n": int(m.group(1)), "q": (m.group(2) + ("\n\n" + q if q else "")).strip(), "a": a})
    return out

def qa_render(items, base_dir, depth):
    return [{"n": it["n"], "q": render(it["q"], base_dir, depth),
             "a": render(it["a"], base_dir, depth) if it["a"] else None} for it in items]

# ---------- load ----------
def load():
    syl = {}
    for l in (ROOT / "SYLLABUS.md").read_text().splitlines():
        m = re.match(r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$', l)
        if m: syl[int(m.group(1))] = {"week": int(m.group(2)), "month": int(m.group(3)), "topic": m.group(4)}
    weeks, days, exams = {}, {}, []
    for f in sorted(glob.glob(str(ROOT / "weeks/week-*.md"))):
        w = int(re.search(r'week-(\d+)', f).group(1))
        text = Path(f).read_text()
        lines = text.splitlines()
        hs = headings(lines)
        title = next((t for (_, l, t) in hs if l == 1), f"Week {w}")
        day_starts = [(i, int(DAY_RE.match(lines[i]).group(1)), DAY_RE.match(lines[i]).group(2)) for (i, l, t) in hs if l == 2 and DAY_RE.match(lines[i])]
        # day ends at next Day heading or a week-level ## heading about quiz/project checklist
        stops = sorted([i for (i, _, _) in day_starts] + [i for (i, l, t) in hs if l == 2 and re.search(r'quiz|project checklist|week \d+ project', t, re.I) and not DAY_RE.match(lines[i])] + [len(lines)])
        nonday = [True] * len(lines)
        for (i, n, t) in day_starts:
            e = next(s for s in stops if s > i)
            body = "\n".join(lines[i + 1:e])
            for k in range(i, e): nonday[k] = False
            codes = []
            for c in re.findall(r'code/[A-Za-z0-9_./-]+', body):
                c = c.rstrip(".,)")
                if (ROOT / c).exists() and c not in codes: codes.append(c)
            blocks = [t2 for (j, l2, t2) in hs if i < j < e and l2 == 3 and re.match(r'Block\s*\d', t2)]
            days[n] = {"n": n, "week": w, "title": t, "md": body, "codes": codes, "blockHeads": blocks}
        week_md = "\n".join(l for k, l in enumerate(lines) if nonday[k] and not (k == 0 and l.startswith("# ")))
        # quiz: last heading mentioning quiz, prefer level 2 non-day
        qcands = [k for k, (i, l, t) in enumerate(hs) if re.search(r'quiz', t, re.I) and not DAY_RE.match(lines[i]) and not ANS_RE.fullmatch(t)]
        q2 = [k for k in qcands if hs[k][1] == 2]
        quiz = []
        if q2 or qcands:
            k = (q2 or qcands)[-1]
            quiz = parse_qa(lines, hs, k, len(lines))
        weeks[w] = {"n": w, "title": title, "md": week_md, "quiz": quiz}
        # exams inside week files
        for k, (i, l, t) in enumerate(hs):
            if DAY_RE.match(lines[i]) or ANS_RE.search(t) and not re.search(r'exam\b.*\(', t): pass
            if DAY_RE.match(lines[i]): continue
            first = next((x for x in lines[i + 1:i + 4] if x.strip()), "")
            if (re.search(r'checkpoint exam|written exam', t, re.I) or re.search(r'this block is the exam', first, re.I)) and not re.match(r'.*answers', t, re.I):
                bound = next((j for (j, l2, t2) in hs if j > i and l2 <= 2 and DAY_RE.match(lines[j])), len(lines))
                items = parse_qa(lines, hs, k, bound)
                dayn = max([d for (s, d, _) in day_starts if s < i], default=None)
                if items:
                    eid = f"w{w:02d}-d{dayn}"; k2 = 2
                    while any(x["id"] == eid for x in exams): eid = f"w{w:02d}-d{dayn}-{k2}"; k2 += 1
                    exams.append({"id": eid, "day": dayn, "week": w, "title": f"Day {dayn}: " + ("Month-3 checkpoint exam (" if re.search(r'this block is the exam', first, re.I) else "") + re.sub(r'^Block \d+:\s*', '', t) + (")" if re.search(r'this block is the exam', first, re.I) else ""), "src": f"weeks/week-{w:02d}.md", "items": items})
    for fn, d, t in [("code/week-25/out/day175_exam.md", 175, "Month 6 midpoint checkpoint exam (Day 175)"),
                     ("code/capstone/out/FINAL_EXAM.md", 180, "Final exam (Day 180, cumulative)")]:
        p = ROOT / fn
        if p.exists():
            items = parse_heading_qa(p.read_text())
            if items: exams.append({"id": f"d{d}-file", "day": d, "week": syl.get(d, {}).get("week"), "title": t, "src": fn, "items": items})
    cards = []
    for f in sorted(glob.glob(str(ROOT / "flashcards/month-*.csv"))):
        m = int(re.search(r'month-(\d+)', f).group(1))
        with open(f, newline="") as fh:
            for idx, row in enumerate(csv.reader(fh)):
                if len(row) < 2 or not row[0].strip(): continue
                tags = row[2] if len(row) > 2 else ""
                dm = re.search(r'\bd(\d+)\b', tags)
                cards.append({"id": f"m{m}-{idx}", "f": row[0], "b": row[1], "tags": tags, "day": int(dm.group(1)) if dm else None, "month": m})
    sched = {}
    with open(ROOT / "review-schedule.csv", newline="") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            if not row or not row[0].strip().isdigit(): continue
            rv = [int(x) for x in re.findall(r'\d+', row[1])] if len(row) > 1 else []
            sched[int(row[0])] = {"review": rv, "tags": row[2] if len(row) > 2 else ""}
    return syl, weeks, days, exams, cards, sched

# ---------- html ----------
def page(title, body, depth, kind, attrs=""):
    up = "../" * depth
    nav = [("index.html", "Dashboard"), ("syllabus.html", "Syllabus &amp; search"), ("review.html", "Review &amp; flashcards"),
           ("exams.html", "Exams"), ("settings.html", "Settings &amp; data")]
    links = "".join(f'<a href="{up}{h}">{t}</a>' for h, t in nav)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · AI/ML Zero to Expert</title>
<script>try{{var t=JSON.parse(localStorage.getItem('aiml-course-v1')||'{{}}').theme;if(t)document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
<link rel="stylesheet" href="{up}assets/style.css"><link rel="stylesheet" href="{up}assets/pygments.css">
</head><body data-kind="{kind}" data-root="{up}" {attrs}>
<a class="skip" href="#main">Skip to content</a>
<header class="top"><a class="brand" href="{up}index.html">AI/ML · 180 days</a>
<nav aria-label="Main">{links}</nav>
<button id="theme-toggle" type="button" aria-label="Toggle dark mode">◐ Theme</button></header>
<main id="main" tabindex="-1">{body}</main>
<footer><p>Static, offline site. Progress lives only in this browser (localStorage); use Settings → Export to back it up.</p></footer>
<script src="{up}assets/data.js"></script><script src="{up}assets/core.js"></script><script src="{up}assets/app.js"></script>
</body></html>"""

def main():
    syl, weeks, days, exams, cards, sched = load()
    missing = [d for d in range(1, 181) if d not in days]
    if missing: print("WARNING missing day sections:", missing)
    for d in ("days", "weeks", "exams"): (SITE / d).mkdir(exist_ok=True)
    # data.js (loaded via <script>, so it works from file://)
    daymeta = []
    search = []
    for n in sorted(days):
        d = days[n]; s = syl.get(n, {})
        daymeta.append({"n": n, "week": d["week"], "month": s.get("month", (n - 1) // 30 + 1), "title": d["title"], "codes": d["codes"]})
        search.append({"n": n, "t": d["title"], "x": plain(d["md"])[:6000]})
    data = {"days": daymeta, "weeks": [{"n": w, "title": weeks[w]["title"], "nq": len(weeks[w]["quiz"])} for w in sorted(weeks)],
            "exams": [{"id": e["id"], "day": e["day"], "week": e["week"], "title": e["title"], "nq": len(e["items"])} for e in exams],
            "cards": cards, "sched": sched, "blocks": BLOCKS, "totalHours": 1080}
    (SITE / "assets/data.js").write_text("window.COURSE=" + json.dumps(data, ensure_ascii=False) + ";\n")
    (SITE / "assets/search.js").write_text("window.SEARCH=" + json.dumps(search, ensure_ascii=False) + ";\n")
    # pygments css (light + dark)
    light = HtmlFormatter(style="default").get_style_defs(".codehilite")
    dark = HtmlFormatter(style="monokai").get_style_defs('[data-theme="dark"] .codehilite')
    (SITE / "assets/pygments.css").write_text(light + "\n" + dark + "\n")
    # day pages
    ns = sorted(days)
    for idx, n in enumerate(ns):
        d = days[n]; w = d["week"]
        prev = f'<a rel="prev" href="day-{ns[idx-1]:03d}.html">← Day {ns[idx-1]}</a>' if idx else '<span></span>'
        nxt = f'<a rel="next" href="day-{ns[idx+1]:03d}.html">Day {ns[idx+1]} →</a>' if idx + 1 < len(ns) else '<span></span>'
        codes = "".join(f'<li><a href="../../{html.escape(c)}"><code>{html.escape(c)}</code></a></li>' for c in d["codes"]) or "<li>No code file referenced for this day.</li>"
        checks = "".join(f'<li><label><input type="checkbox" class="blk" data-i="{i}"> {i+1}. {html.escape(b)}' +
                         (f' <small>({html.escape(d["blockHeads"][i])})</small>' if i < len(d["blockHeads"]) else "") + '</label></li>'
                         for i, b in enumerate(BLOCKS))
        body = f"""<nav class="pn" aria-label="Day navigation">{prev}<a href="../weeks/week-{w:02d}.html">Week {w}</a>{nxt}</nav>
<h1>Day {n}: {html.escape(d['title'])}</h1>
<p class="meta">Week {w} · Month {syl.get(n,{}).get('month','?')} · <span id="planned"></span> · <a href="../../weeks/week-{w:02d}.md">source: weeks/week-{w:02d}.md</a></p>
<section class="tracker card" aria-label="Day progress">
<h2>Your progress</h2><p id="day-status" role="status"></p>
<ul class="checks">{checks}</ul>
<p><label><input type="checkbox" id="manual"> Mark day complete (manual)</label></p>
<p><label>Hours studied <input type="number" id="hours" min="0" max="24" step="0.25" inputmode="decimal"></label></p>
<p><label for="notes">Notes</label><br><textarea id="notes" rows="5"></textarea></p>
<h3>Code files</h3><ul>{codes}</ul></section>
<article class="lesson">{render(d['md'], 'weeks', 1)}</article>
<nav class="pn" aria-label="Day navigation">{prev}<a href="../weeks/week-{w:02d}.html">Week {w}</a>{nxt}</nav>"""
        (SITE / f"days/day-{n:03d}.html").write_text(page(f"Day {n}", body, 1, "day", f'data-day="{n}"'))
    # quizzes json per page embedded
    def quiz_html(qid, items, depth):
        r = qa_render(items, "weeks", depth)
        return f'<section class="quiz card" data-quiz="{qid}"><p class="qscore" role="status"></p>' + "".join(
            f'''<div class="q" data-q="{it['n']}"><div class="qtext"><strong>Q{it['n']}.</strong> {it['q']}</div>
<label>Your answer <textarea class="qa" rows="2"></textarea></label>
<div class="qbtns">''' + (f'<button type="button" class="reveal" aria-expanded="false">Reveal answer</button>' if it['a'] else '<em>No written answer in the source file; grade against the lesson.</em>') +
            f''' <button type="button" class="mark" data-v="right">✓ Right</button><button type="button" class="mark" data-v="wrong">✗ Wrong</button></div>
<div class="ans" hidden>{it['a'] or ''}</div></div>''' for it in r) + '<p><button type="button" class="qreset">Clear this quiz</button></p></section>'
    for w in sorted(weeks):
        W = weeks[w]
        dl = "".join(f'<li><a href="../days/day-{n:03d}.html">Day {n}: {html.escape(days[n]["title"])}</a> <span class="st" data-st="{n}"></span></li>' for n in ns if days[n]["week"] == w)
        quiz = quiz_html(f"week-{w:02d}", W["quiz"], 1) if W["quiz"] else "<p>No numbered quiz could be parsed from this week file.</p>"
        wk_ex = "".join(f'<li><a href="../exams/{e["id"]}.html">{html.escape(e["title"])}</a></li>' for e in exams if e["week"] == w)
        prevw = f'<a href="week-{w-1:02d}.html">← Week {w-1}</a>' if w - 1 in weeks else "<span></span>"
        nextw = f'<a href="week-{w+1:02d}.html">Week {w+1} →</a>' if w + 1 in weeks else "<span></span>"
        body = f"""<nav class="pn">{prevw}<a href="../syllabus.html">Syllabus</a>{nextw}</nav>
<h1>{html.escape(W['title'])}</h1><p class="meta"><a href="../../weeks/week-{w:02d}.md">source: weeks/week-{w:02d}.md</a> · <a href="../../pdf/week-{w:02d}.pdf">PDF</a></p>
<section class="card"><h2>Days</h2><ul class="daylist">{dl}</ul>{('<h3>Exams this week</h3><ul>'+wk_ex+'</ul>') if wk_ex else ''}</section>
<section><h2>Interactive weekly quiz ({len(W['quiz'])} questions)</h2>{quiz}</section>
<article class="lesson"><h2>Week overview, goals, project and quiz text</h2>{render(W['md'], 'weeks', 1)}</article>"""
        (SITE / f"weeks/week-{w:02d}.html").write_text(page(f"Week {w}", body, 1, "week", f'data-week="{w}"'))
    for e in exams:
        base = posixpath.dirname(e["src"])
        body = f"""<h1>{html.escape(e['title'])}</h1><p class="meta">Day {e['day']} · {len(e['items'])} questions parsed from <a href="../../{e['src']}"><code>{e['src']}</code></a>{f' · <a href="../days/day-{e["day"]:03d}.html">Day {e["day"]} page</a>' if e['day'] else ''}</p>
{quiz_html('exam-'+e['id'], e['items'], 1).replace('"weeks"', '"weeks"')}"""
        (SITE / f"exams/{e['id']}.html").write_text(page(e["title"], body, 1, "exam"))
    exl = "".join(f'<li><a href="exams/{e["id"]}.html">{html.escape(e["title"])}</a> — Day {e["day"]}, {len(e["items"])} Q <span class="st" data-quizst="exam-{e["id"]}"></span></li>' for e in exams)
    wql = "".join(f'<li><a href="weeks/week-{w:02d}.html">Week {w} quiz</a> — {len(weeks[w]["quiz"])} Q <span class="st" data-quizst="week-{w:02d}"></span></li>' for w in sorted(weeks))
    (SITE / "exams.html").write_text(page("Exams & quizzes", f"<h1>Exams and quizzes</h1><p>Only questions present in the course files are shown. Self-grade: reveal, then mark right or wrong.</p><section class='card'><h2>Checkpoint and final exams</h2><ul>{exl}</ul></section><section class='card'><h2>Weekly quizzes</h2><ul>{wql}</ul></section>", 0, "exams"))
    (SITE / "index.html").write_text(page("Dashboard", """<h1>Dashboard</h1>
<p><a id="continue" class="btn primary" href="days/day-001.html">Continue where you left off</a></p>
<section class="stats" id="stats" aria-live="polite"></section>
<section class="card"><h2>180-day map</h2><p class="legend"><span class="cell done"></span> done <span class="cell partial"></span> in progress <span class="cell behind"></span> behind schedule <span class="cell"></span> not started <span class="cell today"></span> planned today</p><div id="heatmap"></div></section>""", 0, "home"))
    (SITE / "syllabus.html").write_text(page("Syllabus & search", """<h1>Syllabus and search</h1>
<div class="filters"><label>Search titles and content <input type="search" id="q" placeholder="e.g. LoRA, PSI, broadcasting"></label>
<label>Month <select id="fm"><option value="">All</option></select></label>
<label>Status <select id="fs"><option value="">All</option><option value="done">Done</option><option value="partial">In progress</option><option value="todo">Not started</option><option value="behind">Behind schedule</option></select></label></div>
<p id="count" role="status"></p>
<table class="syl"><thead><tr><th>Day</th><th>Week</th><th>Month</th><th>Title</th><th>Planned</th><th>Status</th></tr></thead><tbody id="rows"></tbody></table>
<script src="assets/search.js"></script>""", 0, "syllabus"))
    (SITE / "review.html").write_text(page("Review & flashcards", """<h1>Review and flashcards</h1>
<section class="card"><h2>Review today</h2><div id="review-today"></div></section>
<section class="card"><h2>Flashcards (SM-2 style)</h2>
<div class="filters"><label>Deck <select id="deck"><option value="due">Due from days you completed</option><option value="all-due">Due (all cards)</option></select></label><span id="deck-info" role="status"></span></div>
<div id="fc" class="fc" tabindex="0" aria-live="polite"></div>
<div class="fcbtns"><button type="button" id="flip">Flip (Space)</button>
<button type="button" class="grade" data-g="0">Again (1)</button><button type="button" class="grade" data-g="3">Hard (2)</button><button type="button" class="grade" data-g="4">Good (3)</button><button type="button" class="grade" data-g="5">Easy (4)</button></div></section>""", 0, "review"))
    (SITE / "settings.html").write_text(page("Settings & data", """<h1>Settings and data</h1>
<section class="card"><h2>Schedule</h2><p><label>Start date (Day 1, Asia/Kolkata) <input type="date" id="start"></label></p>
<p>One course day per calendar day (6 h/day). Planned dates and behind/ahead flags use this date.</p></section>
<section class="card"><h2>Backup</h2><p><button type="button" id="export">Export progress (JSON)</button>
<label class="btn">Import JSON <input type="file" id="import" accept="application/json,.json" hidden></label></p>
<p><button type="button" id="reset" class="danger">Reset all progress…</button></p><p id="msg" role="status"></p></section>""", 0, "settings"))
    print(f"days={len(days)} weeks={len(weeks)} exams={len(exams)} cards={len(cards)}")
    for w in sorted(weeks):
        q = weeks[w]["quiz"]; print(f"  week {w:02d}: quiz {len(q)} Q, {sum(1 for x in q if x['a'])} with answers")
    for e in exams: print(f"  exam {e['id']}: {len(e['items'])} Q, {sum(1 for x in e['items'] if x['a'])} with answers — {e['title'][:60]}")

if __name__ == "__main__":
    main()
