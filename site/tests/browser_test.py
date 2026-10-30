"""Optional real-browser test (Playwright + Firefox). GitHub's API is MOCKED via request routing; no real token or network.
Run from repo root with `python3 -m http.server 8765` running:  <venv-with-playwright>/bin/python site/tests/browser_test.py"""
import json, sys
from playwright.sync_api import sync_playwright
B = "http://127.0.0.1:8765/site/"; TOKEN = "ghp_" + "F" * 36
gists, calls = {}, []
def api(route):
    r = route.request; calls.append((r.method, r.url, r.headers.get("authorization", "")))
    if r.headers.get("authorization") != "Bearer " + TOKEN: return route.fulfill(status=401, json={"message": "Bad credentials"})
    path = r.url.split("api.github.com")[1].split("?")[0]
    hdr = {"Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag"}
    if path == "/gists" and r.method == "GET": return route.fulfill(json=[{"id": k, "files": {f: {} for f in v["files"]}} for k, v in gists.items()], headers=hdr)
    if path == "/gists" and r.method == "POST":
        b = json.loads(r.post_data); gid = "g%d" % (len(gists) + 1); gists[gid] = {"files": b["files"], "rev": 1, "public": b["public"]}
        return route.fulfill(status=201, json={"id": gid}, headers=hdr)
    gid = path.split("/")[2]; g = gists.get(gid)
    if not g: return route.fulfill(status=404, json={}, headers=hdr)
    et = '"%s-%d"' % (gid, g["rev"])
    if r.method == "GET": return route.fulfill(json={"id": gid, "files": {k: {"content": v["content"], "truncated": False} for k, v in g["files"].items()}}, headers={**hdr, "ETag": et})
    if r.method == "PATCH":
        if r.headers.get("if-match") not in (None, et): return route.fulfill(status=412, json={}, headers=hdr)
        g["files"].update(json.loads(r.post_data)["files"]); g["rev"] += 1; return route.fulfill(json={"id": gid}, headers=hdr)
ok = []
def check(name, cond): ok.append(cond); print(("PASS " if cond else "FAIL ") + name)
with sync_playwright() as p:
    br = p.firefox.launch()
    errs = []
    A = br.new_context(); A.route("https://api.github.com/**", api); a = A.new_page(); a.on("pageerror", lambda e: errs.append(str(e)))
    a.goto(B + "index.html"); check("heatmap has 180 cells", a.locator(".heatmap a, .cell").count() >= 180)
    check("header shows sync status", "Not synced" in a.inner_text("#sync-st"))
    # old v1 data migrates
    a.evaluate("localStorage.setItem('aiml-course-v1', JSON.stringify({v:1,startDate:'2026-10-01',days:{2:{blocks:[true,true,true,true,true],manual:false,notes:'old',hours:6,completedAt:'2026-10-02'}},quizzes:{},cards:{},theme:'',last:2}))")
    a.goto(B + "days/day-002.html"); check("v1 data migrated (day 2 complete)", a.locator(".blk:checked").count() == 5)
    a.goto(B + "days/day-001.html"); a.locator(".blk").first.check(); a.reload()
    check("checkbox persists across reload", a.locator(".blk").first.is_checked())
    a.goto(B + "settings.html")
    a.fill("#sync-token", TOKEN); a.click("#sync-connect"); a.wait_for_function("document.getElementById('sync-st').textContent.includes('Synced')", timeout=10000)
    check("connect created a private gist", len(gists) == 1 and list(gists.values())[0]["public"] is False)
    html = a.content(); check("token not in DOM after connect", TOKEN not in html)
    a.goto(B + "days/day-003.html"); a.fill("#notes, textarea", "note from A"); a.locator("textarea").first.blur()
    a.wait_for_timeout(3500); a.wait_for_function("document.getElementById('sync-st').textContent.includes('Synced')", timeout=10000)
    content = json.loads(list(gists.values())[0]["files"]["aiml-course-progress.json"]["content"])
    check("debounced auto-sync pushed note", content["items"].get("day:3:notes", {}).get("val") == "note from A")
    check("token not in gist", TOKEN not in json.dumps(gists))
    # export excludes token
    a.goto(B + "settings.html")
    with a.expect_download() as d: a.click("#export")
    exp = open(d.value.path()).read(); check("export has no token and is v2", TOKEN not in exp and json.loads(exp)["v"] == 2)
    # second computer
    Bc = br.new_context(); Bc.route("https://api.github.com/**", api); b = Bc.new_page(); b.on("pageerror", lambda e: errs.append(str(e)))
    b.goto(B + "settings.html"); b.fill("#sync-token", TOKEN); b.click("#sync-connect")
    b.wait_for_function("document.getElementById('sync-st').textContent.includes('Synced')", timeout=10000)
    check("second computer reused the same gist", len(gists) == 1)
    b.goto(B + "days/day-001.html"); b.wait_for_timeout(1500)
    check("second computer sees day-1 block from A", b.locator(".blk").first.is_checked())
    b.locator(".blk").first.uncheck(); b.wait_for_timeout(3500)
    a.goto(B + "days/day-001.html"); a.wait_for_timeout(2500)
    check("uncheck on B reaches A", not a.locator(".blk").first.is_checked())
    b.goto(B + "settings.html"); b.on("dialog", lambda dl: dl.accept()); b.click("#sync-disconnect")
    check("disconnect clears token", b.evaluate("localStorage.getItem('aiml-sync-token')") is None and "Not synced" in b.inner_text("#sync-st"))
    check("only api.github.com got the token", all("api.github.com" in u for _, u, h in calls if h))
    check("no page JS errors", not errs) ; print(errs[:3])
    br.close()
print("%d/%d passed" % (sum(ok), len(ok))); sys.exit(0 if all(ok) else 1)
