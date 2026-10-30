/* Cross-device sync via a PRIVATE (secret) GitHub Gist. No dependencies; fetch/storage/timers are injected so it is testable in node.
   The token lives only in this browser's localStorage under TOKEN_KEY and is only ever sent to https://api.github.com. */
(function (root) {
  'use strict';
  var API = 'https://api.github.com', FILE = 'aiml-course-progress.json', DESC = 'aiml-course-progress';
  var TOKEN_KEY = 'aiml-sync-token', GIST_KEY = 'aiml-sync-gist', DEBOUNCE = 2500, RETRIES = 4;

  function createSync(o) {
    var M = o.merge, fetchFn = o.fetch, store = o.storage, now = o.now || Date.now;
    var setT = o.setTimeout || setTimeout, clearT = o.clearTimeout || clearTimeout;
    var st = { state: 'off', msg: 'Not connected', at: 0, need: null }, running = null, again = false, timer = null, retryTimer = null;

    function get(k) { try { return store.getItem(k) || ''; } catch (e) { return ''; } }
    function put(k, v) { try { if (v) store.setItem(k, v); else store.removeItem(k); } catch (e) { /* ignore */ } }
    function connected() { return !!(get(TOKEN_KEY) && get(GIST_KEY)); }
    function set(state, msg, need) { st = { state: state, msg: msg, at: now(), need: need || null }; if (o.onStatus) o.onStatus(st); return st; }
    function fail(kind, msg, need) { var e = new Error(msg); e.kind = kind; e.need = need; return e; }

    function req(method, path, body, extra) {
      var h = { 'Accept': 'application/vnd.github+json', 'Authorization': 'Bearer ' + get(TOKEN_KEY), 'X-GitHub-Api-Version': '2022-11-28' };
      for (var k in (extra || {})) h[k] = extra[k];
      if (body !== undefined) h['Content-Type'] = 'application/json';
      return Promise.resolve().then(function () {
        return fetchFn(API + path, { method: method, headers: h, body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store' });
      }).then(function (r) {
        if (r.ok) return r.json().then(function (j) { return { status: r.status, etag: r.headers.get('ETag'), json: j }; });
        throw httpError(r);
      }, function (e) { if (e && e.kind) throw e; throw fail('offline', 'Offline or GitHub unreachable'); });
    }
    function httpError(r) {
      var s = r.status, h = function (n) { return r.headers.get(n); };
      if (s === 401) { put(TOKEN_KEY, ''); return fail('auth', 'GitHub rejected the token (401). Paste a valid token again.', 'token'); }
      if (s === 429 || (s === 403 && h('x-ratelimit-remaining') === '0')) {
        var reset = +h('x-ratelimit-reset'); return fail('rate', 'GitHub rate limit reached' + (reset ? '; retry after ' + new Date(reset * 1000).toLocaleTimeString() : '') + '.');
      }
      if (s === 403) return fail('forbidden', 'GitHub refused (403). Does the token have the "gist" scope?', 'token');
      if (s === 404) return fail('missing', 'Sync gist not found (deleted, or token from another account). Recreate it or connect another gist id.', 'gist');
      if (s === 409 || s === 412) return fail('conflict', 'Conflict (' + s + ')');
      return fail('http', 'GitHub error ' + s + '.');
    }
    function readGist(id) {
      return req('GET', '/gists/' + encodeURIComponent(id)).then(function (r) {
        var f = r.json.files && r.json.files[FILE];
        if (!f) return { etag: r.etag, doc: { v: 2, items: {} } };
        var content = f.truncated && f.raw_url
          ? fetchFn(f.raw_url, { cache: 'no-store' }).then(function (x) { return x.text(); }) // raw URL needs no token; none is sent
          : Promise.resolve(f.content || '');
        return content.then(function (txt) {
          var obj; try { obj = txt ? JSON.parse(txt) : null; } catch (e) { throw fail('corrupt', 'Remote progress file is not valid JSON; not overwriting it.'); }
          return { etag: r.etag, doc: obj ? M.toDoc(obj) : { v: 2, items: {} } };
        });
      });
    }
    function body(doc) { var f = {}; f[FILE] = { content: JSON.stringify({ v: 2, app: 'aiml-course', items: doc.items }) }; return { files: f }; }

    function findOrCreate() {
      var page = 1;
      function next() {
        return req('GET', '/gists?per_page=100&page=' + page).then(function (r) {
          var hit = (r.json || []).filter(function (g) { return g.files && g.files[FILE]; })[0];
          if (hit) return { id: hit.id, created: false };
          if ((r.json || []).length === 100 && page < 5) { page++; return next(); }
          var b = body(o.getDoc()); b.description = DESC; b.public = false;
          return req('POST', '/gists', b).then(function (c) { return { id: c.json.id, created: true }; });
        });
      }
      return next();
    }
    function connect(token, gistId) {
      token = String(token || '').trim();
      if (!/^[A-Za-z0-9_]{20,255}$/.test(token)) return Promise.reject(fail('auth', 'That does not look like a GitHub token.', 'token'));
      put(TOKEN_KEY, token); set('syncing', 'Connecting…');
      var p = gistId ? readGist(String(gistId).trim()).then(function () { return { id: String(gistId).trim(), created: false }; }) : findOrCreate();
      return p.then(function (res) { put(GIST_KEY, res.id); return syncNow().then(function () { return res; }); },
        function (e) { set(e.kind === 'offline' ? 'offline' : 'error', e.message, e.need); throw e; });
    }
    function recreate() { put(GIST_KEY, ''); return connect(get(TOKEN_KEY)); }
    function disconnect() { clearT(timer); clearT(retryTimer); put(TOKEN_KEY, ''); put(GIST_KEY, ''); set('off', 'Not connected'); }

    function once() {
      var id = get(GIST_KEY), attempt = 0;
      function go() {
        return readGist(id).then(function (rem) {
          var local = o.getDoc(), merged = M.merge(local.items, rem.doc.items);
          if (!M.itemsEqual(merged, local.items)) o.setDoc({ v: 2, items: merged });
          if (M.itemsEqual(merged, rem.doc.items)) return;
          var hdr = rem.etag ? { 'If-Match': rem.etag } : {};
          return req('PATCH', '/gists/' + encodeURIComponent(id), body({ items: merged }), hdr).catch(function (e) {
            if (e.kind === 'conflict' && ++attempt < RETRIES) return go(); // someone else wrote: re-fetch, re-merge, retry
            throw e;
          });
        });
      }
      return go();
    }
    function syncNow() {
      if (!get(TOKEN_KEY)) return Promise.resolve(set('off', 'Not connected', 'token'));
      if (!get(GIST_KEY)) return Promise.resolve(set('error', 'No sync gist set. Connect again.', 'gist'));
      if (running) { again = true; return running; }
      clearT(retryTimer); set('syncing', 'Syncing…');
      running = once().then(function () { set('synced', 'Synced'); }, function (e) {
        if (e.kind === 'offline' || e.kind === 'rate' || e.kind === 'http' || e.kind === 'conflict') {
          set(e.kind === 'offline' ? 'offline' : 'error', e.kind === 'offline' ? 'Offline: changes saved here, will sync when online' : e.message);
          retryTimer = setT(syncNow, e.kind === 'rate' ? 300000 : 30000);
        } else set('error', e.message, e.need);
      }).then(function () { running = null; if (again) { again = false; return syncNow(); } });
      return running;
    }
    function schedule() { if (!connected()) return; clearT(timer); if (st.state !== 'syncing') set('pending', 'Changes pending…'); timer = setT(syncNow, DEBOUNCE); }
    function init() { if (connected()) return syncNow(); set('off', 'Not connected', get(TOKEN_KEY) ? 'gist' : null); return Promise.resolve(); }
    return { connect: connect, recreate: recreate, disconnect: disconnect, syncNow: syncNow, schedule: schedule, init: init,
      connected: connected, hasToken: function () { return !!get(TOKEN_KEY); }, gistId: function () { return get(GIST_KEY); }, status: function () { return st; } };
  }
  var api = { createSync: createSync, FILE: FILE, DESC: DESC, TOKEN_KEY: TOKEN_KEY, GIST_KEY: GIST_KEY, DEBOUNCE: DEBOUNCE };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Sync = api;
})(this);
