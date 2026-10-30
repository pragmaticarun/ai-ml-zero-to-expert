const test = require('node:test'); const assert = require('node:assert/strict');
const M = require('../assets/merge.js'); const S = require('../assets/sync.js'); const C = require('../assets/core.js');
const TOKEN = 'ghp_' + 'T'.repeat(36); // fake token, never a real one

function mockGitHub() {
  const g = { gists: {}, seq: 0, calls: [], mode: null, conflictOnce: 0, extraGists: 0 };
  const res = (status, json, hdr = {}) => ({ ok: status < 300, status, headers: { get: n => hdr[n] ?? hdr[n.toLowerCase()] ?? null }, json: async () => json, text: async () => JSON.stringify(json) });
  g.fetch = async (url, opt = {}) => {
    g.calls.push({ url, method: opt.method || 'GET', headers: opt.headers || {} });
    if (g.mode === 'offline') throw new TypeError('Failed to fetch');
    if (!url.startsWith('https://api.github.com/')) throw new Error('unexpected host ' + url);
    if ((opt.headers || {}).Authorization !== 'Bearer ' + TOKEN || g.mode === '401') return res(401, { message: 'Bad credentials' });
    if (g.mode === 'rate') return res(403, { message: 'rate' }, { 'x-ratelimit-remaining': '0', 'x-ratelimit-reset': '2000000000' });
    const u = new URL(url), p = u.pathname, m = opt.method || 'GET';
    const body = opt.body ? JSON.parse(opt.body) : null;
    const etag = id => '"' + id + '-' + g.gists[id].rev + '"';
    if (p === '/gists' && m === 'GET') {
      const all = [...Array(g.extraGists)].map((_, i) => ({ id: 'other' + i, files: { 'x.txt': {} } })).concat(Object.values(g.gists).map(x => ({ id: x.id, files: { [Object.keys(x.files)[0]]: {} } })));
      const page = +u.searchParams.get('page'), per = +u.searchParams.get('per_page');
      return res(200, all.slice((page - 1) * per, page * per));
    }
    if (p === '/gists' && m === 'POST') { const id = 'g' + (++g.seq); g.gists[id] = { id, rev: 1, public: body.public, description: body.description, files: body.files }; return res(201, { id }); }
    const id = p.split('/')[2], x = g.gists[id];
    if (!x) return res(404, { message: 'Not Found' });
    if (m === 'GET') return res(200, { id, files: Object.fromEntries(Object.entries(x.files).map(([k, v]) => [k, { content: v.content, truncated: false }])) }, { ETag: etag(id) });
    if (m === 'PATCH') {
      if (g.conflictOnce > 0) { g.conflictOnce--; if (g.onConflict) g.onConflict(); return res(412, { message: 'Precondition Failed' }); }
      if (opt.headers['If-Match'] && opt.headers['If-Match'] !== etag(id)) return res(412, {});
      Object.assign(x.files, body.files); x.rev++; return res(200, { id });
    }
    return res(405, {});
  };
  g.remoteItems = id => JSON.parse(g.gists[id].files[S.FILE].content).items;
  return g;
}
function memStore() { const m = new Map(); return { getItem: k => m.has(k) ? m.get(k) : null, setItem: (k, v) => m.set(k, String(v)), removeItem: k => m.delete(k), m }; }
function device(gh, t0 = 1000) {
  const d = { doc: { v: 2, items: {} }, store: memStore(), statuses: [], t: t0, timers: [] };
  d.sync = S.createSync({ merge: M, fetch: gh.fetch, storage: d.store, now: () => d.t, getDoc: () => d.doc, setDoc: x => { d.doc = x; },
    onStatus: s => d.statuses.push(s), setTimeout: (f, ms) => { d.timers.push({ f, ms }); return d.timers.length; }, clearTimeout: () => {} });
  d.edit = fn => { const s = M.unflatten(d.doc.items); fn(s); d.t += 10; d.doc = { v: 2, items: M.stamp(d.doc.items, s, d.t) }; };
  d.state = () => M.unflatten(d.doc.items);
  return d;
}

test('first setup creates a private gist; second device finds and reuses it; merges both ways', async () => {
  const gh = mockGitHub(); gh.extraGists = 150; // marker gist must be found beyond page 1
  const a = device(gh), b = device(gh, 5000);
  a.edit(s => { C.dayRec(s, 1).blocks.fill(true); C.touchDay(s, 1); s.startDate = '2026-11-01'; });
  const r1 = await a.sync.connect(TOKEN); assert.equal(r1.created, true);
  const gist = Object.values(gh.gists)[0];
  assert.equal(gist.public, false); assert.equal(gist.description, 'aiml-course-progress'); assert.ok(gist.files['aiml-course-progress.json']);
  b.edit(s => { C.dayRec(s, 2).notes = 'from B'; });
  const r2 = await b.sync.connect(TOKEN); assert.equal(r2.created, false); assert.equal(r2.id, r1.id);
  assert.ok(gh.calls.some(c => c.url.includes('page=2')), 'paginated');
  assert.equal(b.state().days[1].blocks.every(Boolean), true); assert.equal(b.state().startDate, '2026-11-01');
  await a.sync.syncNow(); assert.equal(a.state().days[2].notes, 'from B');
  assert.equal(a.sync.status().state, 'synced'); assert.equal(Object.keys(gh.gists).length, 1);
});
test('connect to an existing gist id; 404 asks to recreate; recreate works', async () => {
  const gh = mockGitHub(); const a = device(gh); await a.sync.connect(TOKEN); const id = a.sync.gistId();
  const b = device(gh); await b.sync.connect(TOKEN, id); assert.equal(b.sync.gistId(), id);
  delete gh.gists[id];
  await a.sync.syncNow(); assert.equal(a.sync.status().state, 'error'); assert.equal(a.sync.status().need, 'gist');
  await a.sync.recreate(); assert.notEqual(a.sync.gistId(), id); assert.equal(a.sync.status().state, 'synced');
  const c = device(gh); await assert.rejects(c.sync.connect(TOKEN, 'nope'));
  assert.equal(c.sync.status().need, 'gist');
});
test('401 clears the token and asks for it again', async () => {
  const gh = mockGitHub(); const a = device(gh); await a.sync.connect(TOKEN);
  gh.mode = '401'; await a.sync.syncNow();
  assert.equal(a.sync.status().state, 'error'); assert.equal(a.sync.status().need, 'token'); assert.equal(a.sync.hasToken(), false);
  assert.equal(a.store.getItem(S.TOKEN_KEY), null);
  await assert.rejects(device(gh).sync.connect('not a token!'));
});
test('rate limit (403) reports and schedules a retry', async () => {
  const gh = mockGitHub(); const a = device(gh); await a.sync.connect(TOKEN);
  gh.mode = 'rate'; await a.sync.syncNow();
  assert.equal(a.sync.status().state, 'error'); assert.match(a.sync.status().msg, /rate limit/); assert.equal(a.sync.hasToken(), true);
  assert.equal(a.timers.at(-1).ms, 300000);
});
test('offline: keeps local changes, retries, then pushes when back online', async () => {
  const gh = mockGitHub(); const a = device(gh); await a.sync.connect(TOKEN); const id = a.sync.gistId();
  gh.mode = 'offline'; a.edit(s => { C.dayRec(s, 9).hours = 5; }); await a.sync.syncNow();
  assert.equal(a.sync.status().state, 'offline'); assert.equal(a.state().days[9].hours, 5);
  gh.mode = null; await a.timers.at(-1).f();
  assert.equal(a.sync.status().state, 'synced'); assert.equal(gh.remoteItems(id)['day:9:hours'].val, 5);
});
test('412/409 conflict: re-fetch, merge, retry without clobbering the other write', async () => {
  const gh = mockGitHub(); const a = device(gh); const b = device(gh, 9000);
  await a.sync.connect(TOKEN); await b.sync.connect(TOKEN); const id = a.sync.gistId();
  a.edit(s => { C.dayRec(s, 3).notes = 'A wrote'; });
  // while A is pushing, B sneaks in a write -> A's PATCH gets 412
  gh.conflictOnce = 1; gh.onConflict = () => { const x = gh.gists[id]; const items = gh.remoteItems(id);
    items['day:4:notes'] = { u: 99999, val: 'B wrote' }; x.files[S.FILE] = { content: JSON.stringify({ v: 2, items }) }; x.rev++; };
  await a.sync.syncNow();
  const r = gh.remoteItems(id); assert.equal(r['day:3:notes'].val, 'A wrote'); assert.equal(r['day:4:notes'].val, 'B wrote');
  assert.equal(a.state().days[4].notes, 'B wrote'); assert.equal(a.sync.status().state, 'synced');
  assert.ok(gh.calls.filter(c => c.method === 'PATCH').every(c => c.headers['If-Match']), 'If-Match sent');
});
test('uncheck + flashcards across devices end-to-end; no-op sync does not PATCH', async () => {
  const gh = mockGitHub(); const a = device(gh); const b = device(gh, 1000);
  a.edit(s => { C.dayRec(s, 1).blocks.fill(true); C.touchDay(s, 1); s.cards.k = C.sm2(null, 4, '2026-11-01'); });
  await a.sync.connect(TOKEN); await b.sync.connect(TOKEN);
  b.t = a.t + 100; b.edit(s => { C.dayRec(s, 1).blocks[0] = false; C.touchDay(s, 1); s.cards.k = C.sm2(s.cards.k, 5, '2026-11-02'); });
  await b.sync.syncNow(); await a.sync.syncNow();
  assert.equal(C.isComplete(a.state().days[1]), false); assert.equal(a.state().cards.k.reps, 2);
  const n = gh.calls.filter(c => c.method === 'PATCH').length; await a.sync.syncNow();
  assert.equal(gh.calls.filter(c => c.method === 'PATCH').length, n);
});
test('token goes only to api.github.com, never into the gist, status text or URLs; disconnect clears it', async () => {
  const gh = mockGitHub(); const a = device(gh); await a.sync.connect(TOKEN); await a.sync.syncNow();
  for (const c of gh.calls) { assert.ok(c.url.startsWith('https://api.github.com/')); assert.ok(!c.url.includes(TOKEN)); }
  assert.ok(!JSON.stringify(gh.gists).includes(TOKEN)); assert.ok(!JSON.stringify(a.statuses).includes(TOKEN)); assert.ok(!JSON.stringify(a.doc).includes(TOKEN));
  a.sync.disconnect(); assert.equal(a.store.getItem(S.TOKEN_KEY), null); assert.equal(a.store.getItem(S.GIST_KEY), null); assert.equal(a.sync.status().state, 'off');
  assert.ok(Object.keys(gh.gists).length === 1, 'disconnect never deletes the gist');
});
test('schedule() debounces about 2.5 s', async () => {
  const gh = mockGitHub(); const a = device(gh); await a.sync.connect(TOKEN);
  a.sync.schedule(); assert.equal(a.timers.at(-1).ms, S.DEBOUNCE); assert.equal(a.sync.status().state, 'pending');
});
