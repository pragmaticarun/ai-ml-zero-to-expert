/* Pure, dependency-free sync data model (browser: window.Merge; node: module.exports).
   Document v2: { v: 2, items: { key: { u: <ms timestamp>, val: <JSON value or null = tombstone> } } }
   Keys: day:N:b:I (block I checked), day:N:manual, day:N:hours, day:N:notes, day:N:completedAt,
         quiz:ID:a:K (answer text), quiz:ID:m:K (right/wrong), card:ID (SM-2 state),
         set:startDate, set:theme, set:last
   Merge = last-writer-wins per key; deletions are tombstones (val null) with their own timestamp. */
(function (root) {
  'use strict';
  var MIGRATED_U = 1; // migrated/old-import data: older than any real edit, newer than "absent"

  function flatten(s) {
    var o = {}, k, i;
    s = s || {};
    for (k in (s.days || {})) {
      var r = s.days[k] || {};
      var b = Array.isArray(r.blocks) ? r.blocks : [];
      for (i = 0; i < 5; i++) o['day:' + k + ':b:' + i] = !!b[i];
      o['day:' + k + ':manual'] = r.manual === true;
      o['day:' + k + ':hours'] = isFinite(parseFloat(r.hours)) ? +r.hours : 0;
      o['day:' + k + ':notes'] = typeof r.notes === 'string' ? r.notes : '';
      o['day:' + k + ':completedAt'] = typeof r.completedAt === 'string' ? r.completedAt : '';
    }
    for (k in (s.quizzes || {})) {
      var q = s.quizzes[k] || {}, j;
      for (j in (q.answers || {})) if (q.answers[j] !== '' && q.answers[j] != null) o['quiz:' + k + ':a:' + j] = String(q.answers[j]);
      for (j in (q.marks || {})) if (q.marks[j]) o['quiz:' + k + ':m:' + j] = q.marks[j];
    }
    for (k in (s.cards || {})) if (s.cards[k]) o['card:' + k] = s.cards[k];
    if (s.startDate) o['set:startDate'] = s.startDate;
    if (s.theme) o['set:theme'] = s.theme;
    if (s.last) o['set:last'] = +s.last;
    return o;
  }
  function unflatten(items) {
    var s = { v: 1, startDate: '', days: {}, quizzes: {}, cards: {}, theme: '', last: 0 };
    for (var key in (items || {})) {
      var it = items[key]; if (!it || it.val === null || it.val === undefined) continue;
      var p = key.split(':'), v = it.val;
      if (p[0] === 'day') {
        var r = s.days[p[1]] || (s.days[p[1]] = { blocks: [false, false, false, false, false], manual: false, notes: '', hours: 0, completedAt: '' });
        if (p[2] === 'b') r.blocks[+p[3]] = !!v; else if (p[2] in r) r[p[2]] = v;
      } else if (p[0] === 'quiz') {
        var id = p.slice(1, -2).join(':'), kind = p[p.length - 2], qk = p[p.length - 1];
        var q = s.quizzes[id] || (s.quizzes[id] = { answers: {}, marks: {} });
        (kind === 'a' ? q.answers : q.marks)[qk] = v;
      } else if (p[0] === 'card') s.cards[p.slice(1).join(':')] = v;
      else if (p[0] === 'set' && p[1] in s) s[p[1]] = v;
    }
    return s;
  }
  function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
  /* New items after the in-memory state s was edited at time now: changed keys get u=now, removed keys become tombstones. */
  function stamp(prev, s, now) {
    prev = prev || {}; var out = {}, flat = flatten(s), k;
    for (k in prev) out[k] = prev[k];
    for (k in flat) if (!prev[k] || !same(prev[k].val, flat[k])) out[k] = { u: now, val: flat[k] };
    for (k in prev) if (!(k in flat) && prev[k].val !== null) out[k] = { u: now, val: null };
    return out;
  }
  function newer(a, b) { // true if a beats b
    if (!b) return true; if (!a) return false;
    if (a.u !== b.u) return a.u > b.u;
    return JSON.stringify(a.val) > JSON.stringify(b.val); // deterministic tie-break
  }
  function merge(a, b) {
    var out = {}, k; a = a || {}; b = b || {};
    for (k in a) out[k] = a[k];
    for (k in b) if (newer(b[k], out[k])) out[k] = b[k];
    return out;
  }
  function itemsEqual(a, b) {
    var k; a = a || {}; b = b || {};
    for (k in a) if (!b[k] || b[k].u !== a[k].u || !same(a[k].val, b[k].val)) return false;
    for (k in b) if (!a[k]) return false;
    return true;
  }
  function migrate(v1) { return { v: 2, items: stamp({}, v1 || {}, MIGRATED_U) }; }
  /* Accepts a v2 doc, a v2 export ({v:2, items, ...}), or an old v1 state/export. Returns a v2 doc. */
  function toDoc(obj) {
    if (obj && obj.v === 2 && obj.items && typeof obj.items === 'object' && !Array.isArray(obj.items)) {
      var items = {};
      for (var k in obj.items) { var it = obj.items[k]; if (it && typeof it.u === 'number' && 'val' in it) items[k] = { u: it.u, val: it.val }; }
      return { v: 2, items: items };
    }
    return migrate(obj);
  }
  var api = { flatten: flatten, unflatten: unflatten, stamp: stamp, merge: merge, itemsEqual: itemsEqual, migrate: migrate, toDoc: toDoc, MIGRATED_U: MIGRATED_U };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Merge = api;
})(this);
