/* Pure progress / scheduling logic. No DOM. Used by app.js and by tests (node). */
(function (root) {
  'use strict';
  var TZ = 'Asia/Kolkata';
  var BLOCKS = 5, TOTAL_DAYS = 180;
  function todayISO(now) {
    // YYYY-MM-DD in Asia/Kolkata
    return new Intl.DateTimeFormat('en-CA', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit' }).format(now || new Date());
  }
  function addDays(iso, n) {
    var d = new Date(iso + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + n);
    return d.toISOString().slice(0, 10);
  }
  function diffDays(a, b) { // b - a in days
    return Math.round((Date.parse(b + 'T00:00:00Z') - Date.parse(a + 'T00:00:00Z')) / 86400000);
  }
  function emptyState() { return { v: 1, startDate: '', days: {}, quizzes: {}, cards: {}, theme: '', last: 0 }; }
  function normalize(s) {
    var e = emptyState();
    if (!s || typeof s !== 'object') return e;
    for (var k in e) if (s[k] === undefined) s[k] = e[k];
    ['days', 'quizzes', 'cards'].forEach(function (k) { if (typeof s[k] !== 'object' || s[k] === null || Array.isArray(s[k])) s[k] = {}; });
    if (typeof s.startDate !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(s.startDate)) s.startDate = '';
    return s;
  }
  function dayRec(s, n) {
    var r = s.days[n];
    if (!r) r = s.days[n] = { blocks: [false, false, false, false, false], manual: false, notes: '', hours: 0, completedAt: '' };
    if (!Array.isArray(r.blocks)) r.blocks = [false, false, false, false, false];
    return r;
  }
  function isComplete(r) { return !!r && (r.manual === true || (Array.isArray(r.blocks) && r.blocks.length >= BLOCKS && r.blocks.slice(0, BLOCKS).every(Boolean))); }
  function isStarted(r) { return !!r && (isComplete(r) || (r.blocks || []).some(Boolean) || (+r.hours > 0) || !!r.notes); }
  /* Call after any change to a day; stamps/clears completedAt. */
  function touchDay(s, n, now) {
    var r = dayRec(s, n);
    if (isComplete(r)) { if (!r.completedAt) r.completedAt = todayISO(now); } else r.completedAt = '';
    return r;
  }
  function completedDays(s) { var out = []; for (var k in s.days) if (isComplete(s.days[k])) out.push(+k); return out.sort(function (a, b) { return a - b; }); }
  function hoursLogged(s) { var h = 0; for (var k in s.days) { var x = parseFloat(s.days[k].hours); if (isFinite(x) && x > 0) h += x; } return Math.round(h * 100) / 100; }
  function nextDay(s) { for (var n = 1; n <= TOTAL_DAYS; n++) if (!isComplete(s.days[n])) return n; return null; }
  /* Streak: consecutive calendar days (Asia/Kolkata) with >=1 completion, ending today, or yesterday if today has none yet. */
  function streak(s, now) {
    var set = {}; for (var k in s.days) { var r = s.days[k]; if (isComplete(r) && r.completedAt) set[r.completedAt] = true; }
    var t = todayISO(now), d = set[t] ? t : addDays(t, -1), c = 0;
    while (set[d]) { c++; d = addDays(d, -1); }
    return c;
  }
  function plannedDate(s, n) { return s.startDate ? addDays(s.startDate, n - 1) : ''; }
  /* 'done' | 'behind' | 'partial' | 'todo' */
  function status(s, n, now) {
    var r = s.days[n];
    if (isComplete(r)) return 'done';
    var p = plannedDate(s, n);
    if (p && diffDays(p, todayISO(now)) > 0) return 'behind';
    return isStarted(r) ? 'partial' : 'todo';
  }
  /* Positive = ahead of schedule by N days; negative = behind. null if no start date. */
  function scheduleDelta(s, now) {
    if (!s.startDate) return null;
    var expected = Math.min(TOTAL_DAYS, Math.max(0, diffDays(s.startDate, todayISO(now)) + 1)); // days that should be done by end of today
    return completedDays(s).length - expected;
  }
  /* Estimated finish: from observed pace since start date (or first completion). */
  function estimateFinish(s, now) {
    var done = completedDays(s).length, t = todayISO(now);
    if (done >= TOTAL_DAYS) return { date: t, basis: 'finished' };
    var begin = s.startDate;
    if (!begin) { var ds = []; for (var k in s.days) if (s.days[k].completedAt) ds.push(s.days[k].completedAt); ds.sort(); begin = ds[0] || ''; }
    if (!begin || done === 0) return { date: s.startDate ? addDays(s.startDate, TOTAL_DAYS - 1) : '', basis: s.startDate ? 'plan' : 'none' };
    var elapsed = Math.max(1, diffDays(begin, t) + 1), pace = done / elapsed; // days per calendar day
    return { date: addDays(t, Math.ceil((TOTAL_DAYS - done) / pace)), basis: 'pace', pace: pace };
  }
  /* Reviews due today: completed day d is reviewed on completedAt + 1, 3, 7, 21. */
  function reviewsDue(s, now) {
    var t = todayISO(now), out = [];
    for (var k in s.days) {
      var r = s.days[k]; if (!isComplete(r) || !r.completedAt) continue;
      var g = diffDays(r.completedAt, t);
      if ([1, 3, 7, 21].indexOf(g) >= 0) out.push({ day: +k, gap: g });
    }
    return out.sort(function (a, b) { return a.day - b.day; });
  }
  /* SM-2 style scheduler. q in 0..5 (Again 0, Hard 3, Good 4, Easy 5). */
  function sm2(card, q, today) {
    var c = { ef: 2.5, reps: 0, interval: 0, due: today, lapses: 0 };
    if (card) for (var k in card) c[k] = card[k];
    if (q < 3) { c.reps = 0; c.interval = 1; c.lapses = (c.lapses || 0) + 1; }
    else {
      c.reps += 1;
      if (c.reps === 1) c.interval = 1;
      else if (c.reps === 2) c.interval = 6;
      else c.interval = Math.round(c.interval * c.ef);
      if (q === 3) c.interval = Math.max(1, Math.round(c.interval * 0.8));
      if (q === 5) c.interval = Math.round(c.interval * 1.3) || 1;
    }
    c.ef = Math.max(1.3, c.ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)));
    c.ef = Math.round(c.ef * 1000) / 1000;
    c.due = addDays(today, q < 3 ? 0 : c.interval); // Again: due again today
    c.last = today;
    return c;
  }
  function cardDue(st, today) { return !st || !st.due || st.due <= today; }
  function quizScore(qs) { var r = 0, w = 0; if (qs && qs.marks) for (var k in qs.marks) { if (qs.marks[k] === 'right') r++; else if (qs.marks[k] === 'wrong') w++; } return { right: r, wrong: w }; }
  function validateImport(obj) {
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) throw new Error('Not a JSON object');
    if (obj.days && typeof obj.days !== 'object') throw new Error('days must be an object');
    for (var k in (obj.days || {})) if (!/^\d+$/.test(k) || +k < 1 || +k > TOTAL_DAYS) throw new Error('bad day key ' + k);
    return normalize(obj);
  }
  var api = { TZ: TZ, TOTAL_DAYS: TOTAL_DAYS, todayISO: todayISO, addDays: addDays, diffDays: diffDays, emptyState: emptyState, normalize: normalize,
    dayRec: dayRec, isComplete: isComplete, isStarted: isStarted, touchDay: touchDay, completedDays: completedDays, hoursLogged: hoursLogged,
    nextDay: nextDay, streak: streak, plannedDate: plannedDate, status: status, scheduleDelta: scheduleDelta, estimateFinish: estimateFinish,
    reviewsDue: reviewsDue, sm2: sm2, cardDue: cardDue, quizScore: quizScore, validateImport: validateImport };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Core = api;
})(this);
