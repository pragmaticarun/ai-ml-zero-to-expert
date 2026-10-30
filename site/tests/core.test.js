// Run: node --test site/tests/
const test = require('node:test'); const assert = require('node:assert/strict');
const C = require('../assets/core.js');
const at = (iso, hh = '12:00') => new Date(iso + 'T' + hh + ':00+05:30');

test('todayISO uses Asia/Kolkata', () => {
  assert.equal(C.todayISO(new Date('2026-10-29T19:00:00Z')), '2026-10-30'); // 00:30 IST next day
  assert.equal(C.todayISO(new Date('2026-10-29T18:00:00Z')), '2026-10-29'); // 23:30 IST
});
test('day completes when all 5 blocks are checked, or manually', () => {
  const s = C.emptyState(); const r = C.dayRec(s, 1);
  r.blocks = [true, true, true, true, false]; C.touchDay(s, 1, at('2026-11-01'));
  assert.equal(C.isComplete(r), false); assert.equal(r.completedAt, '');
  r.blocks[4] = true; C.touchDay(s, 1, at('2026-11-01'));
  assert.equal(C.isComplete(r), true); assert.equal(r.completedAt, '2026-11-01');
  r.blocks[0] = false; C.touchDay(s, 1, at('2026-11-02')); assert.equal(r.completedAt, '');
  r.manual = true; C.touchDay(s, 1, at('2026-11-02')); assert.equal(r.completedAt, '2026-11-02');
  assert.deepEqual(C.completedDays(s), [1]); assert.equal(C.nextDay(s), 2);
});
test('hours, streak, schedule delta, statuses', () => {
  const s = C.emptyState(); s.startDate = '2026-11-01';
  for (const [n, d] of [[1, '2026-11-01'], [2, '2026-11-02'], [3, '2026-11-03']]) { const r = C.dayRec(s, n); r.manual = true; r.hours = 5.5; C.touchDay(s, n, at(d)); }
  assert.equal(C.hoursLogged(s), 16.5);
  assert.equal(C.streak(s, at('2026-11-03')), 3);
  assert.equal(C.streak(s, at('2026-11-04')), 3); // yesterday counts
  assert.equal(C.streak(s, at('2026-11-05')), 0);
  assert.equal(C.scheduleDelta(s, at('2026-11-03')), 0);
  assert.equal(C.scheduleDelta(s, at('2026-11-05')), -2);
  assert.equal(C.status(s, 4, at('2026-11-05')), 'behind');
  assert.equal(C.status(s, 5, at('2026-11-05')), 'todo');
  assert.equal(C.status(s, 3, at('2026-11-05')), 'done');
  C.dayRec(s, 6).blocks[0] = true; assert.equal(C.status(s, 6, at('2026-11-05')), 'partial');
  assert.equal(C.plannedDate(s, 180), '2027-04-29');
  const e = C.estimateFinish(s, at('2026-11-03')); assert.equal(e.basis, 'pace'); assert.equal(e.date, '2027-04-29') // pace 1 day/day => same as plan;
});
test('reviews due at +1/+3/+7/+21', () => {
  const s = C.emptyState(); const r = C.dayRec(s, 10); r.manual = true; C.touchDay(s, 10, at('2026-11-01'));
  assert.deepEqual(C.reviewsDue(s, at('2026-11-02')), [{ day: 10, gap: 1 }]);
  assert.deepEqual(C.reviewsDue(s, at('2026-11-03')), []);
  assert.deepEqual(C.reviewsDue(s, at('2026-11-22')), [{ day: 10, gap: 21 }]);
});
test('SM-2 scheduler', () => {
  let c = C.sm2(null, 4, '2026-11-01'); assert.equal(c.interval, 1); assert.equal(c.due, '2026-11-02'); assert.equal(c.ef, 2.5);
  c = C.sm2(c, 4, '2026-11-02'); assert.equal(c.interval, 6); assert.equal(c.due, '2026-11-08');
  c = C.sm2(c, 5, '2026-11-08'); assert.equal(c.interval, Math.round(Math.round(6 * 2.5) * 1.3)); assert.ok(c.ef > 2.5);
  const a = C.sm2(c, 0, '2026-12-01'); assert.equal(a.reps, 0); assert.equal(a.interval, 1); assert.equal(a.due, '2026-12-01'); assert.ok(a.ef < c.ef); assert.equal(a.lapses, 1);
  let low = null; for (let i = 0; i < 20; i++) low = C.sm2(low, 0, '2026-11-01'); assert.equal(low.ef, 1.3);
  assert.ok(C.cardDue(undefined, '2026-11-01')); assert.ok(!C.cardDue({ due: '2026-11-02' }, '2026-11-01'));
});
test('import validation and quiz score', () => {
  assert.throws(() => C.validateImport([])); assert.throws(() => C.validateImport({ days: { 999: {} } }));
  const s = C.validateImport({ days: { 3: { manual: true } }, startDate: 'bogus' }); assert.equal(s.startDate, ''); assert.deepEqual(s.cards, {});
  assert.deepEqual(C.quizScore({ marks: { 1: 'right', 2: 'wrong', 3: 'right' } }), { right: 2, wrong: 1 });
});
