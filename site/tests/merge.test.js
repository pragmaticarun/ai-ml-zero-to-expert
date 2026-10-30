const test = require('node:test'); const assert = require('node:assert/strict');
const M = require('../assets/merge.js'); const C = require('../assets/core.js');
function dev(items) { return { items: items || {}, s() { return M.unflatten(this.items); }, edit(fn, t) { const s = this.s(); fn(s); this.items = M.stamp(this.items, s, t); } };
}
const blocks = (s, n) => C.dayRec(s, n).blocks;

test('two devices edit different days', () => {
  const a = dev(), b = dev();
  a.edit(s => { blocks(s, 1)[0] = true; C.dayRec(s, 1).notes = 'A'; }, 100);
  b.edit(s => { blocks(s, 2)[3] = true; C.dayRec(s, 2).hours = 4; }, 200);
  const m = M.unflatten(M.merge(a.items, b.items));
  assert.equal(m.days[1].blocks[0], true); assert.equal(m.days[1].notes, 'A');
  assert.equal(m.days[2].blocks[3], true); assert.equal(m.days[2].hours, 4);
  assert.ok(M.itemsEqual(M.merge(a.items, b.items), M.merge(b.items, a.items)), 'commutative');
});
test('same day edited on both devices: per-field last writer wins', () => {
  const base = dev(); base.edit(s => { C.dayRec(s, 5).notes = 'base'; }, 10);
  const a = dev({ ...base.items }), b = dev({ ...base.items });
  a.edit(s => { C.dayRec(s, 5).notes = 'from A'; blocks(s, 5)[1] = true; }, 100);
  b.edit(s => { C.dayRec(s, 5).notes = 'from B'; C.dayRec(s, 5).hours = 3; }, 200);
  const m = M.unflatten(M.merge(a.items, b.items)).days[5];
  assert.equal(m.notes, 'from B'); assert.equal(m.blocks[1], true); assert.equal(m.hours, 3);
});
test('uncheck on one device is respected', () => {
  const a = dev(); a.edit(s => { blocks(s, 7).fill(true); C.touchDay(s, 7, new Date('2026-11-07T06:00:00Z')); }, 100);
  const b = dev(M.merge({}, a.items));
  b.edit(s => { blocks(s, 7)[2] = false; C.touchDay(s, 7); }, 300);
  const m = M.unflatten(M.merge(a.items, b.items));
  assert.equal(m.days[7].blocks[2], false); assert.equal(C.isComplete(m.days[7]), false); assert.equal(m.days[7].completedAt, '');
  // deleted quiz mark -> tombstone wins over older value
  const q = dev(); q.edit(s => { s.quizzes.w01 = { answers: {}, marks: { 1: 'right' } }; }, 100);
  const q2 = dev({ ...q.items }); q2.edit(s => { delete s.quizzes.w01.marks[1]; }, 200);
  assert.equal(q2.items['quiz:w01:m:1'].val, null);
  assert.deepEqual(M.unflatten(M.merge(q.items, q2.items)).quizzes.w01 || { marks: {} }, M.unflatten(q2.items).quizzes.w01 || { marks: {} });
  assert.equal((M.unflatten(M.merge(q.items, q2.items)).quizzes.w01 || { marks: {} }).marks[1], undefined);
});
test('flashcard reviewed on both devices: latest review state wins per card', () => {
  const a = dev(), b = dev();
  a.edit(s => { s.cards.c1 = C.sm2(null, 4, '2026-11-01'); s.cards.c2 = C.sm2(null, 0, '2026-11-01'); }, 100);
  b.edit(s => { s.cards.c1 = C.sm2(C.sm2(null, 4, '2026-11-01'), 5, '2026-11-02'); }, 200);
  const m = M.unflatten(M.merge(a.items, b.items));
  assert.equal(m.cards.c1.reps, 2); assert.equal(m.cards.c2.reps, 0); assert.deepEqual(m.cards.c2, M.unflatten(a.items).cards.c2);
});
test('empty remote / empty local', () => {
  const a = dev(); a.edit(s => { s.startDate = '2026-11-01'; blocks(s, 1)[0] = true; }, 5);
  assert.ok(M.itemsEqual(M.merge(a.items, {}), a.items)); assert.ok(M.itemsEqual(M.merge({}, a.items), a.items));
  assert.deepEqual(M.merge({}, {}), {});
});
test('migration of v1 localStorage data keeps everything; real edits beat migrated values', () => {
  const v1 = { v: 1, startDate: '2026-10-01', theme: 'dark', last: 3,
    days: { 1: { blocks: [true, true, true, true, true], manual: false, notes: 'n1', hours: 6, completedAt: '2026-10-02' }, 3: { blocks: [false, false, false, false, false], manual: true, notes: '', hours: 1.5, completedAt: '2026-10-04' } },
    quizzes: { 'w01': { answers: { 1: 'x' }, marks: { 1: 'right', 2: 'wrong' } } }, cards: { 'k1': { ef: 2.5, reps: 1, interval: 1, due: '2026-10-05', lapses: 0, last: '2026-10-04' } } };
  const doc = M.toDoc(JSON.parse(JSON.stringify(v1)));
  assert.equal(doc.v, 2); assert.deepEqual(M.unflatten(doc.items), v1);
  const other = dev(); other.edit(s => { C.dayRec(s, 1).notes = 'newer'; }, 1000);
  assert.equal(M.unflatten(M.merge(doc.items, other.items)).days[1].notes, 'newer');
  assert.equal(M.unflatten(M.merge(doc.items, other.items)).days[3].hours, 1.5);
  // v2 export round-trip, and garbage items dropped
  const exp = { v: 2, items: { ...doc.items, bad: { nope: 1 } } };
  assert.deepEqual(M.toDoc(exp).items, doc.items);
});
test('stamp only bumps changed keys', () => {
  const a = dev(); a.edit(s => { blocks(s, 1)[0] = true; }, 10); a.edit(s => { blocks(s, 1)[1] = true; }, 20);
  assert.equal(a.items['day:1:b:0'].u, 10); assert.equal(a.items['day:1:b:1'].u, 20);
});
