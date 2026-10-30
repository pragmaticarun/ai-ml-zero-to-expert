/* DOM layer. Data: window.COURSE (data.js); logic: window.Core (core.js). */
(function () {
  'use strict';
  var C = window.Core, D = window.COURSE, KEY = 'aiml-course-v1';
  var body = document.body, ROOT = body.dataset.root || '';
  function load() { try { return C.normalize(JSON.parse(localStorage.getItem(KEY) || 'null')); } catch (e) { return C.emptyState(); } }
  var S = load();
  function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) { alert('Could not save progress (storage full or disabled).'); } }
  function $(sel, el) { return (el || document).querySelector(sel); }
  function $$(sel, el) { return Array.prototype.slice.call((el || document).querySelectorAll(sel)); }
  function esc(t) { return String(t).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function dayHref(n) { return ROOT + 'days/day-' + String(n).padStart(3, '0') + '.html'; }
  function fmtDate(iso) { if (!iso) return '—'; var d = new Date(iso + 'T00:00:00Z'); return d.toLocaleDateString('en-IN', { timeZone: 'UTC', day: 'numeric', month: 'short', year: 'numeric', weekday: 'short' }); }
  var dayMeta = {}; D.days.forEach(function (d) { dayMeta[d.n] = d; });

  // theme
  var tb = $('#theme-toggle');
  if (tb) tb.addEventListener('click', function () {
    var cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    S.theme = cur === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = S.theme; save();
  });
  // keyboard: [ / ] or Alt+arrows for prev/next on day & week pages
  document.addEventListener('keydown', function (e) {
    var t = e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;
    var go = null;
    if (e.key === '[' || (e.altKey && e.key === 'ArrowLeft')) go = $('a[rel=prev]') || $('.pn a:first-child');
    if (e.key === ']' || (e.altKey && e.key === 'ArrowRight')) go = $('a[rel=next]') || $('.pn a:last-child');
    if (go && go.href && body.dataset.kind !== 'review') location.href = go.href;
  });
  // status badges anywhere
  function badge(st) { return { done: '✓ done', partial: '… in progress', behind: '! behind', todo: '' }[st] || ''; }
  $$('[data-st]').forEach(function (el) { var st = C.status(S, +el.dataset.st); el.textContent = badge(st); el.className = 'st ' + st; });
  $$('[data-quizst]').forEach(function (el) { var sc = C.quizScore(S.quizzes[el.dataset.quizst]); if (sc.right + sc.wrong) el.textContent = '· ' + sc.right + ' right / ' + sc.wrong + ' wrong'; });

  var kind = body.dataset.kind;
  if (kind === 'home') home();
  if (kind === 'day') day(+body.dataset.day);
  if (kind === 'syllabus') syllabus();
  if (kind === 'review') review();
  if (kind === 'settings') settings();
  $$('.quiz').forEach(quiz);

  function home() {
    var done = C.completedDays(S), nd = C.nextDay(S), meta = nd ? dayMeta[nd] : null;
    var est = C.estimateFinish(S), delta = C.scheduleDelta(S);
    var cont = $('#continue');
    var target = S.last && !C.isComplete(S.days[S.last]) ? S.last : nd;
    if (target) { cont.href = dayHref(target); cont.textContent = 'Continue: Day ' + target + ' — ' + dayMeta[target].title; }
    else { cont.textContent = 'Course complete — review flashcards'; cont.href = ROOT + 'review.html'; }
    var cards = [
      ['Days completed', done.length + ' / 180', Math.round(done.length / 1.8) + '%'],
      ['Hours logged', C.hoursLogged(S) + ' / ~' + D.totalHours, 'from the hours box on each day'],
      ['Current month / week', meta ? 'Month ' + meta.month + ' · Week ' + meta.week : 'Finished', meta ? 'next: Day ' + nd : ''],
      ['Current streak', C.streak(S) + ' day' + (C.streak(S) === 1 ? '' : 's'), 'calendar days with a completed day (Asia/Kolkata)'],
      ['Estimated finish', est.date ? fmtDate(est.date) : 'set a start date or finish a day', est.basis === 'pace' ? 'at your pace so far (' + est.pace.toFixed(2) + ' days/day)' : est.basis === 'plan' ? 'planned (1 day per day)' : ''],
      ['Schedule', delta === null ? 'no start date' : delta === 0 ? 'on schedule' : delta > 0 ? delta + ' day(s) ahead' : (-delta) + ' day(s) behind', delta === null ? 'Settings → Start date' : 'start ' + fmtDate(S.startDate)]
    ];
    $('#stats').innerHTML = cards.map(function (c) { return '<div class="stat"><div class="k">' + esc(c[0]) + '</div><div class="v">' + esc(c[1]) + '</div><div class="s">' + esc(c[2]) + '</div></div>'; }).join('') +
      '<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="180" aria-valuenow="' + done.length + '" aria-label="Course progress"><span style="width:' + (done.length / 1.8) + '%"></span></div>';
    var today = C.todayISO(), html = '';
    for (var m = 1; m <= 6; m++) {
      html += '<div class="month"><h3>Month ' + m + '</h3>';
      var weeks = {};
      D.days.forEach(function (d) { if (d.month === m) (weeks[d.week] = weeks[d.week] || []).push(d); });
      Object.keys(weeks).sort(function (a, b) { return a - b; }).forEach(function (w) {
        html += '<div class="wk"><a class="wl" href="weeks/week-' + String(w).padStart(2, '0') + '.html">W' + w + '</a>';
        weeks[w].forEach(function (d) {
          var st = C.status(S, d.n), p = C.plannedDate(S, d.n);
          html += '<a class="cell ' + st + (p === today ? ' today' : '') + '" href="' + dayHref(d.n) + '" title="Day ' + d.n + ': ' + esc(d.title) + (p ? ' (planned ' + p + ')' : '') + ' — ' + st + '" aria-label="Day ' + d.n + ', ' + st + '">' + d.n + '</a>';
        });
        html += '</div>';
      });
      html += '</div>';
    }
    $('#heatmap').innerHTML = html;
  }

  function day(n) {
    S.last = n; save();
    var r = C.dayRec(S, n);
    var p = C.plannedDate(S, n);
    $('#planned').textContent = p ? 'planned ' + fmtDate(p) + ' (Asia/Kolkata)' : 'set a start date in Settings for planned dates';
    function refresh() {
      var st = C.status(S, n), nb = r.blocks.filter(Boolean).length;
      $('#day-status').textContent = (st === 'done' ? '✓ Day complete' + (r.completedAt ? ' on ' + r.completedAt : '') : nb + ' of 5 blocks done') + (st === 'behind' ? ' — behind schedule' : '');
      $('#day-status').className = st;
    }
    $$('.blk').forEach(function (cb) {
      cb.checked = !!r.blocks[+cb.dataset.i];
      cb.addEventListener('change', function () { r.blocks[+cb.dataset.i] = cb.checked; C.touchDay(S, n); save(); refresh(); });
    });
    var man = $('#manual'); man.checked = !!r.manual;
    man.addEventListener('change', function () { r.manual = man.checked; C.touchDay(S, n); save(); refresh(); });
    var h = $('#hours'); h.value = r.hours || '';
    h.addEventListener('change', function () { var v = parseFloat(h.value); r.hours = isFinite(v) && v >= 0 ? Math.min(24, v) : 0; save(); });
    var nt = $('#notes'); nt.value = r.notes || ''; var tmr;
    nt.addEventListener('input', function () { clearTimeout(tmr); tmr = setTimeout(function () { r.notes = nt.value; save(); }, 300); });
    nt.addEventListener('blur', function () { r.notes = nt.value; save(); });
    refresh();
  }

  function quiz(sec) {
    var id = sec.dataset.quiz, q = S.quizzes[id] = S.quizzes[id] || { answers: {}, marks: {} };
    function score() { var s = C.quizScore(q), tot = $$('.q', sec).length; $('.qscore', sec).textContent = 'Score: ' + s.right + ' right, ' + s.wrong + ' wrong, ' + (tot - s.right - s.wrong) + ' unmarked (of ' + tot + ')'; }
    $$('.q', sec).forEach(function (el) {
      var k = el.dataset.q, ta = $('.qa', el), ans = $('.ans', el), rv = $('.reveal', el);
      ta.value = q.answers[k] || '';
      ta.addEventListener('input', function () { q.answers[k] = ta.value; save(); });
      if (rv) rv.addEventListener('click', function () { ans.hidden = !ans.hidden; rv.setAttribute('aria-expanded', String(!ans.hidden)); rv.textContent = ans.hidden ? 'Reveal answer' : 'Hide answer'; });
      function paint() { $$('.mark', el).forEach(function (b) { b.setAttribute('aria-pressed', String(q.marks[k] === b.dataset.v)); }); el.dataset.mark = q.marks[k] || ''; }
      $$('.mark', el).forEach(function (b) { b.addEventListener('click', function () { q.marks[k] = q.marks[k] === b.dataset.v ? '' : b.dataset.v; if (!q.marks[k]) delete q.marks[k]; save(); paint(); score(); }); });
      paint();
    });
    $('.qreset', sec).addEventListener('click', function () { if (confirm('Clear your answers and marks for this quiz?')) { S.quizzes[id] = { answers: {}, marks: {} }; save(); location.reload(); } });
    score();
  }

  function syllabus() {
    var fm = $('#fm'); for (var m = 1; m <= 6; m++) fm.insertAdjacentHTML('beforeend', '<option>' + m + '</option>');
    var idx = {}; (window.SEARCH || []).forEach(function (s) { idx[s.n] = (s.t + ' ' + s.x).toLowerCase(); });
    var params = new URLSearchParams(location.search); if (params.get('q')) $('#q').value = params.get('q');
    function draw() {
      var q = $('#q').value.trim().toLowerCase(), terms = q ? q.split(/\s+/) : [], mo = fm.value, sf = $('#fs').value, rows = [];
      D.days.forEach(function (d) {
        if (mo && d.month !== +mo) return;
        var st = C.status(S, d.n); if (sf && st !== sf) return;
        var hay = idx[d.n] || d.title.toLowerCase();
        if (terms.some(function (t) { return hay.indexOf(t) < 0; })) return;
        var snip = '';
        if (terms.length) { var i = hay.indexOf(terms[0]); snip = '<div class="snip">…' + esc(hay.slice(Math.max(0, i - 60), i + 90)) + '…</div>'; }
        rows.push('<tr class="' + st + '"><td>' + d.n + '</td><td><a href="weeks/week-' + String(d.week).padStart(2, '0') + '.html">' + d.week + '</a></td><td>' + d.month + '</td><td><a href="' + dayHref(d.n) + '">' + esc(d.title) + '</a>' + snip + '</td><td>' + (C.plannedDate(S, d.n) || '—') + '</td><td>' + (badge(st) || 'not started') + '</td></tr>');
      });
      $('#rows').innerHTML = rows.join(''); $('#count').textContent = rows.length + ' of 180 days shown';
    }
    ['input', 'change'].forEach(function (ev) { $('#q').addEventListener(ev, draw); fm.addEventListener(ev, draw); $('#fs').addEventListener(ev, draw); });
    draw();
  }

  function review() {
    var t = C.todayISO(), due = C.reviewsDue(S, t), nd = C.nextDay(S), html = '';
    html += '<h3>From your completions (+1/+3/+7/+21 days)</h3>';
    html += due.length ? '<ul>' + due.map(function (x) { return '<li><a href="' + dayHref(x.day) + '">Day ' + x.day + ': ' + esc(dayMeta[x.day].title) + '</a> (+' + x.gap + ')</li>'; }).join('') + '</ul>' : '<p>Nothing due from your completion dates today.</p>';
    if (nd && D.sched[nd]) {
      var sc = D.sched[nd];
      html += '<h3>Syllabus reviews for your next day (Day ' + nd + ', from review-schedule.csv)</h3>';
      html += sc.review.length ? '<ul>' + sc.review.map(function (d) { return '<li><a href="' + dayHref(d) + '">Day ' + d + ': ' + esc(dayMeta[d] ? dayMeta[d].title : '') + '</a> ' + badge(C.status(S, d)) + '</li>'; }).join('') + '</ul>' : '<p>None listed.</p>';
      if (sc.tags) html += '<p>Flashcard tags due: <code>' + esc(sc.tags) + '</code></p>';
    }
    $('#review-today').innerHTML = html;
    // flashcards
    var deck = $('#deck');
    for (var m = 1; m <= 6; m++) deck.insertAdjacentHTML('beforeend', '<option value="m' + m + '">Month ' + m + ' (due)</option>');
    var doneSet = {}; C.completedDays(S).forEach(function (d) { doneSet[d] = 1; });
    var queue = [], cur = null, flipped = false;
    function build() {
      var v = deck.value;
      queue = D.cards.filter(function (c) {
        if (!C.cardDue(S.cards[c.id], t)) return false;
        if (v === 'due') return c.day && doneSet[c.day];
        if (v === 'all-due') return true;
        return 'm' + c.month === v;
      });
      next();
    }
    function next() {
      cur = queue.shift() || null; flipped = false; show();
    }
    function show() {
      var fc = $('#fc');
      $('#deck-info').textContent = ' ' + (cur ? queue.length + 1 : 0) + ' card(s) due in this deck';
      $$('.grade').forEach(function (b) { b.disabled = !cur || !flipped; });
      $('#flip').disabled = !cur;
      if (!cur) { fc.innerHTML = '<p>No cards due here. ' + (deck.value === 'due' ? 'Complete days (or pick another deck) to unlock their cards.' : 'Come back tomorrow.') + '</p>'; return; }
      fc.innerHTML = '<div class="side">' + (flipped ? 'Answer' : 'Question') + ' · <small>' + esc(cur.tags) + '</small></div><div class="face">' + esc(flipped ? cur.b : cur.f) + '</div>' + (flipped ? '<div class="faint">Q: ' + esc(cur.f) + '</div>' : '');
    }
    function grade(g) {
      if (!cur || !flipped) return;
      S.cards[cur.id] = C.sm2(S.cards[cur.id], g, t); save();
      if (g < 3) queue.push(cur); // Again: see it later this session
      next();
    }
    $('#flip').addEventListener('click', function () { if (cur) { flipped = !flipped; show(); } });
    $$('.grade').forEach(function (b) { b.addEventListener('click', function () { grade(+b.dataset.g); }); });
    document.addEventListener('keydown', function (e) {
      var tg = e.target.tagName; if (tg === 'INPUT' || tg === 'TEXTAREA' || tg === 'SELECT') return;
      if (e.key === ' ') { e.preventDefault(); $('#flip').click(); }
      var map = { '1': 0, '2': 3, '3': 4, '4': 5 }; if (e.key in map) grade(map[e.key]);
    });
    deck.addEventListener('change', build);
    build();
  }

  function settings() {
    var st = $('#start'); st.value = S.startDate || '';
    st.addEventListener('change', function () { S.startDate = /^\d{4}-\d{2}-\d{2}$/.test(st.value) ? st.value : ''; save(); msg('Start date saved.'); });
    function msg(t) { $('#msg').textContent = t; }
    $('#export').addEventListener('click', function () {
      var blob = new Blob([JSON.stringify(S, null, 2)], { type: 'application/json' });
      var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'aiml-progress-' + C.todayISO() + '.json';
      document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500); msg('Exported.');
    });
    $('#import').addEventListener('change', function (e) {
      var f = e.target.files[0]; if (!f) return;
      var rd = new FileReader();
      rd.onload = function () {
        try { var obj = C.validateImport(JSON.parse(rd.result)); if (!confirm('Replace your current progress with the imported file?')) return; S = obj; save(); msg('Imported ' + C.completedDays(S).length + ' completed days.'); st.value = S.startDate || ''; }
        catch (err) { msg('Import failed: ' + err.message); }
      };
      rd.readAsText(f);
    });
    $('#reset').addEventListener('click', function () {
      if (confirm('Delete ALL progress, notes, quiz marks and flashcard history in this browser?') && prompt('Type RESET to confirm') === 'RESET') { var th = S.theme; S = C.emptyState(); S.theme = th; save(); st.value = ''; msg('Progress reset.'); }
    });
  }
})();
