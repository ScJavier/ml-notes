/* Notas ML — lector con progreso por posición en el contenido (no por píxeles).
   Todo vive en el dispositivo: localStorage 'prog:<id>' = {k, f, p, rev, t}
   k = índice de bloque, f = fracción dentro del bloque, p = progreso lineal en [0,1]. */
(function () {
  'use strict';
  var APP_VERSION = '1.0';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function norm(t) { return t.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase(); }

  /* ---------- preferencias ---------- */
  var FF = { g: "Georgia,'Iowan Old Style','Palatino Linotype',serif", s: "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif", m: "ui-monospace,Menlo,Consolas,monospace" };
  var TC = { sepia: '#efe4c8', light: '#f4f4f4', dark: '#1e2128' };
  var prefs = { fs: 19, ff: 'g', th: 'sepia' };
  try { var sp = JSON.parse(lsGet('prefs') || '{}'); for (var k0 in sp) prefs[k0] = sp[k0]; } catch (e) {}

  var S = { id: null, rev: '', notes: [], blocks: [], w: [], cum: [], total: 1, ready: false, restoring: false, dragging: false, sidx: null, timer: 0 };

  function applyPrefs() {
    var pos = (S.id !== null && S.ready) ? locate() : null;
    var root = document.documentElement;
    root.style.setProperty('--fs', prefs.fs + 'px');
    root.style.setProperty('--ff', FF[prefs.ff]);
    root.setAttribute('data-theme', prefs.th);
    $('#ff').value = prefs.ff; $('#th').value = prefs.th;
    var m = $('meta[name=theme-color]'); if (m) m.setAttribute('content', TC[prefs.th]);
    lsSet('prefs', JSON.stringify(prefs));
    if (pos) { // reanclar: el cambio de letra mueve los píxeles, no el contenido
      S.restoring = true;
      requestAnimationFrame(function () {
        if (pos.p >= 1) toBottom(); else goTo(pos.k, pos.f);
        setTimeout(function () { S.restoring = false; updateUI(); }, 150);
      });
    }
  }
  $('#fm').onclick = function () { prefs.fs = Math.max(13, prefs.fs - 1); applyPrefs(); };
  $('#fp').onclick = function () { prefs.fs = Math.min(34, prefs.fs + 1); applyPrefs(); };
  $('#ff').onchange = function () { prefs.ff = this.value; applyPrefs(); };
  $('#th').onchange = function () { prefs.th = this.value; applyPrefs(); };
  $('#menu').onclick = function () { document.body.classList.toggle('open'); };

  /* ---------- progreso ---------- */
  function getProg(id) { try { return JSON.parse(lsGet('prog:' + id)); } catch (e) { return null; } }
  function topLine() { return $('#bar').offsetHeight + 6; }

  function locate() {
    var B = S.blocks; if (!B.length) return { k: 0, f: 0, p: 0 };
    if (window.scrollY <= 1) return { k: 0, f: 0, p: 0 };
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) return { k: B.length - 1, f: 1, p: 1 };
    var t = topLine(), lo = 0, hi = B.length - 1;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (B[mid].getBoundingClientRect().bottom > t) hi = mid; else lo = mid + 1; }
    var r = B[lo].getBoundingClientRect();
    var f = r.height > 0 ? Math.min(1, Math.max(0, (t - r.top) / r.height)) : 0;
    return { k: lo, f: f, p: (S.cum[lo] + f * S.w[lo]) / S.total };
  }
  function goTo(k, f) {
    var b = S.blocks[k]; if (!b) return;
    var r = b.getBoundingClientRect();
    window.scrollTo({ top: Math.max(0, window.scrollY + r.top + f * r.height - topLine()), behavior: 'instant' });
  }
  function toBottom() { window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'instant' }); }
  function goToP(p) {
    if (p >= 1) return toBottom();
    var target = p * S.total, lo = 0, hi = S.blocks.length - 1;
    while (lo < hi) { var mid = (lo + hi + 1) >> 1; if (S.cum[mid] <= target) lo = mid; else hi = mid - 1; }
    goTo(lo, S.w[lo] > 0 ? (target - S.cum[lo]) / S.w[lo] : 0);
  }
  function setSlider(p) { $('#sl').value = Math.round(p * 1000); $('#pct').textContent = Math.round(p * 100) + ' %'; }
  function updateUI() { if (S.id === null || S.restoring || S.dragging) return; setSlider(locate().p); }
  function save() {
    if (!S.ready || S.id === null || S.restoring) return;
    var pos = locate();
    lsSet('prog:' + S.id, JSON.stringify({ k: pos.k, f: pos.f, p: pos.p, rev: S.rev, t: Date.now() }));
  }
  var raf = 0;
  window.addEventListener('scroll', function () {
    if (S.id === null) return;
    if (!raf) raf = requestAnimationFrame(function () { raf = 0; updateUI(); });
    clearTimeout(S.timer); S.timer = setTimeout(save, 500);
  }, { passive: true });
  document.addEventListener('visibilitychange', function () { if (document.visibilityState === 'hidden') save(); });
  window.addEventListener('pagehide', save);

  var sl = $('#sl');
  sl.addEventListener('input', function () { S.dragging = true; var p = sl.value / 1000; goToP(p); $('#pct').textContent = Math.round(p * 100) + ' %'; });
  sl.addEventListener('change', function () { S.dragging = false; save(); });

  /* ---------- biblioteca ---------- */
  function loadIndex() {
    return fetch('notes/index.json', { cache: 'no-cache' }).then(function (r) { if (!r.ok) throw 0; return r.json(); })
      .then(function (j) { S.notes = j; lsSet('index-cache', JSON.stringify(j)); return j; })
      .catch(function () { var c = lsGet('index-cache'); if (c) { try { S.notes = JSON.parse(c); } catch (e) {} } return S.notes; });
  }
  function prefetchNotes() { S.notes.forEach(function (n) { fetch(n.file, { cache: 'no-cache' }).catch(function () {}); }); }

  function showView(v) { document.body.setAttribute('data-view', v); document.body.classList.remove('open'); }
  function renderLib() {
    showView('lib'); S.id = null; S.ready = false;
    document.title = 'Notas ML'; $('#ttl').textContent = 'Notas ML';
    var h = '';
    S.notes.forEach(function (n) {
      var pr = getProg(n.id), p = pr ? pr.p : 0, pct = Math.round(p * 100);
      var lbl = p >= 0.98 ? 'Leída ✓' : (p > 0 ? 'Continuar · ' + pct + ' %' : 'Empezar');
      h += '<a class="card" href="#/n/' + esc(n.id) + '"><b>' + esc(n.title) + '</b><div class="bar"><i style="width:' + pct + '%"></i></div>' +
        '<div class="meta"><span>' + esc(n.author || '') + '</span><span>' + lbl + '</span></div></a>';
    });
    $('#list').innerHTML = h || '<p class="muted">No hay notas disponibles todavía.</p>';
    $('#status').textContent = navigator.onLine ? '' : 'Sin conexión: se muestran las notas guardadas en este dispositivo.';
    $('#ver').textContent = 'Versión ' + APP_VERSION + ' · el progreso se guarda solo en este dispositivo.';
    window.scrollTo({ top: 0, behavior: 'instant' });
  }

  /* ---------- lector ---------- */
  var SEL = '#content > h1, #content > h2, #content > h3, #content > p, #content li, #content > table, #content > blockquote > p';
  function openNote(n) {
    return fetch(n.file, { cache: 'no-cache' }).then(function (r) { if (!r.ok) throw 0; return r.text(); }).then(function (html) {
      showView('reader');
      S.id = n.id; S.rev = n.rev || ''; S.ready = false; S.restoring = true; S.sidx = null;
      document.title = n.title; $('#ttl').textContent = n.title;
      $('#content').innerHTML = '<h1 class="title">' + esc(n.title) + '</h1><p class="author">' + esc(n.author || '') + '</p>' + html;
      S.blocks = [].slice.call(document.querySelectorAll(SEL));
      S.w = S.blocks.map(function (b) { return Math.max(1, b.textContent.length); });
      S.cum = []; var acc = 0; S.w.forEach(function (w, i) { S.cum[i] = acc; acc += w; }); S.total = acc || 1;
      buildToc(); $('#q').value = ''; $('#res').hidden = true; $('#toc').hidden = false;
      window.scrollTo({ top: 0, behavior: 'instant' });
      var pr = getProg(n.id);
      function restore() {
        if (!pr || !(pr.p > 0)) return;
        if (pr.p >= 1) toBottom();
        else if (pr.rev === S.rev && S.blocks[pr.k]) goTo(pr.k, pr.f);
        else goToP(pr.p);
      }
      setSlider(pr && pr.p > 0 ? pr.p : 0);
      restore();
      var fontsReady = (document.fonts && document.fonts.ready) || Promise.resolve();
      fontsReady.then(function () {
        setTimeout(function () { restore(); setTimeout(function () { S.restoring = false; S.ready = true; updateUI(); }, 300); }, 150);
      });
    });
  }

  var io = null;
  function buildToc() {
    var toc = $('#toc'); toc.innerHTML = '';
    var hs = [].slice.call(document.querySelectorAll('#content h1:not(.title), #content h2'));
    if (io) io.disconnect();
    io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { hs.forEach(function (h) { h._a.classList.remove('on'); }); e.target._a.classList.add('on'); } });
    }, { rootMargin: '-10% 0px -80% 0px' });
    hs.forEach(function (h) {
      var a = document.createElement('a'); a.href = '#'; a.textContent = h.textContent; a.className = h.tagName === 'H2' ? 'h2' : '';
      a.onclick = function (ev) { ev.preventDefault(); document.body.classList.remove('open'); window.scrollTo({ top: Math.max(0, window.scrollY + h.getBoundingClientRect().top - topLine()), behavior: 'smooth' }); };
      toc.appendChild(a); h._a = a; io.observe(h);
    });
  }

  function buildSearch() {
    var sec = '';
    S.sidx = S.blocks.map(function (b) {
      if (b.tagName === 'H1' || b.tagName === 'H2') sec = b.textContent;
      var c = b.cloneNode(true);
      [].forEach.call(c.querySelectorAll('.katex'), function (kx) {
        var an = kx.querySelector('annotation'); kx.replaceWith(document.createTextNode(an ? '$' + an.textContent.trim() + '$' : ''));
      });
      var raw = c.textContent.replace(/\s+/g, ' ').trim();
      return { raw: raw, n: norm(raw), sec: sec };
    });
  }
  $('#q').addEventListener('input', function () {
    var q = norm(this.value.trim()), res = $('#res'), toc = $('#toc');
    if (q.length < 2) { res.hidden = true; toc.hidden = false; return; }
    if (!S.sidx) buildSearch();
    toc.hidden = true; res.hidden = false; res.innerHTML = ''; var n = 0;
    S.sidx.forEach(function (s, i) {
      var j = s.n.indexOf(q); if (j < 0 || n >= 40) return; n++;
      var a = document.createElement('a'); a.href = '#';
      var small = document.createElement('small'); small.textContent = s.sec; a.appendChild(small);
      var st = Math.max(0, j - 30), txt = s.raw.slice(st, j + q.length + 60), sp = document.createElement('span'), jj = norm(txt).indexOf(q);
      if (jj >= 0) {
        sp.appendChild(document.createTextNode((st > 0 ? '…' : '') + txt.slice(0, jj)));
        var m = document.createElement('mark'); m.textContent = txt.slice(jj, jj + q.length); sp.appendChild(m);
        sp.appendChild(document.createTextNode(txt.slice(jj + q.length)));
      } else sp.textContent = txt;
      a.appendChild(sp);
      a.onclick = function (ev) {
        ev.preventDefault(); document.body.classList.remove('open');
        var b = S.blocks[i]; window.scrollTo({ top: Math.max(0, window.scrollY + b.getBoundingClientRect().top - topLine() - 20), behavior: 'instant' });
        b.classList.remove('flash'); void b.offsetWidth; b.classList.add('flash');
      };
      res.appendChild(a);
    });
    if (!n) { var d = document.createElement('div'); d.className = 'none'; d.textContent = 'Sin resultados'; res.appendChild(d); }
  });

  /* ---------- exportar / importar progreso ---------- */
  function toast(msg) { var t = $('#toast'); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(function () { t.hidden = true; }, 3000); }
  $('#exp').onclick = function () {
    var out = { app: 'notas-ml', exported: new Date().toISOString(), prog: {} };
    try { for (var i = 0; i < localStorage.length; i++) { var k = localStorage.key(i); if (k.indexOf('prog:') === 0) out.prog[k.slice(5)] = JSON.parse(localStorage.getItem(k)); } } catch (e) {}
    var a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: 'application/json' }));
    a.download = 'progreso-notas-ml.json'; document.body.appendChild(a); a.click(); a.remove();
    toast('Progreso exportado');
  };
  $('#imp').onclick = function () { $('#impf').click(); };
  $('#impf').onchange = function () {
    var f = this.files[0]; if (!f) return; var self = this;
    f.text().then(function (txt) {
      var j = JSON.parse(txt), n = 0;
      Object.keys(j.prog || {}).forEach(function (id) {
        var cur = getProg(id), inc = j.prog[id];
        if (!cur || (inc.t || 0) > (cur.t || 0)) { lsSet('prog:' + id, JSON.stringify(inc)); n++; }
      });
      toast('Progreso importado (' + n + ')'); renderLib();
    }).catch(function () { toast('Archivo no válido'); }).then(function () { self.value = ''; });
  };

  /* ---------- rutas ---------- */
  var indexP = loadIndex();
  function route() {
    var m = /^#\/n\/(.+)$/.exec(location.hash);
    indexP.then(function () {
      if (m) {
        var id = decodeURIComponent(m[1]), n = S.notes.filter(function (x) { return x.id === id; })[0];
        if (n) { if (S.id !== null && S.id !== id) save(); return openNote(n).catch(function () { toast('No se pudo cargar la nota (¿sin conexión?)'); location.hash = '#/'; }); }
      }
      if (S.id !== null) save();
      renderLib();
    });
  }
  window.addEventListener('hashchange', route);
  window.addEventListener('online', function () { if (S.id === null) { loadIndex().then(function () { renderLib(); prefetchNotes(); }); } });
  window.addEventListener('offline', function () { if (S.id === null) renderLib(); });

  applyPrefs();
  route();
  indexP.then(prefetchNotes);
  if (navigator.storage && navigator.storage.persist) { try { navigator.storage.persist(); } catch (e) {} }
  if ('serviceWorker' in navigator) { window.addEventListener('load', function () { navigator.serviceWorker.register('sw.js').catch(function () {}); }); }
})();
