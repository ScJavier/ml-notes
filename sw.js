/* Service worker: la app funciona sin conexión. Sube VERSION cuando cambie la lista SHELL. */
var VERSION = 'v2';
var SHELL = ['./', 'index.html', 'style.css', 'app.js', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'vendor/katex/katex.min.css',
  'vendor/katex/fonts/KaTeX_AMS-Regular.woff2','vendor/katex/fonts/KaTeX_Caligraphic-Bold.woff2','vendor/katex/fonts/KaTeX_Caligraphic-Regular.woff2','vendor/katex/fonts/KaTeX_Fraktur-Bold.woff2','vendor/katex/fonts/KaTeX_Fraktur-Regular.woff2','vendor/katex/fonts/KaTeX_Main-Bold.woff2','vendor/katex/fonts/KaTeX_Main-BoldItalic.woff2','vendor/katex/fonts/KaTeX_Main-Italic.woff2','vendor/katex/fonts/KaTeX_Main-Regular.woff2','vendor/katex/fonts/KaTeX_Math-BoldItalic.woff2','vendor/katex/fonts/KaTeX_Math-Italic.woff2','vendor/katex/fonts/KaTeX_SansSerif-Bold.woff2','vendor/katex/fonts/KaTeX_SansSerif-Italic.woff2','vendor/katex/fonts/KaTeX_SansSerif-Regular.woff2','vendor/katex/fonts/KaTeX_Script-Regular.woff2','vendor/katex/fonts/KaTeX_Size1-Regular.woff2','vendor/katex/fonts/KaTeX_Size2-Regular.woff2','vendor/katex/fonts/KaTeX_Size3-Regular.woff2','vendor/katex/fonts/KaTeX_Size4-Regular.woff2','vendor/katex/fonts/KaTeX_Typewriter-Regular.woff2'];
self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(VERSION).then(function (c) { return c.addAll(SHELL); }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener('activate', function (e) {
  e.waitUntil(caches.keys().then(function (ks) {
    return Promise.all(ks.filter(function (k) { return k !== VERSION; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
function networkFirst(req) {
  return new Promise(function (resolve) {
    var done = false;
    function fromCache() {
      return caches.match(req, { ignoreSearch: true }).then(function (m) {
        if (m) return m;
        if (req.mode === 'navigate') return caches.match('index.html');
        return new Response('Sin conexión', { status: 504 });
      });
    }
    var timer = setTimeout(function () {
      fromCache().then(function (m) { if (!done && m && m.status !== 504) { done = true; resolve(m); } });
    }, 4000);
    fetch(new Request(req.url, { cache: 'no-cache' })).then(function (res) {
      clearTimeout(timer);
      if (res && res.ok) { var cp = res.clone(); caches.open(VERSION).then(function (c) { c.put(req.url, cp); }); }
      if (!done) { done = true; resolve(res); }
    }).catch(function () {
      clearTimeout(timer);
      fromCache().then(function (m) { if (!done) { done = true; resolve(m); } });
    });
  });
}
self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (url.origin !== location.origin) return;
  if (url.pathname.indexOf('/vendor/katex/fonts/') !== -1) {
    e.respondWith(caches.match(req).then(function (m) {
      return m || fetch(req).then(function (res) { var cp = res.clone(); caches.open(VERSION).then(function (c) { c.put(req, cp); }); return res; });
    }));
    return;
  }
  e.respondWith(networkFirst(req));
});
