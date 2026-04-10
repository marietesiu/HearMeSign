// sw.js — SignFuture service worker
// Minimal: makes the app installable as a PWA on iOS/Android.
// Does NOT cache API responses (video clips, model inference) — those always go to network.

const CACHE   = 'signfuture-v1';
const SHELL   = ['/', '/manifest.json', '/icon.png'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  // Never intercept API calls, clips, or non-GET requests
  const url = e.request.url;
  if (e.request.method !== 'GET') return;
  if (url.includes('/clips/') || url.includes('/health') ||
      url.includes('/sign-to') || url.includes('/audio-to') ||
      url.includes('/text-to') || url.includes('/learn-check') ||
      url.includes('/train')   || url.includes('/samples') ||
      url.includes('/feedback')|| url.includes('/prepare')) return;

  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
