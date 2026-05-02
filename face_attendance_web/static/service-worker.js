const CACHE_NAME = 'face-attendance-v1';

self.addEventListener('install', function(event) {
  console.log('🚀 Service Worker: Installed');
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    fetch(event.request)
  );
});