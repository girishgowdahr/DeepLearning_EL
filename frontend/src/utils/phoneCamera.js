// ForestGuard AI — pure helpers for the Android phone-camera page.
// No DOM/React imports, so they can be unit-tested with `node --test`.

/** Map a getUserMedia / camera error to a plain-language message with a fix hint. */
export function mapCameraError(err) {
  const name = (err && err.name) || 'Error';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return {
        code: 'permission_denied',
        message:
          "Camera permission denied. Tap the lock icon in Chrome's address bar → Permissions → Camera → Allow, then press Start again.",
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return { code: 'no_camera', message: 'No camera found on this device.' };
    case 'NotReadableError':
    case 'TrackStartError':
      return {
        code: 'camera_busy',
        message: 'Camera is in use by another app. Close other apps that use the camera and try again.',
      };
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return { code: 'overconstrained', message: 'This camera does not support the requested settings.' };
    case 'SecurityError':
      return {
        code: 'insecure',
        message: 'Camera needs HTTPS. Open the https:// link from the dashboard (not the http://192.168… address).',
      };
    case 'TypeError':
      return {
        code: 'insecure',
        message: 'Camera is not available on this page. Open the https:// link shown on the dashboard.',
      };
    default:
      return { code: 'unknown', message: `Camera error: ${name}${err && err.message ? ' — ' + err.message : ''}` };
  }
}

/** Exponential backoff: 1s, 2s, 4s, 8s, then capped at 10s. `attempt` starts at 1. */
export function backoffDelayMs(attempt, baseMs = 1000, maxMs = 10000) {
  const n = Math.max(1, attempt | 0);
  return Math.min(maxMs, baseMs * 2 ** (n - 1));
}

/** Read the access token from a URL fragment such as "#t=abc...". Returns null if absent. */
export function parseTokenFromHash(hash) {
  const m = /(?:^#|&)t=([A-Za-z0-9_-]{16,128})(?:&|$)/.exec(hash || '');
  return m ? m[1] : null;
}

/** ws:// or wss:// URL on the same origin as the page (WSS automatically on HTTPS). */
export function streamUrlFor(loc) {
  return `${loc.protocol === 'https:' ? 'wss:' : 'ws:'}//${loc.host}/ws/stream`;
}

/** True when the browser will allow getUserMedia (HTTPS or localhost + API present). */
export function cameraApiAvailable(win) {
  return Boolean(win && win.isSecureContext && win.navigator && win.navigator.mediaDevices && win.navigator.mediaDevices.getUserMedia);
}

/** Interpret a WebSocket close code from the server. */
export function describeClose(code) {
  switch (code) {
    case 4401:
      return { fatal: true, message: 'Access token missing or invalid. Re-scan the QR code shown on the laptop dashboard.' };
    case 4403:
      return { fatal: true, message: 'This device is not allowed to use that connection.' };
    case 4409:
      return { fatal: true, message: 'Another phone connected and took over the stream.' };
    case 1008:
      return { fatal: true, message: 'The server refused the connection.' };
    default:
      return { fatal: false, message: 'Connection lost.' };
  }
}

/** Camera connection states shown on the phone page. */
export const STATE_LABELS = {
  idle: 'Idle',
  requesting: 'Requesting permission…',
  ready: 'Camera ready',
  connecting: 'Connecting…',
  streaming: 'Streaming',
  reconnecting: 'Reconnecting',
  disconnected: 'Disconnected',
  error: 'Error',
};
