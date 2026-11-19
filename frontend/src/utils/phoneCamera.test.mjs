import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mapCameraError,
  backoffDelayMs,
  parseTokenFromHash,
  streamUrlFor,
  cameraApiAvailable,
  describeClose,
} from './phoneCamera.js';

test('maps permission denied to an actionable message', () => {
  for (const name of ['NotAllowedError', 'PermissionDeniedError']) {
    const r = mapCameraError({ name });
    assert.equal(r.code, 'permission_denied');
    assert.match(r.message, /permission denied/i);
    assert.match(r.message, /Allow/);
  }
});

test('maps missing / busy / overconstrained / insecure / unknown errors', () => {
  assert.equal(mapCameraError({ name: 'NotFoundError' }).code, 'no_camera');
  assert.equal(mapCameraError({ name: 'DevicesNotFoundError' }).code, 'no_camera');
  assert.equal(mapCameraError({ name: 'NotReadableError' }).code, 'camera_busy');
  assert.equal(mapCameraError({ name: 'TrackStartError' }).code, 'camera_busy');
  assert.equal(mapCameraError({ name: 'OverconstrainedError' }).code, 'overconstrained');
  assert.equal(mapCameraError({ name: 'SecurityError' }).code, 'insecure');
  assert.equal(mapCameraError({ name: 'TypeError' }).code, 'insecure');
  const u = mapCameraError({ name: 'WeirdError', message: 'boom' });
  assert.equal(u.code, 'unknown');
  assert.match(u.message, /WeirdError/);
  assert.match(u.message, /boom/);
  assert.equal(mapCameraError(null).code, 'unknown');
});

test('backoff doubles from 1s and caps at 10s', () => {
  assert.deepEqual([1, 2, 3, 4, 5, 6, 20].map((n) => backoffDelayMs(n)), [1000, 2000, 4000, 8000, 10000, 10000, 10000]);
  assert.equal(backoffDelayMs(0), 1000);
});

test('token is read from the URL fragment only', () => {
  const tok = 'AbC123_-AbC123_-AbC123_-AbC123_-';
  assert.equal(parseTokenFromHash(`#t=${tok}`), tok);
  assert.equal(parseTokenFromHash(`#x=1&t=${tok}`), tok);
  assert.equal(parseTokenFromHash('#t=short'), null);
  assert.equal(parseTokenFromHash(''), null);
  assert.equal(parseTokenFromHash(undefined), null);
  assert.equal(parseTokenFromHash(`#t=${tok}%0A`), null);
});

test('stream URL uses WSS on HTTPS pages and WS otherwise, same origin', () => {
  assert.equal(streamUrlFor({ protocol: 'https:', host: 'demo.trycloudflare.com' }), 'wss://demo.trycloudflare.com/ws/stream');
  assert.equal(streamUrlFor({ protocol: 'http:', host: 'localhost:8000' }), 'ws://localhost:8000/ws/stream');
});

test('camera API availability requires a secure context and mediaDevices', () => {
  const md = { getUserMedia() {} };
  assert.equal(cameraApiAvailable({ isSecureContext: true, navigator: { mediaDevices: md } }), true);
  assert.equal(cameraApiAvailable({ isSecureContext: false, navigator: { mediaDevices: md } }), false);
  assert.equal(cameraApiAvailable({ isSecureContext: true, navigator: {} }), false);
  assert.equal(cameraApiAvailable(null), false);
});

test('close codes: auth/replace are fatal, others are retried', () => {
  assert.equal(describeClose(4401).fatal, true);
  assert.equal(describeClose(4409).fatal, true);
  assert.equal(describeClose(4403).fatal, true);
  assert.equal(describeClose(1006).fatal, false);
  assert.equal(describeClose(1001).fatal, false);
});
