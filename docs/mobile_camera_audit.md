# Mobile camera — audit and changes

Audit of the existing code before any change (files inspected: `frontend/src/pages/PhoneCamera.jsx`,
`LiveCamera.jsx`, `hooks/useEvents.js`, `backend/api/ws.py`, `routes_camera.py`, `routes_data.py`,
`main.py`, `services/camera_worker.py`, `scripts/run_demo.ps1`).

## Already present (kept, not rewritten)
- `/camera` page using `getUserMedia`, `/ws/stream` JPEG round-trip, `/ws/events`, `/api/stream/annotated`,
  laptop-webcam and IP-camera sources, QR helper, Cloudflare/self-signed launcher.

## Gaps found
| # | Gap | Severity |
|---|---|---|
| 1 | CORS `allow_origins=["*"]` with credentials | medium |
| 2 | SPA fallback served `frontend/dist/<path>` without confining to `dist/` (path-traversal risk) | high when tunneled |
| 3 | Through a tunnel **every** route was public: `/api/*` (history, snapshots, settings), `/ws/events`, `/docs` | high |
| 4 | `/ws/stream` had no authentication, no frame-size or rate limit | high when tunneled |
| 5 | No stall watchdog, no single-session rule, no "clean Stop" vs "lost connection" distinction | medium |
| 6 | Phone page: no error mapping, no reconnect, no camera switch, no wake lock, `facingMode` string not `ideal` | functional |
| 7 | Dashboard had no phone connection status | functional |

## Changes
- `backend/services/security.py` (new): per-run token, `RemoteAccessGuard` ASGI middleware. A request is
  *remote* if the TCP peer is not loopback **or** it carries proxy headers (`Cf-Connecting-Ip`,
  `X-Forwarded-For`, ...). cloudflared connects from 127.0.0.1, so the header check is what identifies tunnel
  traffic. Remote visitors may only load `/camera`, `/assets/*`, `/api/phone/config` (token) and `/ws/stream`
  (token). Everything else returns 403 (HTTP) or close 4403 (WebSocket).
- `ws.py`: first-message `{"type":"auth","token"}` for remote clients (close 4401 otherwise), token never in the
  URL or logs, 1 MB frame cap, decoded-pixel cap, ~10 FPS server pacing, newest phone replaces older (4409),
  watchdog: one `camera_disconnected` alert after 5 s of silence, idle close at 30 s, clean Stop (1000/1001)
  raises no alert, per-frame errors answered with `{"type":"error"}` and never crash the loop.
- `main.py`: CORS limited to `settings.CORS_ORIGINS`; SPA fallback resolves paths and serves only files inside
  `frontend/dist`; unknown `api/`, `ws/`, `results/` paths are real 404s.
- `routes_camera.py`: token-protected `GET /api/phone/config` (send_width, jpeg_quality, fps_cap, clamped).
- `routes_data.py`: phone URL carries the token in the **URL fragment** (`#t=...`), which browsers never send to
  servers, so it stays out of tunnel and uvicorn logs. The phone page removes it from the address bar.
- Frontend: `PhoneCamera.jsx` rewritten around `utils/phoneCamera.js` (pure, unit-tested helpers); dashboard
  `LiveCamera.jsx` shows Waiting / Streaming / Disconnected from `/ws/events` camera events.
- `scripts/run_demo.ps1`: generates the token, passes it as `FG_PHONE_TOKEN`, prints the phone URL and the
  fallbacks (hotspot, `-SelfSigned`, IP Webcam).

## Residual risks (be aware)
- Anyone holding the full phone URL (with token) can stream frames into your pipeline until you stop the script.
  The token changes every run. Do not share or screenshot the QR code.
- The guard trusts that tunnel traffic carries proxy headers (true for cloudflared). Do not expose port 8000 by
  any other reverse proxy that strips them.
- In `-SelfSigned` mode the laptop dashboard is `https://localhost:8000` (the phone link is `https://LAN-IP:8000`).
