import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, CameraOff, AlertTriangle, RefreshCw, SwitchCamera, ShieldAlert } from 'lucide-react';
import {
  mapCameraError,
  backoffDelayMs,
  parseTokenFromHash,
  streamUrlFor,
  cameraApiAvailable,
  describeClose,
  STATE_LABELS,
} from '../utils/phoneCamera';

const TOKEN_KEY = 'fg_phone_token';
const DEFAULT_CONFIG = { send_width: 640, jpeg_quality: 0.6, fps_cap: 5 };

// Read the access token once from the URL fragment (#t=...), keep it for this tab only
// and remove it from the address bar. Fragments are never sent to the server or logged.
function loadToken() {
  let token = null;
  try {
    token = parseTokenFromHash(window.location.hash);
    if (token) {
      sessionStorage.setItem(TOKEN_KEY, token);
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    } else {
      token = sessionStorage.getItem(TOKEN_KEY);
    }
  } catch (_) {
    /* storage may be unavailable (private mode) — token then only lives for this load */
  }
  return token;
}

export function PhoneCamera() {
  const [phase, setPhase] = useState('idle'); // see STATE_LABELS
  const [attempt, setAttempt] = useState(0);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(null);
  const [serverState, setServerState] = useState('NO_FIRE');
  const [reason, setReason] = useState('Awaiting stream');
  const [error, setError] = useState(null);
  const [facing, setFacing] = useState('environment');
  const [hidden, setHidden] = useState(false);
  const [canSwitch, setCanSwitch] = useState(false);

  const apiOk = typeof window !== 'undefined' && cameraApiAvailable(window);
  const isLocalHost = ['localhost', '127.0.0.1'].includes(window.location.hostname);
  const tokenRef = useRef(null);
  const [hasToken, setHasToken] = useState(true);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const wakeLockRef = useRef(null);
  const configRef = useRef(DEFAULT_CONFIG);
  const wantStreamRef = useRef(false); // true between Start and Stop
  const attemptRef = useRef(0);
  const inFlightRef = useRef(false);
  const sentAtRef = useRef(0);
  const ackTimerRef = useRef(null);
  const nextTimerRef = useRef(null);
  const retryTimerRef = useRef(null);
  const framesRef = useRef(0);
  const fpsWindowRef = useRef(performance.now());
  const facingRef = useRef('environment');

  const clearTimers = () => {
    [ackTimerRef, nextTimerRef, retryTimerRef].forEach((r) => {
      if (r.current) clearTimeout(r.current);
      r.current = null;
    });
  };

  const requestWakeLock = useCallback(async () => {
    try {
      if ('wakeLock' in navigator && !wakeLockRef.current) {
        wakeLockRef.current = await navigator.wakeLock.request('screen');
        wakeLockRef.current.addEventListener?.('release', () => {
          wakeLockRef.current = null;
        });
      }
    } catch (e) {
      console.warn('Wake Lock unavailable:', e);
    }
  }, []);

  const stopTracks = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => {
        t.onended = null;
        t.stop();
      });
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  const closeSocket = () => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
      try {
        ws.close(1000);
      } catch (_) {}
    }
  };

  const releaseWakeLock = () => {
    if (wakeLockRef.current) {
      wakeLockRef.current.release?.().catch(() => {});
      wakeLockRef.current = null;
    }
  };

  // ---------------------------------------------------------------- camera
  const openCamera = useCallback(async (facingMode) => {
    const rear = { video: { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 30 } }, audio: false };
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(rear);
    } catch (err) {
      if (err && err.name === 'OverconstrainedError') {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); // relaxed retry
      } else {
        throw err;
      }
    }
    stopTracks();
    streamRef.current = stream;
    stream.getVideoTracks().forEach((track) => {
      track.onended = () => {
        if (!wantStreamRef.current) return;
        wantStreamRef.current = false;
        clearTimers();
        closeSocket();
        releaseWakeLock();
        setPhase('error');
        setError('Camera stopped (another app took it or it was disconnected). Press Start to try again.');
      };
    });
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      setCanSwitch(devices.filter((d) => d.kind === 'videoinput').length > 1);
    } catch (_) {}
  }, []);

  // ---------------------------------------------------------------- frame loop (ack-based backpressure)
  const sendNextFrame = useCallback(() => {
    nextTimerRef.current = null;
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !video || !canvas || !wantStreamRef.current) return;
    if (inFlightRef.current) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      nextTimerRef.current = setTimeout(sendNextFrame, 100);
      return;
    }

    const cfg = configRef.current;
    canvas.width = cfg.send_width;
    canvas.height = Math.round(cfg.send_width * (video.videoHeight / video.videoWidth));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    inFlightRef.current = true;
    canvas.toBlob(
      (blob) => {
        const live = wsRef.current;
        if (!blob || !live || live.readyState !== WebSocket.OPEN) {
          inFlightRef.current = false;
          return;
        }
        sentAtRef.current = performance.now();
        live.send(blob);
        // If the reply is lost, free the slot after 1.5 s instead of stalling forever
        ackTimerRef.current = setTimeout(() => {
          inFlightRef.current = false;
          sendNextFrame();
        }, 1500);
      },
      'image/jpeg',
      cfg.jpeg_quality
    );
  }, []);

  const onAck = useCallback(() => {
    inFlightRef.current = false;
    if (ackTimerRef.current) clearTimeout(ackTimerRef.current);
    ackTimerRef.current = null;
    framesRef.current += 1;
    const now = performance.now();
    setLatency(Math.round(now - sentAtRef.current));
    if (now - fpsWindowRef.current >= 1000) {
      setFps(framesRef.current);
      framesRef.current = 0;
      fpsWindowRef.current = now;
    }
    const interval = 1000 / Math.max(1, configRef.current.fps_cap);
    const wait = Math.max(0, interval - (now - sentAtRef.current));
    nextTimerRef.current = setTimeout(sendNextFrame, wait);
  }, [sendNextFrame]);

  // ---------------------------------------------------------------- websocket with reconnect
  const connectSocket = useCallback(() => {
    if (!wantStreamRef.current) return;
    closeSocket();
    setPhase(attemptRef.current > 0 ? 'reconnecting' : 'connecting');

    const ws = new WebSocket(streamUrlFor(window.location)); // wss:// automatically on https pages
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    let authed = false;

    const startStreaming = () => {
      authed = true;
      attemptRef.current = 0;
      setAttempt(0);
      setError(null);
      setPhase('streaming');
      inFlightRef.current = false;
      sendNextFrame();
    };

    ws.onopen = () => {
      if (tokenRef.current) {
        // Token travels in the first message, never in the URL
        ws.send(JSON.stringify({ type: 'auth', token: tokenRef.current }));
      } else if (isLocalHost) {
        startStreaming(); // local development: no token needed
      } else {
        wantStreamRef.current = false;
        setPhase('error');
        setError('Access token missing. Scan the QR code shown on the laptop dashboard (Live Camera page).');
        closeSocket();
      }
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'auth_ok') {
            if (msg.config) configRef.current = { ...DEFAULT_CONFIG, ...msg.config };
            startStreaming();
          } else if (msg.type === 'error') {
            onAck(); // an error reply still frees the slot for the next frame
          } else if (msg.decision) {
            setServerState(msg.decision.status);
            setReason(msg.decision.reason);
          }
        } catch (_) {}
      } else {
        onAck(); // binary annotated frame = ack
      }
    };

    ws.onerror = () => {
      /* onclose follows; handled there */
    };

    ws.onclose = (ev) => {
      if (wsRef.current !== ws) return; // stale socket
      wsRef.current = null;
      clearTimers();
      inFlightRef.current = false;
      if (!wantStreamRef.current) return;
      const info = describeClose(ev.code);
      if (info.fatal) {
        wantStreamRef.current = false;
        setPhase('error');
        setError(info.message);
        return;
      }
      attemptRef.current += 1;
      setAttempt(attemptRef.current);
      setPhase('reconnecting');
      retryTimerRef.current = setTimeout(connectSocket, backoffDelayMs(attemptRef.current));
    };
  }, [onAck, sendNextFrame, isLocalHost]);

  // ---------------------------------------------------------------- controls
  const startCamera = async () => {
    setError(null);
    if (!apiOk) {
      setPhase('error');
      setError('Camera needs HTTPS. Open the https:// link shown on the laptop dashboard (not the http://192.168… address).');
      return;
    }
    setPhase('requesting');
    try {
      await openCamera(facingRef.current);
    } catch (err) {
      const mapped = mapCameraError(err);
      setPhase('error');
      setError(mapped.message);
      stopTracks();
      return;
    }
    setPhase('ready');
    wantStreamRef.current = true;
    attemptRef.current = 0;
    setAttempt(0);
    await requestWakeLock();
    connectSocket();
  };

  const stopCamera = () => {
    wantStreamRef.current = false;
    clearTimers();
    closeSocket();
    stopTracks();
    releaseWakeLock();
    inFlightRef.current = false;
    setFps(0);
    setLatency(null);
    setServerState('NO_FIRE');
    setReason('Awaiting stream');
    setAttempt(0);
    setPhase('idle');
  };

  const switchCamera = async () => {
    const next = facingRef.current === 'environment' ? 'user' : 'environment';
    try {
      await openCamera(next);
      facingRef.current = next;
      setFacing(next);
    } catch (err) {
      setError(mapCameraError(err).message);
    }
  };

  // ---------------------------------------------------------------- lifecycle
  useEffect(() => {
    tokenRef.current = loadToken();
    setHasToken(Boolean(tokenRef.current) || isLocalHost);

    const onVisibility = () => {
      const isHidden = document.visibilityState === 'hidden';
      setHidden(isHidden);
      if (!isHidden && wantStreamRef.current) {
        requestWakeLock(); // wake lock is released when the tab is hidden
        if (!wsRef.current) {
          attemptRef.current = 0;
          connectSocket();
        }
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      wantStreamRef.current = false;
      clearTimers();
      closeSocket();
      stopTracks();
      releaseWakeLock();
    };
  }, [connectSocket, requestWakeLock, isLocalHost]);

  // ---------------------------------------------------------------- UI
  const active = ['ready', 'connecting', 'streaming', 'reconnecting'].includes(phase);
  const chipColor =
    phase === 'streaming' ? 'var(--color-green)' : phase === 'error' || phase === 'disconnected' ? 'var(--color-red)' : active ? 'var(--color-orange)' : 'var(--text-dim)';
  const chipBg =
    phase === 'streaming' ? 'var(--color-green-glow)' : phase === 'error' ? 'var(--color-red-glow)' : 'var(--bg-subtle)';
  const chipText = phase === 'reconnecting' ? `${STATE_LABELS.reconnecting} (attempt ${attempt})` : STATE_LABELS[phase];
  const statusColor = serverState.includes('FIRE') ? 'var(--color-red)' : serverState.includes('SMOKE') ? 'var(--color-orange)' : serverState === 'UNCERTAIN' || serverState === 'POSSIBLE_FIRE' ? 'var(--color-orange)' : 'var(--color-green)';

  return (
    <div
      style={{
        maxWidth: '480px',
        margin: '0 auto',
        padding: '16px max(16px, env(safe-area-inset-right)) max(16px, env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left))',
        minHeight: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        boxSizing: 'border-box',
        overflowX: 'hidden',
      }}
    >
      {!apiOk && (
        <div role="alert" style={{ background: 'var(--color-red)', color: '#fff', padding: '12px', borderRadius: '8px', marginBottom: '14px', fontSize: '0.85rem' }}>
          <AlertTriangle size={18} style={{ display: 'inline', verticalAlign: 'text-bottom', marginRight: '6px' }} />
          <strong>Camera needs HTTPS.</strong> Android Chrome blocks the camera on plain http:// pages. Open the <code>https://</code> link from the laptop dashboard (Live Camera page) or scan its QR code.
        </div>
      )}

      {apiOk && !hasToken && (
        <div role="alert" style={{ background: 'var(--color-red-glow)', border: '1px solid var(--color-red)', color: 'var(--color-red)', padding: '12px', borderRadius: '8px', marginBottom: '14px', fontSize: '0.85rem' }}>
          <ShieldAlert size={18} style={{ display: 'inline', verticalAlign: 'text-bottom', marginRight: '6px' }} />
          <strong>Access token missing.</strong> Scan the QR code on the laptop dashboard (Live Camera page) so this phone is authorised.
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', gap: '10px' }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '700' }}>Phone Camera Streamer</h2>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Mobile edge sensor · {facing === 'environment' ? 'rear camera' : 'front camera'}</span>
        </div>
        <span
          data-testid="connection-chip"
          style={{ fontSize: '0.75rem', padding: '5px 10px', borderRadius: '999px', backgroundColor: chipBg, color: chipColor, fontWeight: '600', whiteSpace: 'nowrap' }}
        >
          {phase === 'streaming' ? `${fps} FPS · LIVE` : chipText}
        </span>
      </div>

      <div
        style={{
          flex: 1,
          width: '100%',
          minHeight: '300px',
          maxHeight: '60dvh',
          backgroundColor: '#000',
          borderRadius: '12px',
          overflow: 'hidden',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid var(--border-color)',
          marginBottom: '14px',
        }}
      >
        <video ref={videoRef} playsInline muted autoPlay style={{ width: '100%', height: '100%', objectFit: 'cover', display: active ? 'block' : 'none' }} />
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {!active && (
          <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '20px' }}>
            <Camera size={48} style={{ margin: '0 auto 10px', opacity: 0.4 }} />
            <div>Camera is stopped.</div>
            <div style={{ fontSize: '0.8rem', marginTop: '6px' }}>Tap Start Camera to stream to the laptop.</div>
          </div>
        )}

        {phase === 'reconnecting' && (
          <div style={{ position: 'absolute', top: '10px', left: '10px', right: '10px', background: 'rgba(245,158,11,0.92)', color: '#111', padding: '8px 12px', borderRadius: '8px', fontSize: '0.8rem', fontWeight: 600 }}>
            <RefreshCw size={14} style={{ display: 'inline', verticalAlign: 'text-bottom', marginRight: '6px' }} />
            Connection lost — reconnecting (attempt {attempt}). The camera stays on.
          </div>
        )}

        {active && (
          <div style={{ position: 'absolute', bottom: '12px', left: '12px', right: '12px', backgroundColor: 'rgba(17, 27, 23, 0.88)', backdropFilter: 'blur(6px)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px' }}>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Status from laptop</div>
              <strong style={{ color: statusColor, fontSize: '0.9rem' }}>{serverState.replace(/_/g, ' ')}</strong>
            </div>
            <div style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textAlign: 'right' }}>
              {fps} FPS{latency != null ? ` · ${latency} ms` : ''}
            </div>
          </div>
        )}
      </div>

      {active && reason && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '10px' }}>{reason}</div>
      )}

      {hidden && active && (
        <div style={{ fontSize: '0.78rem', color: 'var(--color-orange)', marginBottom: '10px' }}>
          Tab in background — Android may pause streaming. Keep this page open and the screen on.
        </div>
      )}

      {error && (
        <div role="alert" style={{ padding: '12px', background: 'var(--color-red-glow)', border: '1px solid var(--color-red)', borderRadius: '8px', color: 'var(--color-red)', fontSize: '0.85rem', marginBottom: '12px' }}>
          {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: '10px' }}>
        {!active ? (
          <button
            onClick={startCamera}
            className="sound-toggle-btn active"
            style={{ flex: 1, minHeight: '56px', fontSize: '1.05rem', fontWeight: '700', justifyContent: 'center', borderRadius: '10px' }}
          >
            <Camera size={22} />
            <span>{phase === 'error' ? 'Try Again' : 'Start Camera'}</span>
          </button>
        ) : (
          <button
            onClick={stopCamera}
            style={{ flex: 1, minHeight: '56px', fontSize: '1.05rem', fontWeight: '700', backgroundColor: 'var(--color-red)', color: '#fff', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}
          >
            <CameraOff size={22} />
            <span>Stop Camera</span>
          </button>
        )}
        {active && canSwitch && (
          <button
            onClick={switchCamera}
            aria-label="Switch camera"
            style={{ minWidth: '56px', minHeight: '56px', borderRadius: '10px', backgroundColor: 'var(--bg-subtle)', color: 'var(--text-main, #fff)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            <SwitchCamera size={22} />
          </button>
        )}
      </div>

      <div style={{ textAlign: 'center', marginTop: '14px', fontSize: '0.74rem', color: 'var(--text-dim)' }}>
        Keep this screen on while streaming.<br />
        Academic prototype — not a certified fire early-warning system.
      </div>
    </div>
  );
}
