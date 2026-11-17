import React, { useState, useRef, useEffect } from 'react';
import { Camera, CameraOff, Smartphone, AlertTriangle, ShieldCheck, RefreshCw } from 'lucide-react';

export function PhoneCamera() {
  const [streaming, setStreaming] = useState(false);
  const [fps, setFps] = useState(0);
  const [serverState, setServerState] = useState('NO_FIRE');
  const [reason, setReason] = useState('Awaiting stream');
  const [errorMsg, setErrorMsg] = useState(null);
  const [isSecure, setIsSecure] = useState(true);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const wakeLockRef = useRef(null);
  const inFlightRef = useRef(false);
  const animFrameRef = useRef(null);

  useEffect(() => {
    // Check if secure context (HTTPS or localhost)
    const secure = window.isSecureContext || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    setIsSecure(secure);

    return () => {
      stopCamera();
    };
  }, []);

  const requestWakeLock = async () => {
    try {
      if ('wakeLock' in navigator) {
        wakeLockRef.current = await navigator.wakeLock.request('screen');
      }
    } catch (e) {
      console.warn('Wake Lock error:', e);
    }
  };

  const startCamera = async () => {
    setErrorMsg(null);
    try {
      // 1. Get user media (prefer rear environment camera)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      await requestWakeLock();

      // 2. Open WebSocket to /ws/stream
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      let lastFrameTs = performance.now();
      let frameCounter = 0;

      ws.onopen = () => {
        setStreaming(true);
        inFlightRef.current = false;
        sendNextFrame();
      };

      ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          // JSON metadata reply
          try {
            const meta = JSON.parse(event.data);
            if (meta.decision) {
              setServerState(meta.decision.status);
              setReason(meta.decision.reason);
            }
          } catch (_) {}
        } else {
          // Binary JPEG frame reply -> ack received!
          inFlightRef.current = false;
          frameCounter++;
          const now = performance.now();
          if (now - lastFrameTs >= 1000) {
            setFps(frameCounter);
            frameCounter = 0;
            lastFrameTs = now;
          }
          // Schedule next frame with ~5 FPS cap (~200ms interval)
          setTimeout(sendNextFrame, 200);
        }
      };

      ws.onclose = () => {
        setStreaming(false);
      };

      ws.onerror = (e) => {
        console.error('WS Error:', e);
        setErrorMsg('WebSocket connection failed.');
        stopCamera();
      };

    } catch (err) {
      console.error('Camera access error:', err);
      if (err.name === 'NotAllowedError') {
        setErrorMsg('Camera permission was denied. Please grant permission in your browser site settings.');
      } else if (err.name === 'NotFoundError') {
        setErrorMsg('No suitable camera hardware found on this device.');
      } else {
        setErrorMsg(`Camera error: ${err.message}`);
      }
      stopCamera();
    }
  };

  const sendNextFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;
    if (inFlightRef.current) return;

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setTimeout(sendNextFrame, 100);
      return;
    }

    const canvas = canvasRef.current;
    canvas.width = 640;
    canvas.height = Math.round(640 * (video.videoHeight / video.videoWidth));

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    inFlightRef.current = true;

    canvas.toBlob(
      (blob) => {
        if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then((buf) => {
            wsRef.current.send(buf);
          });
        } else {
          inFlightRef.current = false;
        }
      },
      'image/jpeg',
      0.6
    );

    // Timeout safety fallback (if reply gets lost)
    setTimeout(() => {
      if (inFlightRef.current) {
        inFlightRef.current = false;
        sendNextFrame();
      }
    }, 1500);
  };

  const stopCamera = () => {
    setStreaming(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (wakeLockRef.current) {
      wakeLockRef.current.release().catch(() => {});
      wakeLockRef.current = null;
    }
    inFlightRef.current = false;
  };

  return (
    <div style={{ maxWidth: '480px', margin: '0 auto', padding: '16px', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {!isSecure && (
        <div style={{ background: 'var(--color-red)', color: '#fff', padding: '12px', borderRadius: '8px', marginBottom: '14px', fontSize: '0.85rem' }}>
          <AlertTriangle size={18} style={{ display: 'inline', verticalAlign: 'text-bottom', marginRight: '6px' }} />
          <strong>Insecure Context:</strong> Android Chrome requires HTTPS to access the camera. Connect via the Cloudflare Tunnel URL or Self-Signed HTTPS.
        </div>
      )}

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '700' }}>Phone Camera Streamer</h2>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Mobile Edge Sensor Client</span>
        </div>
        <span
          style={{
            fontSize: '0.75rem',
            padding: '4px 10px',
            borderRadius: '999px',
            backgroundColor: streaming ? 'var(--color-green-glow)' : 'var(--bg-subtle)',
            color: streaming ? 'var(--color-green)' : 'var(--text-dim)',
            fontWeight: '600'
          }}
        >
          {streaming ? `${fps} FPS • LIVE` : 'IDLE'}
        </span>
      </div>

      {/* Camera Video Viewport */}
      <div
        style={{
          flex: 1,
          width: '100%',
          minHeight: '340px',
          maxHeight: '480px',
          backgroundColor: '#000',
          borderRadius: '12px',
          overflow: 'hidden',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid var(--border-color)',
          marginBottom: '16px'
        }}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: streaming ? 'block' : 'none' }}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {!streaming && (
          <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '20px' }}>
            <Camera size={48} style={{ margin: '0 auto 10px', opacity: 0.4 }} />
            <div>Camera is currently stopped.</div>
            <div style={{ fontSize: '0.8rem', marginTop: '6px' }}>Tap Start Camera below to stream to laptop.</div>
          </div>
        )}

        {/* Live Server Feedback Overlay */}
        {streaming && (
          <div
            style={{
              position: 'absolute',
              bottom: '12px',
              left: '12px',
              right: '12px',
              backgroundColor: 'rgba(17, 27, 23, 0.88)',
              backdropFilter: 'blur(6px)',
              padding: '10px 14px',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Status from Laptop:</div>
              <strong style={{ color: serverState.includes('FIRE') ? 'var(--color-red)' : serverState.includes('SMOKE') ? 'var(--color-orange)' : 'var(--color-green)', fontSize: '0.9rem' }}>
                {serverState}
              </strong>
            </div>
            <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              {fps} FPS
            </div>
          </div>
        )}
      </div>

      {errorMsg && (
        <div style={{ padding: '12px', background: 'var(--color-red-glow)', border: '1px solid var(--color-red)', borderRadius: '8px', color: 'var(--color-red)', fontSize: '0.85rem', marginBottom: '14px' }}>
          {errorMsg}
        </div>
      )}

      {/* Big Action Controls */}
      <div>
        {!streaming ? (
          <button
            onClick={startCamera}
            className="sound-toggle-btn active"
            style={{
              width: '100%',
              padding: '16px',
              fontSize: '1.05rem',
              fontWeight: '700',
              justifyContent: 'center',
              borderRadius: '10px'
            }}
          >
            <Camera size={22} />
            <span>Start Camera Stream</span>
          </button>
        ) : (
          <button
            onClick={stopCamera}
            style={{
              width: '100%',
              padding: '16px',
              fontSize: '1.05rem',
              fontWeight: '700',
              backgroundColor: 'var(--color-red)',
              color: '#fff',
              borderRadius: '10px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px'
            }}
          >
            <CameraOff size={22} />
            <span>Stop Camera Stream</span>
          </button>
        )}
      </div>

      <div style={{ textAlign: 'center', marginTop: '16px', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
        Backpressure: Client sends frames only upon receiving server ack.<br />
        Screen kept awake via Wake Lock API.
      </div>
    </div>
  );
}
