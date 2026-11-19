import React, { useState, useEffect } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { Video, Smartphone, Camera, Radio, Copy, Check, RefreshCw, AlertTriangle } from 'lucide-react';
import { api } from '../api/client';
import { useEvents } from '../hooks/useEvents';

export function LiveCamera() {
  const { latestDetection, cameraEvent } = useEvents();
  const [networkInfo, setNetworkInfo] = useState(null);
  const [copied, setCopied] = useState(false);
  const [ipUrl, setIpUrl] = useState('http://192.168.1.100:8080/video');
  const [selectedSource, setSelectedSource] = useState('browser');
  const [cameraStatus, setCameraStatus] = useState(null);

  useEffect(() => {
    api.getNetworkInfo().then(setNetworkInfo).catch(() => {});
    api.getCameraStatus().then(setCameraStatus).catch(() => {});
    const interval = setInterval(() => {
      api.getCameraStatus().then(setCameraStatus).catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const phoneUrl = networkInfo?.phone_camera_url || `https://${window.location.hostname}:8000/camera`;
  // Show the link without the secret token; the QR code and Copy button carry the full link.
  const phoneUrlDisplay = phoneUrl.replace(/#t=.*$/, '#t=••••••••');
  const isTunnel = Boolean(networkInfo?.is_tunnel_active);

  // Phone connection state, driven by /ws/events camera events + polled /api/camera/status
  const phoneActive = cameraStatus?.source === 'browser';
  let phoneState = 'waiting';
  if (cameraEvent && cameraEvent.source === 'phone_camera' && cameraEvent.connected === false && !(phoneActive && cameraStatus?.connected)) {
    phoneState = 'disconnected';
  } else if (phoneActive && cameraStatus?.connected) {
    phoneState = 'streaming';
  } else if (phoneActive && cameraStatus && !cameraStatus.connected && cameraStatus.last_frame_age_s < 999) {
    phoneState = 'disconnected';
  }
  const phoneBadge = {
    waiting: { text: 'Waiting for phone camera…', color: 'var(--text-muted)' },
    streaming: { text: 'Android phone · Streaming', color: 'var(--color-green)' },
    disconnected: { text: 'Android phone · Disconnected / reconnecting', color: 'var(--color-orange)' },
  }[phoneState];
  const sourceLabel = { browser: 'Android phone (primary)', webcam: 'Laptop webcam (fallback)', ip: 'IP camera', none: 'None' }[cameraStatus?.source || 'none'] || cameraStatus?.source;

  const handleCopyUrl = () => {
    navigator.clipboard.writeText(phoneUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleStartSource = async (source) => {
    setSelectedSource(source);
    try {
      if (source === 'browser') {
        await api.startCamera('browser');
      } else if (source === 'ip') {
        await api.startCamera('ip', ipUrl);
      } else if (source === 'webcam') {
        await api.startCamera('webcam', null, 0);
      }
      const st = await api.getCameraStatus();
      setCameraStatus(st);
    } catch (e) {
      alert(`Failed to start camera source: ${e.message}`);
    }
  };

  const handleStopSource = async () => {
    await api.stopCamera();
    const st = await api.getCameraStatus();
    setCameraStatus(st);
  };

  const cnnData = latestDetection?.cnn;
  const yoloData = latestDetection?.yolo || [];
  const decision = latestDetection?.decision;
  const status = latestDetection?.state || 'NO_FIRE';
  const reason = latestDetection?.reason || 'Monitoring canopy...';

  const cnnFireProb = cnnData?.probs?.fire ?? (cnnData?.label === 'fire' ? cnnData?.confidence : 0);
  const cnnNoFireProb = cnnData?.probs?.no_fire ?? (cnnData?.label === 'no_fire' ? cnnData?.confidence : 1);

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Live Vision Stream & Fusion Telemetry</h1>
        <p className="page-subtitle">Real-time inference pipeline fusing MobileNetV2 and YOLO nano</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '22px' }}>
        {/* Main Video View */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          <div className="panel" style={{ padding: '0', overflow: 'hidden', backgroundColor: '#000' }}>
            <div style={{ position: 'relative', width: '100%', minHeight: '440px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <img
                src="/api/stream/annotated"
                alt="Live Stream"
                style={{ width: '100%', maxHeight: '520px', objectFit: 'contain' }}
              />
            </div>
            {/* Live Telemetry Bar */}
            <div style={{ padding: '12px 18px', backgroundColor: 'var(--bg-panel)', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Status: </span>
                <strong style={{ color: status.includes('FIRE') ? 'var(--color-red)' : status.includes('SMOKE') ? 'var(--color-orange)' : 'var(--color-green)' }}>
                  {status}
                </strong>
                <span style={{ margin: '0 8px', color: 'var(--text-dim)' }}>|</span>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{reason}</span>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                {sourceLabel && sourceLabel !== 'None' ? `${sourceLabel} · ` : ''}{latestDetection?.fps ? `${latestDetection.fps} FPS` : 'No frames yet'}
              </div>
            </div>
          </div>

          {/* Model Breakdown Panels */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px' }}>
            {/* CNN Panel */}
            <div className="panel">
              <div className="panel-header">
                <h3 className="panel-title">CNN Classification</h3>
                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                  {cnnData?.latency_ms ? `${cnnData.latency_ms} ms` : '224x224'}
                </span>
              </div>
              <div style={{ marginBottom: '14px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                  <span>Fire Probability</span>
                  <strong>{(cnnFireProb * 100).toFixed(1)}%</strong>
                </div>
                <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--bg-subtle)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${cnnFireProb * 100}%`, height: '100%', backgroundColor: 'var(--color-red)', transition: 'width 0.2s' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                  <span>No-Fire Probability</span>
                  <strong>{(cnnNoFireProb * 100).toFixed(1)}%</strong>
                </div>
                <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--bg-subtle)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${cnnNoFireProb * 100}%`, height: '100%', backgroundColor: 'var(--color-green)', transition: 'width 0.2s' }} />
                </div>
              </div>
            </div>

            {/* YOLO Panel */}
            <div className="panel">
              <div className="panel-header">
                <h3 className="panel-title">YOLO Detection</h3>
                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-dim)' }}>
                  {yoloData.length} Box{yoloData.length !== 1 ? 'es' : ''}
                </span>
              </div>
              {yoloData.length === 0 ? (
                <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem', textAlign: 'center', padding: '16px 0' }}>
                  No objects localized in current frame
                </div>
              ) : (
                <div style={{ display: 'flex', flexFlow: 'column', gap: '6px', maxHeight: '110px', overflowY: 'auto' }}>
                  {yoloData.map((d, i) => (
                    <div
                      key={i}
                      style={{
                        padding: '6px 10px',
                        borderRadius: '4px',
                        backgroundColor: 'var(--bg-subtle)',
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.82rem'
                      }}
                    >
                      <span style={{ color: d.cls === 'fire' ? 'var(--color-red)' : 'var(--color-orange)', fontWeight: '600' }}>
                        {d.cls.toUpperCase()}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{(d.conf * 100).toFixed(1)}%</span>
                      <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>
                        [{d.xyxy?.join(', ')}]
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Sidebar: Phone QR Connect & Source Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Phone Connect QR Card */}
          <div className="panel" style={{ textAlign: 'center' }}>
            <h3 className="panel-title" style={{ justifyContent: 'center', marginBottom: '8px' }}>
              <Smartphone size={18} color="var(--color-green)" />
              Connect Phone Camera
            </h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '10px' }}>
              Primary source: <strong>Android phone</strong>. Scan with the phone to open the HTTPS camera page, then tap Start Camera.
            </p>
            <div data-testid="phone-status" style={{ display: 'inline-block', padding: '4px 12px', borderRadius: '999px', fontSize: '0.78rem', fontWeight: 600, marginBottom: '12px', backgroundColor: 'var(--bg-subtle)', color: phoneBadge.color }}>
              {phoneBadge.text}
            </div>
            {!isTunnel && (
              <p style={{ fontSize: '0.74rem', color: 'var(--color-orange)', marginBottom: '10px' }}>
                No HTTPS tunnel detected. Start <code>scripts\run_demo.ps1</code> (tunnel) or <code>-SelfSigned</code>; plain http:// cannot open the phone camera.
              </p>
            )}

            <div style={{ padding: '14px', background: '#fff', borderRadius: '10px', display: 'inline-block', margin: '0 auto 14px' }}>
              <QRCodeSVG value={phoneUrl} size={150} />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'var(--bg-subtle)', padding: '6px 10px', borderRadius: '6px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', overflow: 'hidden' }}>
              <span style={{ flex: 1, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                {phoneUrlDisplay}
              </span>
              <button onClick={handleCopyUrl} style={{ color: copied ? 'var(--color-green)' : 'var(--text-muted)' }}>
                {copied ? <Check size={16} /> : <Copy size={16} />}
              </button>
            </div>
          </div>

          {/* Alternative Camera Sources */}
          <div className="panel">
            <h3 className="panel-title" style={{ marginBottom: '12px' }}>
              <Camera size={18} />
              Alternative Sources
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.85rem' }}>
              <button
                onClick={() => handleStartSource('webcam')}
                className="sound-toggle-btn"
                style={{ justifyContent: 'center', width: '100%' }}
              >
                Use Laptop Webcam (Device 0)
              </button>

              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '10px', marginTop: '4px' }}>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                  IP Webcam App (MJPEG URL):
                </label>
                <input
                  type="text"
                  value={ipUrl}
                  onChange={(e) => setIpUrl(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    backgroundColor: 'var(--bg-subtle)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: '#fff',
                    fontSize: '0.82rem',
                    marginBottom: '8px'
                  }}
                />
                <button
                  onClick={() => handleStartSource('ip')}
                  className="sound-toggle-btn"
                  style={{ justifyContent: 'center', width: '100%' }}
                >
                  Connect IP Webcam
                </button>
              </div>

              {cameraStatus?.active && (
                <button
                  onClick={handleStopSource}
                  style={{ color: 'var(--color-red)', fontSize: '0.8rem', padding: '8px', textAlign: 'center' }}
                >
                  Stop Active Stream
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
