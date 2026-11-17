import React, { useState, useEffect } from 'react';
import { Volume2, VolumeX, AlertTriangle, CheckCircle2, X } from 'lucide-react';
import { api } from '../api/client';
import { initAudio } from '../utils/audio';

export function Header({ isWsConnected, activeAlert, dismissAlert, soundEnabled, toggleSound }) {
  const [modelStatus, setModelStatus] = useState(null);
  const [cameraStatus, setCameraStatus] = useState(null);

  useEffect(() => {
    function loadStatus() {
      api.getModelsStatus().then(setModelStatus).catch(() => {});
      api.getCameraStatus().then(setCameraStatus).catch(() => {});
    }
    loadStatus();
    const interval = setInterval(loadStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSoundClick = () => {
    initAudio();
    toggleSound();
  };

  const isCnnReady = modelStatus?.cnn?.available;
  const isYoloReady = modelStatus?.yolo?.available;
  const isCameraActive = cameraStatus?.active;

  return (
    <>
      {activeAlert && (
        <div className="alert-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <AlertTriangle size={22} />
            <span>
              <strong>EMERGENCY ALERT:</strong> {activeAlert.message || activeAlert.type}
            </span>
          </div>
          <button
            onClick={dismissAlert}
            style={{ color: '#fff', background: 'rgba(0,0,0,0.2)', padding: '4px 10px', borderRadius: '4px' }}
          >
            Acknowledge & Dismiss
          </button>
        </div>
      )}

      <header className="app-header">
        <div className="header-status-group">
          <div className="status-chip">
            <span className={`status-dot ${isWsConnected ? 'green' : 'red'}`} />
            <span>Backend {isWsConnected ? 'Online' : 'Reconnecting'}</span>
          </div>
          <div className="status-chip">
            <span className={`status-dot ${isCnnReady ? 'green' : 'gray'}`} />
            <span>CNN {isCnnReady ? 'Ready' : 'Unavailable'}</span>
          </div>
          <div className="status-chip">
            <span className={`status-dot ${isYoloReady ? 'green' : 'gray'}`} />
            <span>YOLO {isYoloReady ? 'Ready' : 'Unavailable'}</span>
          </div>
          <div className="status-chip">
            <span className={`status-dot ${isCameraActive ? 'green' : 'gray'}`} />
            <span>Camera {isCameraActive ? 'Active' : 'Idle'}</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={handleSoundClick}
            className={`sound-toggle-btn ${soundEnabled ? 'active' : ''}`}
            title={soundEnabled ? 'Alert Siren Enabled' : 'Alert Siren Muted'}
          >
            {soundEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            <span>{soundEnabled ? 'Siren On' : 'Enable Sound'}</span>
          </button>
        </div>
      </header>
    </>
  );
}
