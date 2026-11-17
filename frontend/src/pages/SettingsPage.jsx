import React, { useState, useEffect } from 'react';
import { Settings, Sliders, Volume2, Bell, RefreshCw, Check, Smartphone } from 'lucide-react';
import { api } from '../api/client';
import { initAudio, playFireSiren } from '../utils/audio';

export function SettingsPage() {
  const [settings, setSettings] = useState(null);
  const [saved, setSaved] = useState(false);
  const [testingSound, setTestingSound] = useState(false);

  useEffect(() => {
    api.getSettings().then(setSettings).catch(() => {});
  }, []);

  const handleChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    try {
      await api.updateSettings(settings);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      alert(`Save failed: ${e.message}`);
    }
  };

  const handleTestSound = async () => {
    initAudio();
    playFireSiren();
    setTestingSound(true);
    try {
      await api.triggerTestAlert();
    } catch (_) {}
    setTimeout(() => setTestingSound(false), 2000);
  };

  const handleReloadModels = async () => {
    try {
      const res = await api.reloadModels();
      alert(`Models reloaded! CNN Available: ${res.status.cnn.available}, YOLO Available: ${res.status.yolo.available}`);
    } catch (e) {
      alert(`Reload failed: ${e.message}`);
    }
  };

  if (!settings) {
    return <div className="page-container">Loading settings...</div>;
  }

  return (
    <div className="page-container">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">System Settings & Decision Thresholds</h1>
          <p className="page-subtitle">Configure fusion weights, temporal sliding window confirmation, and alert channels</p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button onClick={handleReloadModels} className="sound-toggle-btn" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <RefreshCw size={16} />
            <span>Reload Model Files</span>
          </button>
          <button onClick={handleSave} className="sound-toggle-btn active" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {saved ? <Check size={16} /> : <Settings size={16} />}
            <span>{saved ? 'Saved!' : 'Save Changes'}</span>
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px' }}>
        {/* Threshold Sliders Panel */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '18px' }}>
            <Sliders size={18} color="var(--color-green)" />
            Per-Frame Confidence Thresholds
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span>YOLO Fire Confidence Threshold:</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{(settings.yolo_conf_fire * 100).toFixed(0)}%</strong>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.80"
                step="0.05"
                value={settings.yolo_conf_fire}
                onChange={(e) => handleChange('yolo_conf_fire', parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span>YOLO Smoke Confidence Threshold:</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{(settings.yolo_conf_smoke * 100).toFixed(0)}%</strong>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.80"
                step="0.05"
                value={settings.yolo_conf_smoke}
                onChange={(e) => handleChange('yolo_conf_smoke', parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span>CNN Fire Agreement Threshold:</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{(settings.cnn_threshold * 100).toFixed(0)}%</strong>
              </div>
              <input
                type="range"
                min="0.40"
                max="0.90"
                step="0.05"
                value={settings.cnn_threshold}
                onChange={(e) => handleChange('cnn_threshold', parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span>CNN Strong Fire (Unlocalized) Threshold:</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{(settings.cnn_strong_threshold * 100).toFixed(0)}%</strong>
              </div>
              <input
                type="range"
                min="0.75"
                max="0.99"
                step="0.01"
                value={settings.cnn_strong_threshold}
                onChange={(e) => handleChange('cnn_strong_threshold', parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>
        </div>

        {/* Temporal Confirmation & Cooldown */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '18px' }}>
            <Bell size={18} color="var(--color-green)" />
            Temporal Window & Siren Controls
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span>Sliding Window Size (M frames):</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{settings.window_size} frames</strong>
              </div>
              <input
                type="range"
                min="3"
                max="12"
                step="1"
                value={settings.window_size}
                onChange={(e) => handleChange('window_size', parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span>Confirmation Threshold (N frames):</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{settings.confirm_frames} frames</strong>
              </div>
              <input
                type="range"
                min="2"
                max="6"
                step="1"
                value={settings.confirm_frames}
                onChange={(e) => handleChange('confirm_frames', parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                  Clear Time (sec):
                </label>
                <input
                  type="number"
                  value={settings.clear_seconds}
                  onChange={(e) => handleChange('clear_seconds', parseFloat(e.target.value))}
                  style={{ width: '100%', padding: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border-color)', color: '#fff', borderRadius: '4px' }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '4px' }}>
                  Cooldown (sec):
                </label>
                <input
                  type="number"
                  value={settings.cooldown_seconds}
                  onChange={(e) => handleChange('cooldown_seconds', parseFloat(e.target.value))}
                  style={{ width: '100%', padding: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border-color)', color: '#fff', borderRadius: '4px' }}
                />
              </div>
            </div>

            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '14px', marginTop: '6px' }}>
              <button
                onClick={handleTestSound}
                className="sound-toggle-btn"
                style={{ width: '100%', justifyContent: 'center', color: 'var(--color-amber)', borderColor: 'var(--color-amber)' }}
              >
                <Volume2 size={16} />
                <span>{testingSound ? 'Playing Siren...' : 'Test Siren Sound & Alert Feed'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Alert Notifiers Status */}
      <div className="panel" style={{ marginTop: '22px' }}>
        <h3 className="panel-title" style={{ marginBottom: '14px' }}>Alert Channels & Hardware Integration</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div style={{ padding: '14px', background: 'var(--bg-subtle)', borderRadius: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <strong>Browser Audio Siren & Feed</strong>
              <span style={{ color: 'var(--color-green)', fontSize: '0.75rem', fontWeight: '600' }}>ACTIVE</span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Web Audio API 880/660 Hz alternating oscillator alert with user-gesture unlock.
            </p>
          </div>

          <div style={{ padding: '14px', background: 'var(--bg-subtle)', borderRadius: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <strong>Telegram Bot Notifications</strong>
              <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>PLANNED / STUB</span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Modular stub in <code>backend/services/notifiers/telegram.py</code>. Configure in <code>.env</code>.
            </p>
          </div>

          <div style={{ padding: '14px', background: 'var(--bg-subtle)', borderRadius: '6px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <strong>ESP32 Physical IoT Buzzer</strong>
              <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>PLANNED / STUB</span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Modular HTTP buzzer webhook contract documented in <code>docs/esp32_buzzer.md</code>.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
