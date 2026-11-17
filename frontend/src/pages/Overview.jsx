import React, { useState, useEffect } from 'react';
import { Flame, CloudRain, ShieldCheck, AlertCircle, Bell, ArrowRight, Eye } from 'lucide-react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';

export function Overview() {
  const [analytics, setAnalytics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [networkInfo, setNetworkInfo] = useState(null);

  useEffect(() => {
    function loadData() {
      api.getAnalytics().then(setAnalytics).catch(() => {});
      api.getAlerts(5).then(res => setAlerts(res.alerts || [])).catch(() => {});
      api.getNetworkInfo().then(setNetworkInfo).catch(() => {});
    }
    loadData();
    const interval = setInterval(loadData, 3000);
    return () => clearInterval(interval);
  }, []);

  const counters = analytics?.counters || {};
  const framesProcessed = counters.frames_processed || 0;
  const fireDetections = counters.fire_detections || 0;
  const smokeDetections = counters.smoke_detections || 0;
  const noFireClassifications = counters.no_fire_classifications || 0;
  const totalAlerts = counters.alerts_generated || 0;

  return (
    <div className="page-container">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Mission Control & System Overview</h1>
          <p className="page-subtitle">Real-time Wildfire Surveillance & Multi-Model Inference Dashboard</p>
        </div>
        <Link to="/live" className="nav-link active" style={{ display: 'inline-flex', padding: '10px 18px', borderRadius: '8px' }}>
          <Eye size={18} />
          <span>Open Live Stream</span>
          <ArrowRight size={16} />
        </Link>
      </div>

      {/* Real KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card" style={{ borderLeft: '4px solid var(--color-green)' }}>
          <span className="kpi-label">Frames Processed</span>
          <span className="kpi-value">{framesProcessed.toLocaleString()}</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>Genuine CPU Inference</span>
        </div>

        <div className="kpi-card" style={{ borderLeft: '4px solid var(--color-red)' }}>
          <span className="kpi-label">Fire Detections</span>
          <span className="kpi-value" style={{ color: fireDetections > 0 ? 'var(--color-red)' : 'inherit' }}>
            {fireDetections.toLocaleString()}
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>YOLO / CNN Confirmed</span>
        </div>

        <div className="kpi-card" style={{ borderLeft: '4px solid var(--color-orange)' }}>
          <span className="kpi-label">Smoke Detections</span>
          <span className="kpi-value" style={{ color: smokeDetections > 0 ? 'var(--color-orange)' : 'inherit' }}>
            {smokeDetections.toLocaleString()}
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>YOLO Localized Plumes</span>
        </div>

        <div className="kpi-card" style={{ borderLeft: '4px solid var(--color-green)' }}>
          <span className="kpi-label">No-Fire Frames</span>
          <span className="kpi-value">{noFireClassifications.toLocaleString()}</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>Normal Canopy / Sky</span>
        </div>

        <div className="kpi-card" style={{ borderLeft: '4px solid var(--color-amber)' }}>
          <span className="kpi-label">Alerts Generated</span>
          <span className="kpi-value">{totalAlerts.toLocaleString()}</span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '4px' }}>With Cooldown Suppression</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '22px' }}>
        {/* Live Stream Preview */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Eye size={18} color="var(--color-green)" />
              Live Camera Feed Preview
            </h3>
            <span style={{ fontSize: '0.8rem', color: 'var(--color-green)', background: 'var(--color-green-glow)', padding: '3px 8px', borderRadius: '4px' }}>
              MJPEG Stream
            </span>
          </div>
          <div style={{ width: '100%', height: '280px', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <img
              src="/api/stream/annotated"
              alt="Live Annotated Stream"
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            <span>Connect mobile phone via <code>/camera</code> page over HTTPS</span>
            <Link to="/live" style={{ color: 'var(--color-green)', fontWeight: '600' }}>Full Stream View →</Link>
          </div>
        </div>

        {/* Recent Alerts Feed */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Bell size={18} color="var(--color-amber)" />
              Recent Alert Events
            </h3>
            <Link to="/history" style={{ fontSize: '0.82rem', color: 'var(--color-green)' }}>View All Logs →</Link>
          </div>

          {alerts.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-dim)' }}>
              <ShieldCheck size={36} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
              <div>No alerts logged yet. System status normal.</div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexFlow: 'column', gap: '10px' }}>
              {alerts.map((alert) => (
                <div
                  key={alert.id}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '8px',
                    backgroundColor: 'var(--bg-subtle)',
                    borderLeft: `4px solid ${alert.type.includes('fire') ? 'var(--color-red)' : 'var(--color-orange)'}`,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <div>
                    <div style={{ fontSize: '0.88rem', fontWeight: '600' }}>
                      {alert.type.toUpperCase().replace('_', ' ')}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      {alert.message || 'Threat detected by vision pipeline'}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right', fontSize: '0.75rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                    {alert.ts ? new Date(alert.ts).toLocaleTimeString() : 'Recent'}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
