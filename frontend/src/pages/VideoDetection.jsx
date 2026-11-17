import React, { useState, useEffect } from 'react';
import { Film, Upload, CheckCircle, AlertTriangle, Play } from 'lucide-react';
import { api } from '../api/client';

export function VideoDetection() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const res = await api.detectVideo(file);
      setJobId(res.job_id);
      setJobStatus({ status: 'processing', progress: 0 });
    } catch (err) {
      setError(err.message || 'Video upload failed');
    } finally {
      setUploading(false);
    }
  };

  useEffect(() => {
    if (!jobId || jobStatus?.status === 'completed' || jobStatus?.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const data = await api.getVideoJobStatus(jobId);
        setJobStatus(data);
      } catch (e) {
        console.error('Job polling error:', e);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [jobId, jobStatus?.status]);

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Video Clip Threat Analysis</h1>
        <p className="page-subtitle">Batch process forest surveillance video clips at ~2 FPS with timeline tracking</p>
      </div>

      <div className="panel" style={{ marginBottom: '22px' }}>
        <div
          style={{
            border: '2px dashed var(--border-light)',
            borderRadius: '10px',
            padding: '34px 20px',
            textAlign: 'center',
            backgroundColor: 'var(--bg-subtle)',
            cursor: 'pointer',
            marginBottom: '16px'
          }}
          onClick={() => document.getElementById('videoInput').click()}
        >
          <Film size={36} color="var(--color-green)" style={{ margin: '0 auto 12px' }} />
          <h4>Upload Video Clip (MP4, AVI, MOV)</h4>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Limit: 50 MB (≤ 60 seconds). Sampled at ~2 FPS for smooth CPU execution.
          </p>
          <input
            id="videoInput"
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                setFile(e.target.files[0]);
                setJobId(null);
                setJobStatus(null);
              }
            }}
          />
        </div>

        {file && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-card)', padding: '10px 14px', borderRadius: '6px' }}>
            <span style={{ fontSize: '0.85rem' }}>{file.name} ({(file.size / (1024 * 1024)).toFixed(1)} MB)</span>
            <button
              onClick={handleUpload}
              disabled={uploading || (jobStatus && jobStatus.status === 'processing')}
              className="sound-toggle-btn active"
            >
              {uploading ? 'Uploading...' : 'Process Video'}
            </button>
          </div>
        )}

        {error && (
          <div style={{ marginTop: '12px', padding: '10px', backgroundColor: 'var(--color-red-glow)', border: '1px solid var(--color-red)', borderRadius: '6px', color: 'var(--color-red)', fontSize: '0.85rem' }}>
            {error}
          </div>
        )}

        {jobStatus && jobStatus.status === 'processing' && (
          <div style={{ marginTop: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
              <span>Analyzing Video Frames...</span>
              <strong>{jobStatus.progress || 0}%</strong>
            </div>
            <div style={{ width: '100%', height: '10px', backgroundColor: 'var(--bg-subtle)', borderRadius: '5px', overflow: 'hidden' }}>
              <div style={{ width: `${jobStatus.progress || 0}%`, height: '100%', backgroundColor: 'var(--color-green)', transition: 'width 0.3s' }} />
            </div>
          </div>
        )}
      </div>

      {/* Completed Results */}
      {jobStatus && jobStatus.status === 'completed' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '22px' }}>
          {/* Summary KPIs */}
          <div className="kpi-grid">
            <div className="kpi-card">
              <span className="kpi-label">Sampled Frames</span>
              <span className="kpi-value">{jobStatus.total_sampled}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Fire Frames</span>
              <span className="kpi-value" style={{ color: jobStatus.fire_frames > 0 ? 'var(--color-red)' : 'inherit' }}>
                {jobStatus.fire_frames}
              </span>
            </div>
            <div className="kpi-card">
              <span className="kpi-label">Smoke Frames</span>
              <span className="kpi-value" style={{ color: jobStatus.smoke_frames > 0 ? 'var(--color-orange)' : 'inherit' }}>
                {jobStatus.smoke_frames}
              </span>
            </div>
          </div>

          {/* Timeline Bar */}
          <div className="panel">
            <h3 className="panel-title" style={{ marginBottom: '14px' }}>Timeline of States</h3>
            <div style={{ display: 'flex', height: '24px', borderRadius: '6px', overflow: 'hidden', backgroundColor: 'var(--bg-subtle)' }}>
              {jobStatus.frames_summary?.map((f, i) => {
                let color = 'var(--color-green)';
                if (f.state.includes('FIRE')) color = 'var(--color-red)';
                else if (f.state.includes('SMOKE')) color = 'var(--color-orange)';
                else if (f.state.includes('UNCERTAIN')) color = 'var(--color-amber)';
                return (
                  <div
                    key={i}
                    title={`${f.time_sec}s: ${f.state}`}
                    style={{ flex: 1, backgroundColor: color }}
                  />
                );
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '6px' }}>
              <span>0s</span>
              <span>Duration Timeline</span>
              <span>{jobStatus.frames_summary?.slice(-1)[0]?.time_sec || 0}s</span>
            </div>
          </div>

          {/* Flagged Previews Gallery */}
          {jobStatus.flagged_previews?.length > 0 && (
            <div className="panel">
              <h3 className="panel-title" style={{ marginBottom: '14px' }}>Flagged Threat Frames</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '14px' }}>
                {jobStatus.flagged_previews.map((item, idx) => (
                  <div key={idx} style={{ borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-color)', backgroundColor: '#000' }}>
                    <img src={item.b64} alt={`Flagged ${item.time_sec}s`} style={{ width: '100%', height: '140px', objectFit: 'cover' }} />
                    <div style={{ padding: '8px 10px', fontSize: '0.78rem', display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: item.state.includes('FIRE') ? 'var(--color-red)' : 'var(--color-orange)', fontWeight: '600' }}>
                        {item.state}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{item.time_sec}s</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
