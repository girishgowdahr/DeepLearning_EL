import React, { useState, useEffect } from 'react';
import { Scan, Upload } from 'lucide-react';
import { api } from '../api/client';

export function YoloDetection() {
  const [modelStatus, setModelStatus] = useState(null);
  const [file, setFile] = useState(null);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getModelsStatus().then(res => setModelStatus(res.yolo)).catch(() => {});
  }, []);

  const handleTest = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await api.detectImage(file);
      setResult(res);
    } catch (e) {
      alert(`YOLO detection failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const detections = result?.yolo || [];

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">YOLO Object Detection Subsystem</h1>
        <p className="page-subtitle">Single-stage deep neural network for spatial bounding box localization</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px' }}>
        {/* Architecture Panel */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Scan size={18} color="var(--color-green)" />
              Detector Architecture
            </h3>
            <span style={{ fontSize: '0.8rem', color: modelStatus?.available ? 'var(--color-green)' : 'var(--text-dim)' }}>
              {modelStatus?.available ? 'Loaded & Active' : 'Model Missing'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.88rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Architecture:</span>
              <strong>Ultralytics YOLO Nano (yolo11n)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Inference Image Size:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>416 × 416</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Classes:</span>
              <span>0: Fire, 1: Smoke</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Target Device:</span>
              <span>CPU (Optimized 4-thread execution)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Trained By:</span>
              <span>{modelStatus?.trained_by || 'Pending / Check model_weights/'}</span>
            </div>
          </div>
        </div>

        {/* Live YOLO Test Panel */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '14px' }}>Test Single Image on YOLO</h3>
          
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                setFile(e.target.files[0]);
                setResult(null);
              }
            }}
            style={{ marginBottom: '14px', fontSize: '0.85rem' }}
          />

          <div style={{ marginBottom: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
              <span>Confidence Threshold:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>{(confThreshold * 100).toFixed(0)}%</strong>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <button
            onClick={handleTest}
            disabled={!file || loading}
            className="sound-toggle-btn active"
            style={{ width: '100%', justifyContent: 'center', marginBottom: '18px' }}
          >
            {loading ? 'Detecting...' : 'Run YOLO Inference'}
          </button>

          {result && (
            <div>
              <div style={{ marginBottom: '14px', borderRadius: '6px', overflow: 'hidden', backgroundColor: '#000' }}>
                <img src={result.annotated_image_b64} alt="YOLO Output" style={{ width: '100%', maxHeight: '220px', objectFit: 'contain' }} />
              </div>

              <div style={{ fontSize: '0.82rem', marginBottom: '8px', color: 'var(--text-muted)' }}>
                Detected {detections.length} objects:
              </div>

              {detections.length === 0 ? (
                <div style={{ color: 'var(--text-dim)', fontSize: '0.82rem', textAlign: 'center', padding: '10px' }}>
                  No objects met the threshold
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {detections.map((d, i) => (
                    <div key={i} style={{ padding: '6px 10px', borderRadius: '4px', background: 'var(--bg-subtle)', display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                      <span style={{ color: d.cls === 'fire' ? 'var(--color-red)' : 'var(--color-orange)', fontWeight: '600' }}>{d.cls.toUpperCase()}</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{(d.conf * 100).toFixed(1)}%</span>
                      <span style={{ color: 'var(--text-dim)' }}>[{d.xyxy?.join(', ')}]</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
