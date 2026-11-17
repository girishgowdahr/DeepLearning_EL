import React, { useState } from 'react';
import { Upload, Image as ImageIcon, Flame, CheckCircle, AlertTriangle, Clock } from 'lucide-react';
import { api } from '../api/client';

export function ImageDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.detectImage(selectedFile);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Still Image Threat Detection</h1>
        <p className="page-subtitle">Upload aerial, surveillance, or ground forest photos for dual-model analysis</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '22px' }}>
        {/* Upload Box */}
        <div className="panel">
          <div
            style={{
              border: '2px dashed var(--border-light)',
              borderRadius: '10px',
              padding: '40px 20px',
              textAlign: 'center',
              backgroundColor: 'var(--bg-subtle)',
              cursor: 'pointer',
              marginBottom: '16px'
            }}
            onClick={() => document.getElementById('imageFileInput').click()}
          >
            <Upload size={36} color="var(--color-green)" style={{ margin: '0 auto 12px' }} />
            <h4 style={{ marginBottom: '6px' }}>Select or Drag Forest Image</h4>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Supports JPG, PNG, WEBP (drone shots, lookout camera stills)
            </p>
            <input
              id="imageFileInput"
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
          </div>

          {selectedFile && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', background: 'var(--bg-card)', padding: '10px 14px', borderRadius: '6px' }}>
              <span style={{ fontSize: '0.85rem' }}>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
              <button
                onClick={handleUpload}
                disabled={loading}
                className="sound-toggle-btn active"
                style={{ padding: '8px 16px' }}
              >
                {loading ? 'Analyzing...' : 'Run Detection'}
              </button>
            </div>
          )}

          {error && (
            <div style={{ padding: '10px', backgroundColor: 'var(--color-red-glow)', border: '1px solid var(--color-red)', borderRadius: '6px', color: 'var(--color-red)', fontSize: '0.85rem' }}>
              {error}
            </div>
          )}
        </div>

        {/* Results Box */}
        {result && (
          <div className="panel">
            <h3 className="panel-title" style={{ marginBottom: '16px' }}>
              Analysis Results & Annotated Frame
            </h3>

            <div style={{ marginBottom: '16px', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#000' }}>
              <img
                src={result.annotated_image_b64}
                alt="Annotated Result"
                style={{ width: '100%', maxHeight: '340px', objectFit: 'contain' }}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {/* Decision State */}
              <div style={{ padding: '10px 14px', borderRadius: '6px', backgroundColor: 'var(--bg-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Fusion Decision:</span>
                <strong style={{ color: result.decision.status.includes('FIRE') ? 'var(--color-red)' : 'var(--color-green)' }}>
                  {result.decision.status}
                </strong>
              </div>

              {/* Reason */}
              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', background: 'var(--bg-card)', padding: '8px 12px', borderRadius: '6px' }}>
                <strong>Reason: </strong> {result.decision.reason}
              </div>

              {/* CNN vs YOLO metrics */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.82rem' }}>
                <div style={{ background: 'var(--bg-subtle)', padding: '10px', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>CNN Classifier:</div>
                  <div>Fire Prob: <strong>{result.cnn?.probs?.fire ? (result.cnn.probs.fire * 100).toFixed(1) : 0}%</strong></div>
                  <div>Label: <strong>{result.cnn?.label}</strong></div>
                </div>

                <div style={{ background: 'var(--bg-subtle)', padding: '10px', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted)', marginBottom: '4px' }}>YOLO Localizer:</div>
                  <div>Boxes: <strong>{result.yolo?.length || 0}</strong></div>
                  <div>Latency: <strong>{result.latency_ms} ms</strong></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
