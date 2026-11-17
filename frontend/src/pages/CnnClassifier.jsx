import React, { useState, useEffect } from 'react';
import { Cpu, Upload, CheckCircle2, AlertCircle } from 'lucide-react';
import { api } from '../api/client';

export function CnnClassifier() {
  const [modelStatus, setModelStatus] = useState(null);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getModelsStatus().then(res => setModelStatus(res.cnn)).catch(() => {});
  }, []);

  const handleTest = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await api.detectImage(file);
      setResult(res.cnn);
    } catch (e) {
      alert(`CNN test failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fireProb = result?.probs?.fire ?? 0;
  const noFireProb = result?.probs?.no_fire ?? 1;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">CNN Classification Subsystem (MobileNetV2)</h1>
        <p className="page-subtitle">Lightweight deep convolutional neural network for full-frame binary classification</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px' }}>
        {/* Architecture Panel */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Cpu size={18} color="var(--color-green)" />
              Architecture & Execution Specs
            </h3>
            <span style={{ fontSize: '0.8rem', color: modelStatus?.available ? 'var(--color-green)' : 'var(--text-dim)' }}>
              {modelStatus?.available ? 'Loaded & Active' : 'Model Missing'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.88rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Base Backbone:</span>
              <strong>MobileNetV2 (ImageNet Pretrained)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Input Dimensions:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>224 × 224 × 3</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Normalization:</span>
              <span>ImageNet (mean=[0.485, 0.456, 0.406])</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Head Architecture:</span>
              <span>Dropout(0.2) → Linear(1280, 2)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Inference Engine:</span>
              <strong>ONNX Runtime (CPU Provider)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Trained By:</span>
              <span>{modelStatus?.trained_by || 'Pending / Check model_weights/'}</span>
            </div>
          </div>
        </div>

        {/* Live CNN Test Panel */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '14px' }}>Test Single Image on CNN</h3>
          
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

          <button
            onClick={handleTest}
            disabled={!file || loading}
            className="sound-toggle-btn active"
            style={{ width: '100%', justifyContent: 'center', marginBottom: '18px' }}
          >
            {loading ? 'Evaluating...' : 'Run CNN Inference'}
          </button>

          {result && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                  <span>Fire Probability</span>
                  <strong>{(fireProb * 100).toFixed(2)}%</strong>
                </div>
                <div style={{ width: '100%', height: '10px', backgroundColor: 'var(--bg-subtle)', borderRadius: '5px', overflow: 'hidden' }}>
                  <div style={{ width: `${fireProb * 100}%`, height: '100%', backgroundColor: 'var(--color-red)' }} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                  <span>No-Fire Probability</span>
                  <strong>{(noFireProb * 100).toFixed(2)}%</strong>
                </div>
                <div style={{ width: '100%', height: '10px', backgroundColor: 'var(--bg-subtle)', borderRadius: '5px', overflow: 'hidden' }}>
                  <div style={{ width: `${noFireProb * 100}%`, height: '100%', backgroundColor: 'var(--color-green)' }} />
                </div>
              </div>

              <div style={{ background: 'var(--bg-subtle)', padding: '12px', borderRadius: '6px', fontSize: '0.82rem', display: 'flex', justifyContent: 'space-between' }}>
                <span>Inference Latency:</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{result.latency_ms} ms</strong>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
