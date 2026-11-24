import React, { useState, useEffect } from 'react';
import { Info, Database, AlertCircle, ShieldAlert, Cpu } from 'lucide-react';
import { api } from '../api/client';

export function ModelInfo() {
  const [modelInfo, setModelInfo] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);

  useEffect(() => {
    api.getModelsInfo().then(setModelInfo).catch(() => {});
    api.getModelsStatus().then(setModelStatus).catch(() => {});
  }, []);

  const cnnStatus = modelStatus?.cnn;
  const yoloStatus = modelStatus?.yolo;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Model Specifications & Provenance</h1>
        <p className="page-subtitle">Transparent technical disclosures, training methodology, and honest operational limits</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px', marginBottom: '22px' }}>
        {/* CNN Specs */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Cpu size={18} color="var(--color-green)" />
              MobileNetV2 CNN Classifier
            </h3>
            <span style={{ fontSize: '0.75rem', padding: '3px 8px', borderRadius: '4px', backgroundColor: 'var(--bg-subtle)' }}>
              {cnnStatus?.trained_by || 'Not trained yet'}
            </span>
          </div>

          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '14px' }}>
            Lightweight inverted residual architecture optimized for CPU edge classification. Evaluates the full frame to answer: <em>"Is fire present anywhere in this camera viewpoint?"</em>
          </p>

          <div style={{ fontSize: '0.82rem', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div>Input: <strong>224 × 224 (RGB, ImageNet Normalization)</strong></div>
            <div>Classes: <strong>fire, no_fire</strong> (Smoke-only excluded to prevent label noise)</div>
            <div>Format: <strong>ONNX Runtime (CPU Execution Provider)</strong></div>
            <div>File Size: <strong>~9 MB</strong></div>
          </div>
        </div>

        {/* YOLO Specs */}
        <div className="panel">
          <div className="panel-header">
            <h3 className="panel-title">
              <Cpu size={18} color="var(--color-orange)" />
              Ultralytics YOLO Nano Detector
            </h3>
            <span style={{ fontSize: '0.75rem', padding: '3px 8px', borderRadius: '4px', backgroundColor: 'var(--bg-subtle)' }}>
              {yoloStatus?.trained_by || 'Not trained yet'}
            </span>
          </div>

          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '14px' }}>
            Single-stage object detector providing spatial localization. Answers: <em>"Where are the fire and smoke plumes located, and what are their bounding boxes?"</em>
          </p>

          <div style={{ fontSize: '0.82rem', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div>Input: <strong>640 × 640 (Dynamic letterbox scaling)</strong></div>
            <div>Classes: <strong>0: smoke, 1: fire</strong></div>
            <div>Device: <strong>CPU (Single executor thread, 4 cores)</strong></div>
            <div>Starting Weights: <strong>yolo11n.pt</strong></div>
          </div>
        </div>
      </div>

      {/* Dataset & Leakage Prevention */}
      <div className="panel" style={{ marginBottom: '22px' }}>
        <h3 className="panel-title" style={{ marginBottom: '12px' }}>
          <Database size={18} color="var(--color-green)" />
          Dataset Integrity & Zero Data-Leakage Protocol
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
          Trained on the public <strong>D-Fire Dataset</strong> (licence not verified).
        </p>
        <ul style={{ fontSize: '0.82rem', color: 'var(--text-main)', paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <li><strong>Zero Data Leakage:</strong> The CNN dataset splits are derived directly from the exact same train/val/test split boundaries as YOLO. No image crosses splits between detection and classification.</li>
          <li><strong>Perceptual Duplicate Detection:</strong> dHash comparison ensures near-identical time-lapse frames are purged across splits.</li>
          <li><strong>Smoke Disambiguation:</strong> Smoke-only frames are strictly excluded from CNN training so the classifier learns pure fire visual signatures without ambiguity.</li>
        </ul>
      </div>

      {/* Honest Limitations */}
      <div className="panel" style={{ borderLeft: '4px solid var(--color-amber)' }}>
        <h3 className="panel-title" style={{ color: 'var(--color-amber)', marginBottom: '12px' }}>
          <ShieldAlert size={18} />
          Documented Real-World Limitations
        </h3>
        <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <li><strong>Sunsets & Sunrises:</strong> Golden hour lighting and saturated orange cloud layers can cause false-positive candidates in both models (mitigated by Decision Engine temporal confirmation).</li>
          <li><strong>Orange/Red Vehicles:</strong> Emergency trucks and reflective gear can trigger YOLO fire proposals, requiring CNN confirmation to validate.</li>
          <li><strong>Fog vs Thin Smoke:</strong> Distant valley fog and light haze share spectral characteristics with dispersed smoke plumes.</li>
          <li><strong>Nighttime Detection:</strong> Unlit nighttime forest terrain limits detection to active visible flames and luminous glowing embers.</li>
          <li><strong>CPU Compute Constraint:</strong> CPU-only execution caps inference at ~3–5 FPS at 640×480 resolution.</li>
        </ul>
      </div>
    </div>
  );
}
