import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { BarChart3, LineChart, FileText, CheckCircle2 } from 'lucide-react';
import { api } from '../api/client';

export function Analytics() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.getAnalytics().then(setData).catch(() => {});
  }, []);

  const counters = data?.counters || {};
  const metrics = data?.model_performance_metrics || {};

  const chartData = [
    { name: 'Fire', count: counters.fire_detections || 0, color: 'var(--color-red)' },
    { name: 'Smoke', count: counters.smoke_detections || 0, color: 'var(--color-orange)' },
    { name: 'No Fire', count: counters.no_fire_classifications || 0, color: 'var(--color-green)' }
  ];

  const cnnMetrics = metrics?.cnn_metrics;
  const yoloMetrics = metrics?.yolo_metrics;

  return (
    <div className="page-container">
      <div className="page-header">
        <h1 className="page-title">Surveillance Analytics & Model Metrics</h1>
        <p className="page-subtitle">Real-time classification counts and verified test-set model evaluation metrics</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px', marginBottom: '24px' }}>
        {/* Detection Breakdown Chart */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '16px' }}>
            <BarChart3 size={18} color="var(--color-green)" />
            Real Detections Distribution
          </h3>
          <div style={{ width: '100%', height: '260px' }}>
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} />
                <YAxis stroke="var(--text-muted)" fontSize={12} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-color)', color: '#fff' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Real Metrics Summary */}
        <div className="panel">
          <h3 className="panel-title" style={{ marginBottom: '16px' }}>
            <FileText size={18} color="var(--color-green)" />
            Verified Test Set Benchmarks
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ padding: '12px', background: 'var(--bg-subtle)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '6px', color: 'var(--color-green)' }}>
                CNN MobileNetV2 (Derived D-Fire Test Split)
              </div>
              {cnnMetrics ? (
                <div style={{ fontSize: '0.82rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                  <div>Test Accuracy: <strong>{(cnnMetrics.accuracy * 100).toFixed(1)}%</strong></div>
                  <div>Macro F1: <strong>{(cnnMetrics.macro_f1 * 100).toFixed(1)}%</strong></div>
                  <div>Fire Precision: <strong>{(cnnMetrics.classes?.fire?.precision * 100).toFixed(1)}%</strong></div>
                  <div>Fire Recall: <strong>{(cnnMetrics.classes?.fire?.recall * 100).toFixed(1)}%</strong></div>
                </div>
              ) : (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                  Training results file <code>results/metrics/cnn_metrics.json</code> not found yet. Complete training to populate.
                </div>
              )}
            </div>

            <div style={{ padding: '12px', background: 'var(--bg-subtle)', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '6px', color: 'var(--color-orange)' }}>
                YOLO Nano (D-Fire Test Split)
              </div>
              {yoloMetrics ? (
                <div style={{ fontSize: '0.82rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                  <div>mAP@50: <strong>{(yoloMetrics.mAP50 * 100).toFixed(1)}%</strong></div>
                  <div>mAP@50-95: <strong>{(yoloMetrics['mAP50-95'] * 100).toFixed(1)}%</strong></div>
                  <div>Precision: <strong>{(yoloMetrics.precision * 100).toFixed(1)}%</strong></div>
                  <div>Recall: <strong>{(yoloMetrics.recall * 100).toFixed(1)}%</strong></div>
                </div>
              ) : (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
                  Training results file <code>results/metrics/yolo_metrics.json</code> not found yet. Complete training to populate.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Training Artifacts Gallery */}
      <div className="panel">
        <h3 className="panel-title" style={{ marginBottom: '16px' }}>
          <LineChart size={18} color="var(--color-green)" />
          Training Performance Curves & Confusion Matrices
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '18px' }}>
          <div style={{ background: 'var(--bg-subtle)', padding: '14px', borderRadius: '8px', textAlign: 'center' }}>
            <h4 style={{ fontSize: '0.85rem', marginBottom: '10px' }}>CNN Confusion Matrix</h4>
            <img
              src="/results/graphs/cnn_confusion_matrix.png"
              alt="CNN Confusion Matrix"
              style={{ width: '100%', maxHeight: '240px', objectFit: 'contain' }}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <div style={{ display: 'none', color: 'var(--text-dim)', fontSize: '0.8rem', padding: '40px 0' }}>
              Not generated yet (Available after training run)
            </div>
          </div>

          <div style={{ background: 'var(--bg-subtle)', padding: '14px', borderRadius: '8px', textAlign: 'center' }}>
            <h4 style={{ fontSize: '0.85rem', marginBottom: '10px' }}>CNN Accuracy & Loss</h4>
            <img
              src="/results/graphs/cnn_accuracy.png"
              alt="CNN Accuracy"
              style={{ width: '100%', maxHeight: '240px', objectFit: 'contain' }}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <div style={{ display: 'none', color: 'var(--text-dim)', fontSize: '0.8rem', padding: '40px 0' }}>
              Not generated yet (Available after training run)
            </div>
          </div>

          <div style={{ background: 'var(--bg-subtle)', padding: '14px', borderRadius: '8px', textAlign: 'center' }}>
            <h4 style={{ fontSize: '0.85rem', marginBottom: '10px' }}>YOLO Precision-Recall & F1</h4>
            <img
              src="/results/graphs/yolo_PR_curve.png"
              alt="YOLO PR Curve"
              style={{ width: '100%', maxHeight: '240px', objectFit: 'contain' }}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <div style={{ display: 'none', color: 'var(--text-dim)', fontSize: '0.8rem', padding: '40px 0' }}>
              Not generated yet (Available after training run)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
