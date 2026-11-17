import React, { useState, useEffect } from 'react';
import { History, Download, Eye, X, Filter } from 'lucide-react';
import { api } from '../api/client';

export function DetectionHistory() {
  const [detections, setDetections] = useState([]);
  const [sourceFilter, setSourceFilter] = useState('');
  const [stateFilter, setStateFilter] = useState('');
  const [modalSnapshot, setModalSnapshot] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadDetections();
  }, [sourceFilter, stateFilter]);

  const loadDetections = async () => {
    setLoading(true);
    try {
      const res = await api.getDetections(100, 0, sourceFilter || null, stateFilter || null);
      setDetections(res.detections || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleExportCSV = () => {
    if (detections.length === 0) return;
    const headers = ['ID', 'Timestamp', 'Source', 'State', 'CNN Label', 'CNN Conf', 'Top Class', 'Top Conf', 'Snapshot'];
    const rows = detections.map(d => [
      d.id,
      d.ts,
      d.source,
      d.state,
      d.cnn_label || '',
      d.cnn_conf || '',
      d.top_class || '',
      d.top_conf || '',
      d.snapshot_path || ''
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `forestguard_detections_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="page-container">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Detection History & Event Logs</h1>
          <p className="page-subtitle">Persistent SQLite audit trail with snapshot snapshots and confidence logs</p>
        </div>
        <button onClick={handleExportCSV} className="sound-toggle-btn active" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Download size={16} />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Filter Bar */}
      <div className="panel" style={{ padding: '14px 20px', marginBottom: '20px', display: 'flex', gap: '16px', alignItems: 'center' }}>
        <Filter size={18} color="var(--text-muted)" />
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', fontSize: '0.85rem' }}>
          <span>State:</span>
          <select
            value={stateFilter}
            onChange={(e) => setStateFilter(e.target.value)}
            style={{ padding: '6px 10px', borderRadius: '4px', backgroundColor: 'var(--bg-subtle)', border: '1px solid var(--border-color)', color: '#fff' }}
          >
            <option value="">All States</option>
            <option value="FIRE_CONFIRMED">FIRE_CONFIRMED</option>
            <option value="SMOKE_CONFIRMED">SMOKE_CONFIRMED</option>
            <option value="POSSIBLE_FIRE">POSSIBLE_FIRE</option>
            <option value="UNCERTAIN">UNCERTAIN</option>
          </select>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', fontSize: '0.85rem' }}>
          <span>Source:</span>
          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            style={{ padding: '6px 10px', borderRadius: '4px', backgroundColor: 'var(--bg-subtle)', border: '1px solid var(--border-color)', color: '#fff' }}
          >
            <option value="">All Sources</option>
            <option value="phone_camera">phone_camera</option>
            <option value="image_upload">image_upload</option>
            <option value="webcam">webcam</option>
            <option value="ip">ip</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="panel" style={{ padding: '0', overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Timestamp</th>
              <th>State</th>
              <th>Top Threat</th>
              <th>Confidence</th>
              <th>Source</th>
              <th>Snapshot</th>
            </tr>
          </thead>
          <tbody>
            {detections.length === 0 ? (
              <tr>
                <td colSpan="7" style={{ textAlign: 'center', padding: '30px', color: 'var(--text-dim)' }}>
                  {loading ? 'Loading logs...' : 'No detection events found matching filters.'}
                </td>
              </tr>
            ) : (
              detections.map((row) => (
                <tr key={row.id}>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>#{row.id}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                    {new Date(row.ts).toLocaleString()}
                  </td>
                  <td>
                    <span
                      style={{
                        padding: '3px 8px',
                        borderRadius: '4px',
                        fontSize: '0.78rem',
                        fontWeight: '600',
                        backgroundColor: row.state.includes('FIRE') ? 'var(--color-red-glow)' : 'var(--color-amber-glow)',
                        color: row.state.includes('FIRE') ? 'var(--color-red)' : 'var(--color-amber)'
                      }}
                    >
                      {row.state}
                    </span>
                  </td>
                  <td style={{ fontWeight: '500' }}>{row.top_class?.toUpperCase()}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {row.top_conf ? `${(row.top_conf * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>{row.source}</td>
                  <td>
                    {row.snapshot_path ? (
                      <button
                        onClick={() => setModalSnapshot(row.snapshot_path)}
                        style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--color-green)', fontSize: '0.82rem' }}
                      >
                        <Eye size={14} /> View
                      </button>
                    ) : (
                      <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>No Snapshot</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Modal for Snapshot */}
      {modalSnapshot && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100
          }}
          onClick={() => setModalSnapshot(null)}
        >
          <div style={{ position: 'relative', maxWidth: '800px', width: '90%', background: 'var(--bg-panel)', padding: '16px', borderRadius: '10px' }} onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setModalSnapshot(null)}
              style={{ position: 'absolute', top: '12px', right: '12px', color: '#fff' }}
            >
              <X size={20} />
            </button>
            <h4 style={{ marginBottom: '12px' }}>Event Evidence Snapshot</h4>
            <img src={modalSnapshot} alt="Evidence" style={{ width: '100%', maxHeight: '500px', objectFit: 'contain', borderRadius: '6px' }} />
          </div>
        </div>
      )}
    </div>
  );
}
