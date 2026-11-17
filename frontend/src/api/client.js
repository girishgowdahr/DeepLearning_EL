// ForestGuard AI API Client

const API_BASE = '/api';

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let errorMsg = `HTTP Error ${res.status}`;
    try {
      const err = await res.json();
      errorMsg = err.detail || errorMsg;
    } catch (_) {}
    throw new Error(errorMsg);
  }
  return res.json();
}

export const api = {
  getHealth: () => fetchJSON(`${API_BASE}/health`),
  getModelsStatus: () => fetchJSON(`${API_BASE}/models/status`),
  getModelsInfo: () => fetchJSON(`${API_BASE}/models/info`),
  reloadModels: () => fetchJSON(`${API_BASE}/models/reload`, { method: 'POST' }),
  
  getCameraStatus: () => fetchJSON(`${API_BASE}/camera/status`),
  startCamera: (source, url = null, index = 0) =>
    fetchJSON(`${API_BASE}/camera/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, url, index })
    }),
  stopCamera: () => fetchJSON(`${API_BASE}/camera/stop`, { method: 'POST' }),

  getDetections: (limit = 50, offset = 0, source = null, state = null) => {
    const params = new URLSearchParams({ limit, offset });
    if (source) params.append('source', source);
    if (state) params.append('state', state);
    return fetchJSON(`${API_BASE}/detections?${params.toString()}`);
  },

  getAlerts: (limit = 50, acknowledged = null) => {
    const params = new URLSearchParams({ limit });
    if (acknowledged !== null) params.append('acknowledged', acknowledged);
    return fetchJSON(`${API_BASE}/alerts?${params.toString()}`);
  },
  ackAlert: (id) => fetchJSON(`${API_BASE}/alerts/${id}/ack`, { method: 'POST' }),
  triggerTestAlert: () => fetchJSON(`${API_BASE}/alerts/test`, { method: 'POST' }),

  getAnalytics: () => fetchJSON(`${API_BASE}/analytics`),
  getSettings: () => fetchJSON(`${API_BASE}/settings`),
  updateSettings: (settings) =>
    fetchJSON(`${API_BASE}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    }),

  getNetworkInfo: () => fetchJSON(`${API_BASE}/system/network`),

  detectImage: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJSON(`${API_BASE}/detect/image`, {
      method: 'POST',
      body: formData
    });
  },

  detectVideo: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJSON(`${API_BASE}/detect/video`, {
      method: 'POST',
      body: formData
    });
  },
  getVideoJobStatus: (jobId) => fetchJSON(`${API_BASE}/detect/video/${jobId}`)
};
