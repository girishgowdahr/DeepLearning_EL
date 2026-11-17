import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Overview } from './pages/Overview';
import { LiveCamera } from './pages/LiveCamera';
import { ImageDetection } from './pages/ImageDetection';
import { VideoDetection } from './pages/VideoDetection';
import { CnnClassifier } from './pages/CnnClassifier';
import { YoloDetection } from './pages/YoloDetection';
import { DetectionHistory } from './pages/DetectionHistory';
import { Analytics } from './pages/Analytics';
import { ModelInfo } from './pages/ModelInfo';
import { SettingsPage } from './pages/SettingsPage';
import { PhoneCamera } from './pages/PhoneCamera';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Main Dashboard with Sidebar & Control Room Theme */}
        <Route path="/" element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="live" element={<LiveCamera />} />
          <Route path="image" element={<ImageDetection />} />
          <Route path="video" element={<VideoDetection />} />
          <Route path="cnn" element={<CnnClassifier />} />
          <Route path="yolo" element={<YoloDetection />} />
          <Route path="history" element={<DetectionHistory />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="models" element={<ModelInfo />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>

        {/* Mobile Edge Sensor Client (Phone Camera Streamer) */}
        <Route path="/camera" element={<PhoneCamera />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
