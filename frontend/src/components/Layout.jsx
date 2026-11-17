import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useEvents } from '../hooks/useEvents';

export function Layout() {
  const { isConnected, activeAlert, dismissAlert, soundEnabled, toggleSound } = useEvents();

  return (
    <div className="app-layout">
      <Sidebar />
      <div className="app-main">
        <Header
          isWsConnected={isConnected}
          activeAlert={activeAlert}
          dismissAlert={dismissAlert}
          soundEnabled={soundEnabled}
          toggleSound={toggleSound}
        />
        <main style={{ flex: 1 }}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
