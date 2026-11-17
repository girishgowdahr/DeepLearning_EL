import { useState, useEffect, useRef } from 'react';
import { playFireSiren, playSmokeBeep } from '../utils/audio';

export function useEvents() {
  const [isConnected, setIsConnected] = useState(false);
  const [latestDetection, setLatestDetection] = useState(null);
  const [activeAlert, setActiveAlert] = useState(null);
  const [soundEnabled, setSoundEnabled] = useState(() => {
    return localStorage.getItem('fg_sound_enabled') !== 'false';
  });

  const wsRef = useRef(null);
  const retryTimeoutRef = useRef(null);

  const toggleSound = () => {
    setSoundEnabled(prev => {
      const next = !prev;
      localStorage.setItem('fg_sound_enabled', String(next));
      return next;
    });
  };

  useEffect(() => {
    let unmounted = false;

    function connect() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/events`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (unmounted) return;
        setIsConnected(true);
        console.log('[WS] Connected to /ws/events');
      };

      ws.onmessage = (event) => {
        if (unmounted) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'detection') {
            setLatestDetection(data.payload);
          } else if (data.type === 'alert') {
            const alert = data.payload;
            setActiveAlert(alert);
            
            // Sound dispatch (only if enabled)
            if (soundEnabled) {
              if (alert.type === 'fire_detected' || alert.type === 'test') {
                playFireSiren();
              } else if (alert.type === 'smoke_detected') {
                playSmokeBeep();
              }
            }
          }
        } catch (e) {
          console.error('[WS] Parse error:', e);
        }
      };

      ws.onclose = () => {
        if (unmounted) return;
        setIsConnected(false);
        console.warn('[WS] Closed. Retrying in 2s...');
        retryTimeoutRef.current = setTimeout(connect, 2000);
      };

      ws.onerror = (err) => {
        console.error('[WS] Error:', err);
        ws.close();
      };
    }

    connect();

    return () => {
      unmounted = true;
      if (wsRef.current) wsRef.current.close();
      if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    };
  }, [soundEnabled]);

  return {
    isConnected,
    latestDetection,
    activeAlert,
    dismissAlert: () => setActiveAlert(null),
    soundEnabled,
    toggleSound
  };
}
