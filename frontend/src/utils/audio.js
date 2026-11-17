// ForestGuard AI — Web Audio API Siren & Alert Sound System
// Pure oscillator synthesis — zero external audio files required!

let audioCtx = null;
let isUnlocked = false;

export function initAudio() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) {
      audioCtx = new AudioContext();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  isUnlocked = true;
  return isUnlocked;
}

export function isAudioUnlocked() {
  return isUnlocked && audioCtx && audioCtx.state === 'running';
}

export function playFireSiren() {
  if (!initAudio()) return;

  const now = audioCtx.currentTime;
  const duration = 2.8; // seconds

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();

  osc.type = 'sawtooth';
  
  // Alternating between 880Hz and 660Hz every 0.35s
  const cycleTime = 0.35;
  for (let t = 0; t < duration; t += cycleTime) {
    const freq = (Math.floor(t / cycleTime) % 2 === 0) ? 880 : 660;
    osc.frequency.setValueAtTime(freq, now + t);
  }

  // Volume envelope
  gain.gain.setValueAtTime(0.01, now);
  gain.gain.exponentialRampToValueAtTime(0.3, now + 0.1);
  gain.gain.setValueAtTime(0.3, now + duration - 0.2);
  gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

  osc.connect(gain);
  gain.connect(audioCtx.destination);

  osc.start(now);
  osc.stop(now + duration);
}

export function playSmokeBeep() {
  if (!initAudio()) return;

  const now = audioCtx.currentTime;
  
  // Double beep (587 Hz)
  [0, 0.25].forEach((offset) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587.33, now + offset);
    
    gain.gain.setValueAtTime(0.01, now + offset);
    gain.gain.exponentialRampToValueAtTime(0.2, now + offset + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.001, now + offset + 0.18);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.start(now + offset);
    osc.stop(now + offset + 0.18);
  });
}
