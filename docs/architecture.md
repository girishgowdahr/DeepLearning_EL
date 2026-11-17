# ForestGuard AI — System Architecture Specification

## 1. High-Level System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │                   EDGE CAPTURE TIER                    │
                    └────────────────────────────────────────────────────────┘
                                               │
     ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
     │                                         │                                         │
┌─────────────┐                        ┌─────────────┐                           ┌─────────────┐
│   Android   │                        │  Android IP │                           │   Laptop    │
│Phone Camera │                        │  Webcam App │                           │ Built-in HD │
│(Chrome/PWA) │                        │(MJPEG/RTSP) │                           │   Webcam    │
└──────┬──────┘                        └──────┬──────┘                           └──────┬──────┘
       │                                      │                                         │
       │ getUserMedia                         │ HTTP / RTSP                             │ OpenCV
       │ Canvas 640x480                       │ :8080/video                             │ VideoCapture
       │ JPEG Blob (0.6)                      │ LAN Wi-Fi                               │ Device Index 0
       ▼                                      ▼                                         ▼
 ┌─────────────┐                       ┌─────────────┐                                  │
 │ Cloudflare  │                       │   Camera    │                                  │
 │Tunnel/HTTPS │                       │   Worker    │                                  │
 └──────┬──────┘                       │  (Thread)   │                                  │
        │                                     │                                         │
        │ WebSocket (/ws/stream)              │ Latest Frame Buffer                     │
        └─────────────────────────────────────┼─────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌────────────────────────────────────────────────────────┐
                    │               FASTAPI INFERENCE PIPELINE               │
                    │               (Single Worker Thread / CPU)             │
                    └────────────────────────────────────────────────────────┘
                                              │
                         ┌────────────────────┴───────────────────┐
                         │                                        │
                         ▼                                        ▼
               ┌───────────────────┐                    ┌───────────────────┐
               │  Ultralytics YOLO │                    │   MobileNetV2     │
               │   Nano Detector   │                    │  CNN Classifier   │
               │  (416x416, CPU)   │                    │  (224x224, ONNX)  │
               └─────────┬─────────┘                    └─────────┬─────────┘
                         │                                        │
                         │ Bounding Boxes                         │ Softmax Probs
                         │ {fire, smoke}                          │ {fire, no_fire}
                         │                                        │
                         └────────────────────┬───────────────────┘
                                              │
                                              ▼
                    ┌────────────────────────────────────────────────────────┐
                    │                 DECISION FUSION ENGINE                 │
                    │      (Rule-based evaluation + M=6 Sliding Window)      │
                    └────────────────────────────────────────────────────────┘
                                              │
                         ┌────────────────────┼───────────────────┐
                         │                    │                   │
                         ▼                    ▼                   ▼
                ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
                │ SQLite Database  │ │    OpenCV 4      │ │  Alert Manager   │
                │ (forestguard.db) │ │   Annotator      │ │ (Event Lifecycle │
                │ Detections/Logs  │ │ (Status Banner)  │ │  & Cooldown)     │
                └──────────────────┘ └────────┬─────────┘ └────────┬─────────┘
                                              │                    │
                                              ▼                    ▼
                                       Annotated MJPEG      WebSocket Events
                                        Stream / Frames      (/ws/events)
                                              │                    │
                                              └──────────┬─────────┘
                                                         │
                                                         ▼
                    ┌────────────────────────────────────────────────────────┐
                    │              REACT 18 CONTROL-ROOM DASHBOARD           │
                    │            (10 Pages + Web Audio Siren Synth)          │
                    └────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Edge Capture Tier
- **Phone Browser (`/camera`)**: Primary capture mechanism. Accesses phone rear camera via HTML5 `navigator.mediaDevices.getUserMedia`. Frames are captured to an offscreen canvas at a maximum dimension of 640px, compressed to JPEG blobs, and transmitted over WebSocket `/ws/stream`.
- **Backpressure Mechanism**: Client transmits the next frame strictly upon receiving acknowledgement (annotated frame + detection JSON) from the server, preventing queue build-up and keeping CPU inference real-time.
- **Alternative Ingestion**: Background `CameraWorker` handles MJPEG streams (`http://<phone-ip>:8080/video`) from third-party IP camera apps or local laptop USB webcams (`/dev/video0`).

### 2.2 Deep Learning Inference Tier
- **Ultralytics YOLO (Nano)**:
  - Localizes instances of fire and smoke.
  - Inputs letterboxed to 416×416 for fast CPU execution.
  - Emits normalized bounding box coordinates and confidences.
- **MobileNetV2 CNN (Transfer Learning)**:
  - Binary classifier answering whether fire is visually present across the whole frame.
  - Inputs preprocessed to 224×224 with ImageNet normalization constants.
  - Executed via ONNX Runtime with single-thread/4-thread CPU optimizations.

### 2.3 Transparent Decision Engine
- **No Averaging**: Confidences are never blended or averaged across dissimilar tasks.
- **Rules Matrix**:
  - `FIRE_CONFIRMED`: YOLO fire ≥ 0.40 and CNN fire ≥ 0.60 across ≥ 3 of 6 frames.
  - `SMOKE_CONFIRMED`: YOLO smoke ≥ 0.45 across ≥ 3 of 6 frames.
  - `POSSIBLE_FIRE`: High CNN confidence without YOLO localization.
  - `UNCERTAIN`: Conflicting model predictions.
  - `NO_FIRE`: Default clear state.
- **Event Lifecycle**:
  - Active event suppression ensures one alert and one siren per fire outbreak.
  - 10-second clear-time requirement before an event closes.
  - 30-second cooldown period prevents re-alerting on intermittent smoke.

### 2.4 Control Room Dashboard Tier
- **Frontend**: React 18, Vite, Recharts, Lucide icons, hand-written dark control-room theme.
- **Audio Siren**: Synthesized client-side via HTML5 Web Audio API oscillators (880 Hz / 660 Hz alternating siren; no audio assets to download).
- **Single-Port Serving**: In production/demo mode, FastAPI statically serves the compiled frontend assets from `frontend/dist/` on port 8000.
