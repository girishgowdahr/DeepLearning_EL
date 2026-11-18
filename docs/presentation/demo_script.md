# ForestGuard AI — 5-Minute Live Academic Demo Script

Use this script for a flawless, professional 5-minute presentation and live system demonstration.

---

## 🎙️ Demo Script Timeline (5 Minutes Total)

### Minute 0:00 – 1:00: Problem & Motivation (The "Why")
- **Say**: *"Good morning respected evaluators. Wildfires cause devastating damage within minutes. Traditional smoke sensors are too slow and expensive to deploy over square kilometers of forest. Single computer vision models often trigger false alarms on clouds, dust, and sunsets."*
- **Show**: Slide 1–3 (Architecture overview & problem statement).
- **Key point**: *"ForestGuard AI solves this by fusing two different models: a MobileNetV2 CNN classifier that evaluates the full frame, and a YOLO11 Nano detector that precisely localizes fire and smoke bounding boxes. Neither model has sole authority; a rule-based decision engine coordinates them."*

### Minute 1:00 – 2:00: System Architecture & Dashboard Tour
- **Action**: Open browser to `http://127.0.0.1:8000/`.
- **Say**: *"Here is our central surveillance console built in React 18 and powered by a high-performance Python FastAPI backend."*
- **Click**:
  1. **Overview Page (`/`)**: Show system telemetry, real-time counters, and model statuses.
  2. **Model Specifications (`/models`)**: Point out transparent disclosures, MobileNetV2 ONNX execution provider, and YOLO nano architecture.
  3. **Analytics Page (`/analytics`)**: Show verified test-set accuracy, precision, recall, and loss curves.

### Minute 2:00 – 3:30: Live Pipeline Demonstration (Webcam or Video)
- **Action**: Click **Live Camera (`/live`)** or **Phone Camera (`/camera`)**.
- **Demonstration**:
  - Start the live stream (laptop webcam index 0 or test clip from `backend/data/test_media/test_fire_clip.mp4`).
  - Point the camera at a fire benchmark frame (or hold up phone showing benchmark flame image).
  - Observe the console:
    1. Bounding boxes highlight the flame.
    2. Decision engine transitions from `NO_FIRE` $\rightarrow$ `UNCERTAIN` $\rightarrow$ `FIRE_CONFIRMED`.
    3. Emergency audio siren sounds via Web Audio API.
    4. Incident log is permanently recorded in SQLite database.
- **Say**: *"Notice our temporal confirmation: a single noisy frame does not trigger an alarm. The system requires 3 out of 6 consecutive frames to confirm, eliminating transient glitches."*

### Minute 3:30 – 4:15: Hard Negatives & Fallback Resilience
- **Action**: In **Image Detection (`/image`)**, upload `backend/data/test_media/hard_negative_sunset.jpg`.
- **Show**: The golden hour sunset does NOT trigger a fire alert. Status stays `NO_FIRE`.
- **Say**: *"Here is a classic failure mode for wildfire vision: sunsets. Because our CNN classifier was trained with strict negative canopy examples and our decision engine requires mutual agreement, it rejects the false alarm."*

### Minute 4:15 – 5:00: Conclusion & Q&A Transition
- **Say**: *"In summary, ForestGuard AI achieves sub-25ms roundtrip latency on standard CPU hardware without requiring expensive GPUs in the field. Thank you, and I look forward to your questions."*

---

## 🛠️ Emergency Fallback Checklist
- If Wi-Fi fails: The entire application runs on `localhost:8000` with 0 external internet dependency!
- If phone camera fails: Use laptop webcam (source index 0) or the pre-assembled clips in `backend/data/test_media/test_fire_clip.mp4`.
