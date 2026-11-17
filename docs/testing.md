# ForestGuard AI — Testing, Validation & Verification Report

This document records the systematic verification of the ForestGuard AI prototype across unit tests, API integration tests, and the 12-scenario operational matrix.

---

## 1. Automated Unit & Integration Tests

### 1.1 Decision Engine Unit Test Suite (`backend/tests/test_decision_engine.py`)
All 9 core behavioral specifications of the fusion engine are verified using an injectable deterministic clock:

| Test Case | Scenario Verified | Expected Outcome | Result |
|---|---|---|---|
| `test_agreement_confirms_fire` | YOLO Fire (0.75) + CNN Fire (0.85) in 3 frames | `status=FIRE_CONFIRMED`, `should_alert=True` | ✅ PASSED |
| `test_disagreement_leads_to_uncertain` | YOLO Fire (0.80) vs CNN Fire (0.10) in 4 frames | `status=UNCERTAIN`, `should_alert=False` | ✅ PASSED |
| `test_single_noisy_frame_does_not_confirm` | 5 normal frames followed by 1 noisy fire frame | No confirmation, `should_alert=False` | ✅ PASSED |
| `test_cnn_only_strong_produces_possible_fire` | CNN Fire ≥ 0.90 with no YOLO bounding box in 4 frames | `status=POSSIBLE_FIRE`, `alert_type=possible_fire` | ✅ PASSED |
| `test_smoke_confirmation` | YOLO Smoke (0.65) in 3 frames | `status=SMOKE_CONFIRMED`, `should_alert=True` | ✅ PASSED |
| `test_one_alert_per_continuous_event` | 10 continuous fire frames | Exactly 1 alert generated | ✅ PASSED |
| `test_clear_and_cooldown_behavior` | Event closes after 10s clear; suppressed for 30s | Re-alerts only after cooldown elapses | ✅ PASSED |
| `test_escalation_smoke_to_fire_alerts_once` | Smoke event escalates to confirmed fire | 1 smoke alert followed by 1 fire alert | ✅ PASSED |
| `test_model_unavailable_fallback` | One model weight file removed | Graceful degraded state; app remains stable | ✅ PASSED |

---

## 2. Twelve-Scenario Operational Matrix

| # | Scenario | Input Type | Observed System Behavior | Status |
|---|---|---|---|---|
| **1** | Fire Image | Still photo with forest flames | YOLO places red box around fire front; CNN outputs high fire probability; Decision engine confirms FIRE. | ✅ VERIFIED |
| **2** | Normal Forest Image | Still photo of green canopy / trees | YOLO outputs 0 boxes; CNN outputs >90% no_fire; Status banner remains green (`NO_FIRE`). | ✅ VERIFIED |
| **3** | Smoke Image | Image of rising smoke column | YOLO places orange box around plume; CNN indicates no flames; Confirms SMOKE. | ✅ VERIFIED |
| **4** | Video with Fire | MP4 forest fire surveillance clip | Sampled at ~2 FPS; timeline shows progression to confirmed fire; flagged frames displayed in preview gallery. | ✅ VERIFIED |
| **5** | Video with Smoke | MP4 wildfire smoke plume clip | Accurately identifies smoke onset; flags frames in timeline. | ✅ VERIFIED |
| **6** | Phone Live Camera | Android Chrome via HTTPS | Real-time backpressure stream over `/ws/stream`; annotated video appears smoothly on laptop dashboard. | ✅ VERIFIED |
| **7** | Permission Denied | Phone camera permission blocked | Displays prominent message: *"Camera permission was denied. Please grant permission in your browser site settings."* | ✅ VERIFIED |
| **8** | Camera Disconnect | Abruptly closing phone browser | Server detects >5s without frames; marks camera idle and raises a single `camera_disconnected` alert. | ✅ VERIFIED |
| **9** | CNN Unavailable | Weight file moved or absent | Dashboard status chip turns gray (`CNN Unavailable`); Decision Engine falls back to YOLO-only rules with logged disclaimer. | ✅ VERIFIED |
| **10** | YOLO Unavailable | YOLO weight file moved or absent | Dashboard status chip turns gray (`YOLO Unavailable`); Decision Engine limits max state to `POSSIBLE_FIRE` with unlocalized notice. | ✅ VERIFIED |
| **11** | Model Disagreement | Flame-colored vehicle or sunset | YOLO proposes box but CNN rejects; State marked `UNCERTAIN`; siren is suppressed. | ✅ VERIFIED |
| **12** | Alert Suppression | Continuous 20s flame stream | Exactly 1 initial siren and 1 dashboard alert logged. No repetitive alerts occur until cooldown. | ✅ VERIFIED |

---

## 3. Real Performance & Hardware Telemetry (Windows Laptop, 8 GB RAM, CPU)

- **Peak Memory Usage**: 1.25 GB (Well within the 2.0 GB limit).
- **Average CPU Inference Latency**:
  - MobileNetV2 (ONNX Runtime, 224×224): **18–26 ms**
  - YOLO Nano (Ultralytics CPU, 416×416): **140–180 ms**
  - End-to-End Pipeline Latency: **~195 ms**
- **Observed Stream Frame Rate**: **3.5–5.0 FPS** (Fluid and responsive for surveillance).
