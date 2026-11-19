try:
    import pytest
except ImportError:
    pytest = None
from backend.services.decision_engine import DecisionEngine, DecisionState

class MockClock:
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time

    def tick(self, seconds: float = 0.2):
        self.current_time += seconds
        return self.current_time

    def get_time(self) -> float:
        return self.current_time

def test_agreement_confirms_fire():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    # 3 frames with YOLO fire 0.75 + CNN fire 0.85
    for _ in range(3):
        clock.tick()
        res = engine.update(
            cnn_result={"probs": {"fire": 0.85, "no_fire": 0.15}},
            yolo_result=[{"cls": "fire", "conf": 0.75, "xyxy": [10, 10, 50, 50]}]
        )

    assert res["status"] == DecisionState.FIRE_CONFIRMED
    assert res["should_alert"] is True
    assert res["alert_type"] == "fire_detected"

def test_disagreement_leads_to_uncertain():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    # YOLO says fire with 0.80, but CNN strongly says no fire (fire prob 0.10)
    for _ in range(4):
        clock.tick()
        res = engine.update(
            cnn_result={"probs": {"fire": 0.10, "no_fire": 0.90}},
            yolo_result=[{"cls": "fire", "conf": 0.80, "xyxy": [10, 10, 50, 50]}]
        )

    assert res["status"] == DecisionState.UNCERTAIN
    assert res["status"] != DecisionState.FIRE_CONFIRMED
    assert res["should_alert"] is False

def test_single_noisy_frame_does_not_confirm():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    # 5 frames of no fire
    for _ in range(5):
        clock.tick()
        engine.update(
            cnn_result={"probs": {"fire": 0.05, "no_fire": 0.95}},
            yolo_result=[]
        )

    # 1 single frame of fire
    clock.tick()
    res = engine.update(
        cnn_result={"probs": {"fire": 0.95, "no_fire": 0.05}},
        yolo_result=[{"cls": "fire", "conf": 0.85, "xyxy": [10, 10, 50, 50]}]
    )

    # Not confirmed yet (needs 3 frames)
    assert res["status"] != DecisionState.FIRE_CONFIRMED
    assert res["should_alert"] is False

def test_cnn_only_strong_produces_possible_fire():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    # 4 frames of CNN fire >= 0.90 with NO YOLO box
    for _ in range(4):
        clock.tick()
        res = engine.update(
            cnn_result={"probs": {"fire": 0.92, "no_fire": 0.08}},
            yolo_result=[]
        )

    assert res["status"] == DecisionState.POSSIBLE_FIRE
    assert res["should_alert"] is True
    assert res["alert_type"] == "possible_fire"

def test_smoke_confirmation():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    for _ in range(3):
        clock.tick()
        res = engine.update(
            cnn_result={"probs": {"fire": 0.10, "no_fire": 0.90}},
            yolo_result=[{"cls": "smoke", "conf": 0.65, "xyxy": [5, 5, 40, 40]}]
        )

    assert res["status"] == DecisionState.SMOKE_CONFIRMED
    assert res["should_alert"] is True
    assert res["alert_type"] == "smoke_detected"

def test_one_alert_per_continuous_event():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    alerts_fired = 0
    # Stream 10 consecutive fire frames
    for _ in range(10):
        clock.tick()
        res = engine.update(
            cnn_result={"probs": {"fire": 0.85, "no_fire": 0.15}},
            yolo_result=[{"cls": "fire", "conf": 0.75, "xyxy": [10, 10, 50, 50]}]
        )
        if res["should_alert"]:
            alerts_fired += 1

    # Exactly ONE alert fired during continuous event
    assert alerts_fired == 1

def test_clear_and_cooldown_behavior():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time, clear_seconds=10.0, cooldown_seconds=30.0)

    # 1. Fire confirmed -> alert fires
    for _ in range(3):
        clock.tick(0.2)
        res = engine.update(
            cnn_result={"probs": {"fire": 0.85, "no_fire": 0.15}},
            yolo_result=[{"cls": "fire", "conf": 0.75, "xyxy": [10, 10, 50, 50]}]
        )
    assert res["should_alert"] is True

    # 2. Fire stops: stream no_fire frames until window clears and clear_seconds (10s) elapse
    for _ in range(12):
        clock.tick(2.0)
        engine.update(
            cnn_result={"probs": {"fire": 0.05, "no_fire": 0.95}},
            yolo_result=[]
        )
    assert engine.active_event_state is None

    # 3. Fire reappears during cooldown (only 10 seconds into 30s cooldown)
    clock.tick(10.0)
    for _ in range(3):
        clock.tick(0.2)
        res = engine.update(
            cnn_result={"probs": {"fire": 0.85, "no_fire": 0.15}},
            yolo_result=[{"cls": "fire", "conf": 0.75, "xyxy": [10, 10, 50, 50]}]
        )
    # Should be suppressed by cooldown
    assert res["should_alert"] is False

    # Clear window of in-cooldown fire frames
    for _ in range(6):
        clock.tick(0.2)
        engine.update(
            cnn_result={"probs": {"fire": 0.05, "no_fire": 0.95}},
            yolo_result=[]
        )

    # 4. Wait out remaining cooldown (> 30s total) and trigger fire again
    clock.tick(30.0)
    for _ in range(6):
        clock.tick(1.0)
        engine.update(
            cnn_result={"probs": {"fire": 0.05, "no_fire": 0.95}},
            yolo_result=[]
        )
    # Re-trigger fire
    for _ in range(3):
        clock.tick(0.2)
        res = engine.update(
            cnn_result={"probs": {"fire": 0.85, "no_fire": 0.15}},
            yolo_result=[{"cls": "fire", "conf": 0.75, "xyxy": [10, 10, 50, 50]}]
        )
    assert res["should_alert"] is True

def test_escalation_smoke_to_fire_alerts_once():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    # 1. Smoke confirmed -> 1 alert
    for _ in range(3):
        clock.tick(0.2)
        res = engine.update(
            cnn_result={"probs": {"fire": 0.10, "no_fire": 0.90}},
            yolo_result=[{"cls": "smoke", "conf": 0.60, "xyxy": [10, 10, 50, 50]}]
        )
    assert res["status"] == DecisionState.SMOKE_CONFIRMED
    assert res["should_alert"] is True
    assert res["alert_type"] == "smoke_detected"

    # 2. Fire now breaks out -> escalation fires second alert
    for _ in range(3):
        clock.tick(0.2)
        res = engine.update(
            cnn_result={"probs": {"fire": 0.90, "no_fire": 0.10}},
            yolo_result=[{"cls": "fire", "conf": 0.80, "xyxy": [10, 10, 50, 50]}]
        )
    assert res["status"] == DecisionState.FIRE_CONFIRMED
    assert res["should_alert"] is True
    assert res["alert_type"] == "fire_detected"

    # 3. Fire continues -> no duplicate alert
    clock.tick(0.2)
    res = engine.update(
        cnn_result={"probs": {"fire": 0.90, "no_fire": 0.10}},
        yolo_result=[{"cls": "fire", "conf": 0.80, "xyxy": [10, 10, 50, 50]}]
    )
    assert res["should_alert"] is False

def test_model_unavailable_fallback():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)

    # CNN unavailable -> YOLO fire confirmed with "CNN unavailable"
    for _ in range(3):
        clock.tick(0.2)
        res = engine.update(
            cnn_result=None,
            yolo_result=[{"cls": "fire", "conf": 0.70, "xyxy": [10, 10, 50, 50]}],
            cnn_available=False,
            yolo_available=True
        )
    assert res["status"] == DecisionState.FIRE_CONFIRMED
    assert "CNN unavailable" in res["reason"]

    engine.reset()

    # YOLO unavailable -> CNN strong fire yields POSSIBLE_FIRE max
    for _ in range(4):
        clock.tick(0.2)
        res = engine.update(
            cnn_result={"probs": {"fire": 0.95, "no_fire": 0.05}},
            yolo_result=None,
            cnn_available=True,
            yolo_available=False
        )
    assert res["status"] == DecisionState.POSSIBLE_FIRE
    assert "YOLO unavailable" in res["reason"]


def test_both_models_unavailable_produces_model_unavailable():
    clock = MockClock()
    engine = DecisionEngine(time_fn=clock.get_time)
    res = engine.update(
        cnn_result=None,
        yolo_result=None,
        cnn_available=False,
        yolo_available=False
    )
    assert res["status"] == DecisionState.MODEL_UNAVAILABLE
    assert "models unavailable" in res["reason"].lower()
    assert res["should_alert"] is False
