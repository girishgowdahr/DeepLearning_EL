import time
from collections import deque
from typing import Callable, Optional

class DecisionState:
    NO_FIRE = "NO_FIRE"
    UNCERTAIN = "UNCERTAIN"
    POSSIBLE_FIRE = "POSSIBLE_FIRE"
    SMOKE_CONFIRMED = "SMOKE_CONFIRMED"
    FIRE_CONFIRMED = "FIRE_CONFIRMED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"

class DecisionEngine:
    """
    Transparent, rule-based decision engine fusing CNN classification and YOLO detection.
    Never averages their confidences. Supports sliding window confirmation,
    event escalation, clear-time, and cooldown suppression.
    """
    def __init__(
        self,
        yolo_conf_fire: float = 0.40,
        yolo_conf_smoke: float = 0.45,
        cnn_threshold: float = 0.60,
        cnn_strong_threshold: float = 0.90,
        window_size: int = 6,
        confirm_frames: int = 3,
        confirm_possible_frames: int = 4,
        clear_seconds: float = 10.0,
        cooldown_seconds: float = 30.0,
        time_fn: Optional[Callable[[], float]] = None
    ):
        self.yolo_conf_fire = yolo_conf_fire
        self.yolo_conf_smoke = yolo_conf_smoke
        self.cnn_threshold = cnn_threshold
        self.cnn_strong_threshold = cnn_strong_threshold
        self.window_size = window_size
        self.confirm_frames = confirm_frames
        self.confirm_possible_frames = confirm_possible_frames
        self.clear_seconds = clear_seconds
        self.cooldown_seconds = cooldown_seconds
        self.time_fn = time_fn or time.time

        self.window: deque[tuple[str, str, float]] = deque(maxlen=self.window_size)
        
        # Event tracking
        self.active_event_state: Optional[str] = None
        self.active_event_start: Optional[float] = None
        self.last_confirmed_ts: Optional[float] = None
        self.last_event_closed_ts: Optional[float] = None
        self.escalated: bool = False

    def evaluate_frame(
        self,
        cnn_result: Optional[dict],
        yolo_result: Optional[list[dict]],
        cnn_available: bool = True,
        yolo_available: bool = True
    ) -> tuple[str, str]:
        """
        Evaluates a single frame and returns (candidate_state, reason).
        Candidate states: 'FIRE', 'SMOKE', 'POSSIBLE_FIRE', 'UNCERTAIN', 'NO_FIRE'
        """
        # Extract YOLO confidences
        yolo_fire = 0.0
        yolo_smoke = 0.0
        if yolo_available and yolo_result:
            for det in yolo_result:
                cls_name = det.get("cls", "").lower()
                conf = float(det.get("conf", 0.0))
                if cls_name == "fire" and conf > yolo_fire:
                    yolo_fire = conf
                elif cls_name == "smoke" and conf > yolo_smoke:
                    yolo_smoke = conf

        # Extract CNN fire probability
        cnn_fire = 0.0
        if cnn_available and cnn_result:
            probs = cnn_result.get("probs", {})
            if "fire" in probs:
                cnn_fire = float(probs["fire"])
            else:
                label = cnn_result.get("label", "").lower()
                conf = float(cnn_result.get("confidence", 0.0))
                cnn_fire = conf if label == "fire" else (1.0 - conf)

        # Handle Model Degraded Modes
        if not cnn_available and not yolo_available:
            return DecisionState.MODEL_UNAVAILABLE, "Both CNN and YOLO models unavailable"

        if not cnn_available:
            if yolo_fire >= self.yolo_conf_fire:
                return "FIRE", f"YOLO fire {yolo_fire:.2f} (CNN unavailable)"
            if yolo_smoke >= self.yolo_conf_smoke:
                return "SMOKE", f"YOLO smoke {yolo_smoke:.2f} (CNN unavailable)"
            return "NO_FIRE", "No detections (CNN unavailable)"

        if not yolo_available:
            if cnn_fire >= self.cnn_strong_threshold:
                return "POSSIBLE_FIRE", f"CNN fire {cnn_fire:.2f} (YOLO unavailable, unlocalized)"
            return "NO_FIRE", "No fire detected (YOLO unavailable)"

        # Normal Fusion Rules
        # Rule 1: Models agree on Fire
        if yolo_fire >= self.yolo_conf_fire and cnn_fire >= self.cnn_threshold:
            return "FIRE", f"Models agree: YOLO fire {yolo_fire:.2f} + CNN fire {cnn_fire:.2f}"

        # Rule 2: Strong YOLO fire but CNN disagrees -> UNCERTAIN
        if yolo_fire >= 0.55 and cnn_fire < self.cnn_threshold:
            return "UNCERTAIN", f"Models disagree: YOLO fire {yolo_fire:.2f} vs CNN fire {cnn_fire:.2f}"

        # Rule 3: YOLO fire [0.40, 0.55) and CNN disagrees -> UNCERTAIN
        if 0.40 <= yolo_fire < 0.55 and cnn_fire < self.cnn_threshold:
            return "UNCERTAIN", f"Borderline disagreement: YOLO fire {yolo_fire:.2f} vs CNN fire {cnn_fire:.2f}"

        # Rule 4: CNN strongly confident of fire, but YOLO has no box -> POSSIBLE_FIRE (unlocalized)
        if yolo_fire < self.yolo_conf_fire and cnn_fire >= self.cnn_strong_threshold:
            return "POSSIBLE_FIRE", f"CNN strong fire {cnn_fire:.2f} but no YOLO localization ({yolo_fire:.2f})"

        # Rule 5: YOLO weak fire [0.25, 0.40) and CNN agrees -> POSSIBLE_FIRE
        if 0.25 <= yolo_fire < self.yolo_conf_fire and cnn_fire >= self.cnn_threshold:
            return "POSSIBLE_FIRE", f"Weak YOLO fire {yolo_fire:.2f} supported by CNN {cnn_fire:.2f}"

        # Rule 6: YOLO smoke detection
        if yolo_smoke >= self.yolo_conf_smoke:
            return "SMOKE", f"YOLO smoke {yolo_smoke:.2f}"

        return "NO_FIRE", f"Normal: YOLO fire {yolo_fire:.2f}, smoke {yolo_smoke:.2f}, CNN {cnn_fire:.2f}"

    def update(
        self,
        cnn_result: Optional[dict],
        yolo_result: Optional[list[dict]],
        cnn_available: bool = True,
        yolo_available: bool = True
    ) -> dict:
        """
        Process a new frame through candidate evaluation, sliding window,
        and event state manager. Returns full decision payload.
        """
        now = self.time_fn()
        candidate, frame_reason = self.evaluate_frame(
            cnn_result, yolo_result, cnn_available, yolo_available
        )
        self.window.append((candidate, frame_reason, now))

        # Count occurrences in sliding window
        counts = {"FIRE": 0, "SMOKE": 0, "POSSIBLE_FIRE": 0, "UNCERTAIN": 0, "NO_FIRE": 0, DecisionState.MODEL_UNAVAILABLE: 0}
        for state, _, _ in self.window:
            counts[state] = counts.get(state, 0) + 1

        total_frames = len(self.window)

        # Determine confirmed status
        if counts.get(DecisionState.MODEL_UNAVAILABLE, 0) == total_frames and total_frames > 0:
            confirmed_state = DecisionState.MODEL_UNAVAILABLE
            confirmed_reason = "Both models unavailable: waiting for weights installation"
        elif counts["FIRE"] >= self.confirm_frames:
            confirmed_state = DecisionState.FIRE_CONFIRMED
            confirmed_reason = f"FIRE confirmed in {counts['FIRE']}/{total_frames} frames ({frame_reason})"
        elif counts["SMOKE"] >= self.confirm_frames:
            confirmed_state = DecisionState.SMOKE_CONFIRMED
            confirmed_reason = f"SMOKE confirmed in {counts['SMOKE']}/{total_frames} frames ({frame_reason})"
        elif counts["POSSIBLE_FIRE"] >= self.confirm_possible_frames:
            confirmed_state = DecisionState.POSSIBLE_FIRE
            confirmed_reason = f"POSSIBLE FIRE observed in {counts['POSSIBLE_FIRE']}/{total_frames} frames ({frame_reason})"
        elif counts["UNCERTAIN"] >= self.confirm_frames:
            confirmed_state = DecisionState.UNCERTAIN
            confirmed_reason = f"UNCERTAIN / Model Disagreement in {counts['UNCERTAIN']}/{total_frames} frames ({frame_reason})"
        elif candidate != "NO_FIRE" and candidate != DecisionState.MODEL_UNAVAILABLE:
            confirmed_state = DecisionState.UNCERTAIN
            confirmed_reason = f"Awaiting temporal confirmation ({candidate} {counts[candidate]}/{self.confirm_frames})"
        elif candidate == DecisionState.MODEL_UNAVAILABLE:
            confirmed_state = DecisionState.MODEL_UNAVAILABLE
            confirmed_reason = "Models unavailable: waiting for weights installation"
        else:
            confirmed_state = DecisionState.NO_FIRE
            confirmed_reason = "No threat detected"

        # Event lifecycle & alert trigger calculation
        should_alert = False
        alert_type = None
        alert_message = None

        is_threat = confirmed_state in (DecisionState.FIRE_CONFIRMED, DecisionState.SMOKE_CONFIRMED, DecisionState.POSSIBLE_FIRE)

        if is_threat:
            self.last_confirmed_ts = now
            in_cooldown = (
                self.last_event_closed_ts is not None and 
                (now - self.last_event_closed_ts) < self.cooldown_seconds
            )

            if self.active_event_state is None:
                # Open a new event if not in cooldown
                if not in_cooldown:
                    self.active_event_state = confirmed_state
                    self.active_event_start = now
                    self.escalated = False
                    should_alert = True
                    alert_type = "fire_detected" if confirmed_state == DecisionState.FIRE_CONFIRMED else (
                        "smoke_detected" if confirmed_state == DecisionState.SMOKE_CONFIRMED else "possible_fire"
                    )
                    alert_message = confirmed_reason
            else:
                # Active event exists - check for escalation from SMOKE to FIRE
                if (
                    self.active_event_state in (DecisionState.SMOKE_CONFIRMED, DecisionState.POSSIBLE_FIRE)
                    and confirmed_state == DecisionState.FIRE_CONFIRMED
                    and not self.escalated
                ):
                    self.active_event_state = DecisionState.FIRE_CONFIRMED
                    self.escalated = True
                    should_alert = True
                    alert_type = "fire_detected"
                    alert_message = f"ESCALATION: {confirmed_reason}"
        else:
            # Threat cleared check
            if self.active_event_state is not None:
                if self.last_confirmed_ts and (now - self.last_confirmed_ts) >= self.clear_seconds:
                    self.last_event_closed_ts = now
                    self.active_event_state = None
                    self.escalated = False

        return {
            "status": confirmed_state,
            "candidate": candidate,
            "reason": confirmed_reason,
            "counts": counts,
            "window_size": total_frames,
            "event_active": self.active_event_state is not None,
            "active_event": self.active_event_state,
            "should_alert": should_alert,
            "alert_type": alert_type,
            "alert_message": alert_message,
            "timestamp": now
        }

    def reset(self):
        """Resets the sliding window and active event state."""
        self.window.clear()
        self.active_event_state = None
        self.active_event_start = None
        self.last_confirmed_ts = None
        self.last_event_closed_ts = None
        self.escalated = False
