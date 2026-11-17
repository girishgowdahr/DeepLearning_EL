import time
from typing import Optional

# Color definitions in BGR format
COLOR_GREEN = (34, 197, 94)      # #22c55e OK / NO_FIRE
COLOR_AMBER = (11, 158, 245)     # #f59e0b Warning / UNCERTAIN
COLOR_RED = (68, 68, 239)        # #ef4444 Danger / FIRE
COLOR_ORANGE = (0, 140, 255)     # SMOKE
COLOR_DARK_BAR = (23, 27, 17)    # Control room header background
COLOR_WHITE = (255, 255, 255)

def annotate_frame(
    bgr_image,
    decision_result: dict,
    yolo_detections: Optional[list[dict]] = None,
    cnn_result: Optional[dict] = None,
    fps: float = 0.0
):
    """
    Annotates a video or image frame with YOLO bounding boxes,
    a top status banner with decision state, CNN score, and timestamp.
    """
    import cv2
    import numpy as np

    annotated = bgr_image.copy()
    h, w = annotated.shape[:2]

    # 1. Draw YOLO Bounding Boxes
    if yolo_detections:
        for det in yolo_detections:
            cls_name = det.get("cls", "threat").lower()
            conf = det.get("conf", 0.0)
            xyxy = det.get("xyxy", [0, 0, 0, 0])
            x1, y1, x2, y2 = xyxy

            # Clip box to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            color = COLOR_RED if cls_name == "fire" else COLOR_ORANGE
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label tag
            label_text = f"{cls_name.upper()} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tag_y1 = max(0, y1 - th - baseline - 4)
            tag_y2 = y1
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, tag_y2 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_WHITE,
                1,
                cv2.LINE_AA
            )

    # 2. Draw Top Control-Room Banner
    banner_height = 36
    status = decision_result.get("status", "NO_FIRE")
    
    if status == "FIRE_CONFIRMED":
        status_color = COLOR_RED
        status_label = "FIRE CONFIRMED"
    elif status == "SMOKE_CONFIRMED":
        status_color = COLOR_ORANGE
        status_label = "SMOKE CONFIRMED"
    elif status == "POSSIBLE_FIRE":
        status_color = COLOR_AMBER
        status_label = "POSSIBLE FIRE"
    elif status == "UNCERTAIN":
        status_color = COLOR_AMBER
        status_label = "UNCERTAIN"
    else:
        status_color = COLOR_GREEN
        status_label = "NORMAL (NO FIRE)"

    # Draw semi-transparent header bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), COLOR_DARK_BAR, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Status pill indicator
    cv2.circle(annotated, (16, banner_height // 2), 6, status_color, -1)
    cv2.putText(
        annotated,
        status_label,
        (30, banner_height // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA
    )

    # CNN Info
    cnn_fire_prob = 0.0
    if cnn_result and "probs" in cnn_result:
        cnn_fire_prob = cnn_result["probs"].get("fire", 0.0)
    elif cnn_result and cnn_result.get("label") == "fire":
        cnn_fire_prob = cnn_result.get("confidence", 0.0)

    cnn_text = f"CNN Fire: {cnn_fire_prob:.2f}"
    cv2.putText(
        annotated,
        cnn_text,
        (w // 2 - 50, banner_height // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # FPS & Timestamp
    fps_text = f"{fps:.1f} FPS" if fps > 0 else ""
    time_str = time.strftime("%H:%M:%S")
    right_text = f"{fps_text}  {time_str}".strip()
    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(
        annotated,
        right_text,
        (w - rw - 12, banner_height // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    return annotated
