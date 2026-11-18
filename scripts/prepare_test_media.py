"""
ForestGuard AI — Prepare Benchmark Test Media & Short Video Clips
Generates benchmark test frames and assembled video clips for end-to-end testing and demo fallback.
"""

import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "backend" / "data" / "test_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

def generate_benchmark_media():
    print("🎬 Generating benchmark test frames and video clips...")

    # 1. Fire Scene (Simulated flame core & glowing ember gradients)
    fire_frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background dark forest
        frame[:, :] = (15, 45, 20)
        # Dynamic flickering fire blob
        radius = int(80 + 15 * np.sin(i * 0.4))
        center_x = int(320 + 10 * np.cos(i * 0.3))
        center_y = int(280 + 8 * np.sin(i * 0.5))
        # Orange/red flame gradient
        cv2.circle(frame, (center_x, center_y), radius + 20, (0, 100, 255), -1)
        cv2.circle(frame, (center_x, center_y), radius, (0, 165, 255), -1)
        cv2.circle(frame, (center_x, center_y), int(radius * 0.6), (0, 230, 255), -1)
        cv2.putText(frame, "Benchmark Test: Wildfire Flame Event", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        fire_frames.append(frame)
        if i == 0:
            cv2.imwrite(str(MEDIA_DIR / "benchmark_fire.jpg"), frame)

    # 2. Smoke Scene (Simulated rising plume)
    smoke_frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mountain / sky background
        frame[:240, :] = (200, 180, 140)
        frame[240:, :] = (30, 60, 30)
        # Rising smoke column
        offset_y = (i * 3) % 40
        cv2.ellipse(frame, (320, 260 - offset_y), (100, 160), 10, 0, 360, (170, 170, 170), -1)
        cv2.ellipse(frame, (300, 210 - offset_y), (80, 120), -15, 0, 360, (190, 190, 190), -1)
        cv2.putText(frame, "Benchmark Test: Wildfire Smoke Plume", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        smoke_frames.append(frame)
        if i == 0:
            cv2.imwrite(str(MEDIA_DIR / "benchmark_smoke.jpg"), frame)

    # 3. Forest Clear Scene (Normal green canopy)
    forest_frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)
        # Subtle canopy texture
        cv2.circle(frame, (150, 120), 80, (20, 120, 20), -1)
        cv2.circle(frame, (450, 220), 100, (25, 110, 25), -1)
        cv2.putText(frame, "Benchmark Test: Clear Forest Canopy (No Fire)", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        forest_frames.append(frame)
        if i == 0:
            cv2.imwrite(str(MEDIA_DIR / "benchmark_forest.jpg"), frame)

    # 4. Hard Negative: Sunset (Golden hour light)
    sunset = np.zeros((480, 640, 3), dtype=np.uint8)
    sunset[:240, :] = (0, 140, 255) # Orange sky
    sunset[240:, :] = (20, 50, 20)  # Forest silhouette
    cv2.putText(sunset, "Hard Negative Benchmark: Sunset Golden Hour", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(MEDIA_DIR / "hard_negative_sunset.jpg"), sunset)

    # Write video clips
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out_fire = cv2.VideoWriter(str(MEDIA_DIR / "test_fire_clip.mp4"), fourcc, 10, (640, 480))
    for f in fire_frames: out_fire.write(f)
    out_fire.release()

    out_smoke = cv2.VideoWriter(str(MEDIA_DIR / "test_smoke_clip.mp4"), fourcc, 10, (640, 480))
    for f in smoke_frames: out_smoke.write(f)
    out_smoke.release()

    out_forest = cv2.VideoWriter(str(MEDIA_DIR / "test_forest_clip.mp4"), fourcc, 10, (640, 480))
    for f in forest_frames: out_forest.write(f)
    out_forest.release()

    print(f"✅ Benchmark test media generated in: {MEDIA_DIR}")

if __name__ == "__main__":
    generate_benchmark_media()
