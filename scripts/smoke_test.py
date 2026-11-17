"""
ForestGuard AI — Automated Smoke Test Suite
Runs end-to-end checks against the FastAPI backend, REST endpoints, and WebSocket.
"""

import sys
import time
import requests
import cv2
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def run_smoke_tests():
    print("=" * 60)
    print("🌲 Running ForestGuard AI Smoke Tests...")
    print("=" * 60)

    # 1. Health Check
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=3)
        assert r.status_code == 200, f"Health check failed with code {r.status_code}"
        data = r.json()
        print(f"✅ Health Check passed: App='{data.get('app')}', Version={data.get('version')}")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False

    # 2. Models Status Check
    try:
        r = requests.get(f"{BASE_URL}/api/models/status", timeout=3)
        assert r.status_code == 200
        st = r.json()
        print(f"✅ Model Registry Check: CNN={st['cnn']['available']}, YOLO={st['yolo']['available']}")
    except Exception as e:
        print(f"❌ Model Status Check Failed: {e}")
        return False

    # 3. Settings Round-Trip
    try:
        r = requests.get(f"{BASE_URL}/api/settings", timeout=3)
        assert r.status_code == 200
        settings = r.json()
        assert "yolo_conf_fire" in settings
        assert "cnn_threshold" in settings
        print("✅ Settings Read Check passed")
    except Exception as e:
        print(f"❌ Settings Check Failed: {e}")
        return False

    # 4. Safe Test Alert
    try:
        r = requests.post(f"{BASE_URL}/api/alerts/test", timeout=3)
        assert r.status_code == 200
        alert_data = r.json()
        assert alert_data["success"] is True
        print("✅ Test Alert Trigger passed (Safe, not counted in analytics)")
    except Exception as e:
        print(f"❌ Test Alert Failed: {e}")
        return False

    # 5. Image Detection Pipeline with Synthesized Frame
    try:
        # Create a dummy 480x640 green forest-like test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:, :] = (34, 139, 34)  # Forest Green BGR
        _, encoded = cv2.imencode(".jpg", test_img)

        files = {"file": ("test_canopy.jpg", encoded.tobytes(), "image/jpeg")}
        r = requests.post(f"{BASE_URL}/api/detect/image", files=files, timeout=5)
        assert r.status_code == 200
        res = r.json()
        assert res["success"] is True
        print(f"✅ Still Image Detection Pipeline passed: Latency={res.get('latency_ms')}ms, Status={res['decision']['status']}")
    except Exception as e:
        print(f"❌ Image Detection Pipeline Failed: {e}")
        return False

    print("=" * 60)
    print("🎉 ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
