"""
Phone-camera / remote-access tests.

"Remote" requests are simulated with a Cloudflare forwarding header, exactly like traffic that
arrives through `cloudflared tunnel` (the TCP peer is loopback, the header marks it as tunneled).
These are automated protocol/security tests; they do NOT prove a real Android phone works.
"""
import json
import time
import cv2
import numpy as np
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.main import app
from backend.config import settings
from backend.database import db
from backend.services import security
from backend.api import ws as ws_module

TOKEN = "unit-test-token-0123456789abcdef"
TUNNEL = {"Cf-Connecting-Ip": "203.0.113.9"}
DIST = settings.BASE_DIR / "frontend" / "dist"


@pytest.fixture(autouse=True)
def phone_token(monkeypatch):
    monkeypatch.setenv("FG_PHONE_TOKEN", TOKEN)
    security.reset_token_cache_for_tests()
    yield
    security.reset_token_cache_for_tests()


@pytest.fixture
def client():
    db.init_db()
    with TestClient(app) as c:
        yield c


def jpeg(width=320, height=240, color=(34, 139, 34)) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def auth_msg(token=TOKEN) -> str:
    return json.dumps({"type": "auth", "token": token})


def alert_count(alert_type: str) -> int:
    return sum(1 for a in db.get_alerts(limit=500) if a.get("type") == alert_type)


# ---------------------------------------------------------------- tunnel access guard
@pytest.mark.parametrize("path", [
    "/api/health", "/api/detections", "/api/alerts", "/api/analytics", "/api/settings",
    "/api/camera/status", "/api/system/network", "/docs", "/openapi.json", "/results/metrics/x.json",
])
def test_tunneled_requests_cannot_reach_dashboard_or_api(client, path):
    assert client.get(path, headers=TUNNEL).status_code == 403


def test_tunneled_requests_cannot_change_state(client):
    assert client.put("/api/settings", json={"yolo_conf_fire": 0.1}, headers=TUNNEL).status_code == 403
    assert client.post("/api/camera/start", json={"source": "webcam"}, headers=TUNNEL).status_code == 403
    assert client.post("/api/alerts/test", headers=TUNNEL).status_code == 403


def test_local_requests_still_work(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/settings").status_code == 200


@pytest.mark.skipif(not (DIST / "index.html").exists(), reason="frontend not built")
def test_tunneled_phone_page_is_reachable_but_dashboard_routes_are_not(client):
    assert client.get("/camera", headers=TUNNEL).status_code == 200
    assert client.get("/history", headers=TUNNEL).status_code == 403
    assert client.get("/", headers=TUNNEL).status_code == 403


def test_tunneled_events_websocket_is_rejected(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/events", headers=TUNNEL):
            pass
    assert exc.value.code == 4403


def test_other_origin_is_not_whitelisted_by_cors(client):
    r = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") != "*"
    assert r.headers.get("access-control-allow-origin") != "https://evil.example"


@pytest.mark.skipif(not (DIST / "index.html").exists(), reason="frontend not built")
def test_spa_route_does_not_serve_files_outside_dist(client):
    probe = DIST.parent / "traversal_probe_test.txt"
    probe.write_text("SECRET-OUTSIDE-DIST")
    try:
        for path in ("/..%2ftraversal_probe_test.txt", "/%2e%2e/traversal_probe_test.txt"):
            r = client.get(path)
            assert "SECRET-OUTSIDE-DIST" not in r.text
    finally:
        probe.unlink(missing_ok=True)


def test_unknown_api_path_is_404_not_200(client):
    if (DIST / "index.html").exists():
        assert client.get("/api/does-not-exist").status_code == 404


# ---------------------------------------------------------------- phone config route
def test_phone_config_requires_token_when_remote(client):
    assert client.get("/api/phone/config", headers=TUNNEL).status_code == 401
    assert client.get("/api/phone/config", headers={**TUNNEL, "X-FG-Token": "wrong"}).status_code == 401
    r = client.get("/api/phone/config", headers={**TUNNEL, "X-FG-Token": TOKEN})
    assert r.status_code == 200
    cfg = r.json()
    assert 320 <= cfg["send_width"] <= 960 and 0.3 <= cfg["jpeg_quality"] <= 0.9 and 1 <= cfg["fps_cap"] <= 10


def test_phone_config_open_for_local(client):
    assert client.get("/api/phone/config").status_code == 200


# ---------------------------------------------------------------- /ws/stream authentication
def test_remote_stream_without_auth_is_rejected(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream", headers=TUNNEL) as ws:
            ws.send_bytes(jpeg())
            ws.receive_text()
    assert exc.value.code == 4401


def test_remote_stream_with_wrong_token_is_rejected(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream", headers=TUNNEL) as ws:
            ws.send_text(auth_msg("wrong-token"))
            ws.receive_text()
    assert exc.value.code == 4401


def test_remote_stream_with_correct_token_round_trips_a_frame(client):
    with client.websocket_connect("/ws/stream", headers=TUNNEL) as ws:
        ws.send_text(auth_msg())
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "auth_ok" and "send_width" in hello["config"]
        ws.send_bytes(jpeg())
        meta = json.loads(ws.receive_text())
        assert "decision" in meta and "fps" in meta and "latency_ms" in meta
        annotated = ws.receive_bytes()
        decoded = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape[0] > 0


def test_local_stream_works_without_token_for_existing_scripts(client):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(jpeg())
        meta = json.loads(ws.receive_text())
        assert "decision" in meta
        assert ws.receive_bytes()


def test_token_is_never_accepted_from_the_url_query(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/stream?token={TOKEN}", headers=TUNNEL) as ws:
            ws.send_bytes(jpeg())
            ws.receive_text()
    assert exc.value.code == 4401


# ---------------------------------------------------------------- stream robustness
def test_bad_frames_do_not_kill_the_stream(client):
    with client.websocket_connect("/ws/stream", headers=TUNNEL) as ws:
        ws.send_text(auth_msg())
        assert json.loads(ws.receive_text())["type"] == "auth_ok"

        ws.send_bytes(b"this is not a jpeg")
        assert json.loads(ws.receive_text())["error"] == "invalid_frame"

        ws.send_bytes(b"\xff" * (ws_module.MAX_FRAME_BYTES + 10))
        assert json.loads(ws.receive_text())["error"] == "frame_too_large"

        ws.send_bytes(jpeg())  # stream still alive
        assert "decision" in json.loads(ws.receive_text())
        assert ws.receive_bytes()


def test_newer_phone_session_replaces_older_one(client):
    with client.websocket_connect("/ws/stream", headers=TUNNEL) as first:
        first.send_text(auth_msg())
        assert json.loads(first.receive_text())["type"] == "auth_ok"
        with client.websocket_connect("/ws/stream", headers=TUNNEL) as second:
            second.send_text(auth_msg())
            assert json.loads(second.receive_text())["type"] == "auth_ok"
            with pytest.raises(WebSocketDisconnect) as exc:
                first.receive_text()
            assert exc.value.code == 4409
            second.send_bytes(jpeg())
            assert "decision" in json.loads(second.receive_text())


def test_stalled_stream_raises_exactly_one_disconnect_alert_then_recovers(client, monkeypatch):
    monkeypatch.setattr(ws_module, "STALE_ALERT_S", 0.4)
    monkeypatch.setattr(ws_module, "WATCHDOG_TICK_S", 0.1)
    monkeypatch.setattr(ws_module, "IDLE_CLOSE_S", 30.0)
    before = alert_count("camera_disconnected")
    with client.websocket_connect("/ws/stream", headers=TUNNEL) as ws:
        ws.send_text(auth_msg())
        assert json.loads(ws.receive_text())["type"] == "auth_ok"
        ws.send_bytes(jpeg())
        ws.receive_text(); ws.receive_bytes()
        time.sleep(1.5)  # phone "goes silent" (e.g. Wi-Fi drop)
        assert alert_count("camera_disconnected") == before + 1  # one alert, not one per tick
        ws.send_bytes(jpeg())  # frames resume
        assert "decision" in json.loads(ws.receive_text())
        ws.receive_bytes()
    # A clean close after recovery must not add another alert
    time.sleep(0.3)
    assert alert_count("camera_disconnected") == before + 1


def test_clean_stop_does_not_raise_disconnect_alert(client):
    before = alert_count("camera_disconnected")
    with client.websocket_connect("/ws/stream", headers=TUNNEL) as ws:
        ws.send_text(auth_msg())
        ws.receive_text()
        ws.send_bytes(jpeg())
        ws.receive_text(); ws.receive_bytes()
    time.sleep(0.3)
    assert alert_count("camera_disconnected") == before


def test_token_verification_helpers():
    assert security.verify_phone_token(TOKEN)
    assert not security.verify_phone_token("")
    assert not security.verify_phone_token(None)
    assert not security.verify_phone_token(TOKEN + "x")
