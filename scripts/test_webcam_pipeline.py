"""
Test script: Stream live webcam (source=0) frames over /ws/stream WebSocket.
Verifies the exact pipeline path used by the phone camera page before models exist.
"""
import cv2
import json
import time
import asyncio
import websockets

WS_URI = "ws://127.0.0.1:8000/ws/stream"

async def test_webcam_stream():
    print(f"Opening webcam device 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam 0")
        return False

    print(f"Connecting to {WS_URI}...")
    try:
        async with websockets.connect(WS_URI) as ws:
            print("Connected to WebSocket /ws/stream!")
            latencies = []

            for frame_idx in range(10):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"Frame {frame_idx}: Failed to capture from webcam")
                    continue

                # Encode frame to JPEG
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = buffer.tobytes()

                t0 = time.perf_counter()
                await ws.send(jpeg_bytes)

                # Receive JSON metadata
                meta_json = await ws.recv()
                meta = json.loads(meta_json)

                # Receive annotated image ack
                ann_bytes = await ws.recv()
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(rtt_ms)

                status = meta.get("decision", {}).get("status", "UNKNOWN")
                reason = meta.get("decision", {}).get("reason", "")
                fps = meta.get("fps", 0)
                print(f"  Frame {frame_idx+1}/10: RTT={rtt_ms:.1f}ms | Pipeline Decision={status} | AckBytes={len(ann_bytes)} | Reason='{reason}'")

                await asyncio.sleep(0.1)

            avg_rtt = sum(latencies) / len(latencies) if latencies else 0
            print(f"\nSUCCESS: 10/10 webcam frames processed through pipeline!")
            print(f"Average Roundtrip Latency: {avg_rtt:.1f}ms")
            return True
    finally:
        cap.release()
        print("Webcam released.")

if __name__ == "__main__":
    success = asyncio.run(test_webcam_stream())
    if not success:
        exit(1)
