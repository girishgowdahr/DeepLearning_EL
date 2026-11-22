"""
ForestGuard AI — Android phone verification helper (ADB + live app).

What it automates:
  1. Finds an authorised Android device with `adb devices`.
  2. Reads the phone URL from the RUNNING app (http://localhost:8000/api/system/network).
  3. Opens that URL in Chrome on the phone (`adb shell am start`).
  4. Watches the app (camera status + /ws/events) while you tap Start and Allow on the phone.

What it deliberately does NOT do: tap the screen, grant the camera permission, or dismiss any prompt.
Those are approval steps that a human must confirm. It only reports what the server really observed.

Usage (laptop, app already running via scripts/run_demo.ps1):
    python scripts/android_check.py [--base http://localhost:8000] [--wait 120] [--listen 30] [--adb adb]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request


def http_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def adb_devices(adb):
    p = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=20)
    devices = []
    for line in (p.stdout or "").splitlines()[1:]:
        m = re.match(r"^(\S+)\s+(device|unauthorized|offline)\b", line.strip())
        if m:
            devices.append((m.group(1), m.group(2)))
    return devices


def listen_events(base, seconds):
    """Count /ws/events messages for a while. Returns (counts_by_type, error_or_None)."""
    try:
        from websockets.sync.client import connect
    except Exception as e:  # websockets missing or too old
        return {}, f"websockets client unavailable ({e})"
    url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws/events"
    counts, end = {}, time.time() + seconds
    try:
        with connect(url, open_timeout=5) as ws:
            while time.time() < end:
                try:
                    msg = ws.recv(timeout=max(0.1, min(1.0, end - time.time())))
                except TimeoutError:
                    continue
                try:
                    t = json.loads(msg).get("type", "unknown")
                except Exception:
                    t = "unparsed"
                counts[t] = counts.get(t, 0) + 1
    except Exception as e:
        return counts, f"events socket error: {type(e).__name__}: {e}"
    return counts, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--adb", default="adb")
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for the phone to start streaming")
    ap.add_argument("--listen", type=int, default=30, help="seconds to listen to /ws/events once streaming")
    ap.add_argument("--package", default="com.android.chrome")
    a = ap.parse_args()

    auto, human = [], []

    # --- ADB
    if not shutil.which(a.adb) and not shutil.which(a.adb + ".exe"):
        print("RESULT: ADB NOT FOUND. Install Android platform-tools and enable USB debugging on the phone.")
        print("        Automated phone launch is therefore NOT verified. Use the manual steps in docs/demo_guide.md.")
        return 2
    devices = adb_devices(a.adb)
    ready = [d for d, s in devices if s == "device"]
    print("ADB devices:", devices or "none")
    if not ready:
        if any(s == "unauthorized" for _, s in devices):
            print("RESULT: device is UNAUTHORIZED. Unlock the phone and accept the 'Allow USB debugging' prompt, then rerun.")
        else:
            print("RESULT: NO ANDROID DEVICE CONNECTED over ADB. Android verification NOT performed.")
        return 2
    serial = ready[0]
    auto.append(f"ADB device connected: {serial}")

    # --- App running?
    try:
        net = http_json(a.base + "/api/system/network")
        http_json(a.base + "/api/health")
    except Exception as e:
        print(f"RESULT: app not reachable at {a.base} ({type(e).__name__}). Start it first: scripts\\run_demo.ps1")
        return 2
    url = net.get("phone_camera_url", "")
    if not url.startswith("https://"):
        print("RESULT: no HTTPS phone URL available (start the Cloudflare tunnel or use -SelfSigned). Android Chrome cannot open the camera over http://.")
        return 2
    auto.append("App reachable; HTTPS phone URL available (token not printed)")

    # --- Launch Chrome on the phone (the token stays in the URL fragment; never printed here)
    p = subprocess.run([a.adb, "-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", url, a.package],
                       capture_output=True, text=True, timeout=30)
    if p.returncode != 0 or "Error" in (p.stdout + p.stderr):
        print("RESULT: could not launch Chrome via ADB:", (p.stdout + p.stderr).strip()[-300:])
        return 1
    auto.append("Chrome launched on the phone with the ForestGuard camera page")
    print("\nChrome opened on the phone. NOW, ON THE PHONE (human steps this script cannot do):")
    print("  1) tap  Start Camera   2) tap  Allow  on the camera permission prompt")
    print(f"Waiting up to {a.wait}s for the server to see the stream ...\n")

    # --- Wait for the stream
    streaming, status = False, {}
    end = time.time() + a.wait
    while time.time() < end:
        try:
            status = http_json(a.base + "/api/camera/status")
        except Exception:
            status = {}
        if status.get("source") == "browser" and status.get("connected") and (status.get("fps") or 0) > 0:
            streaming = True
            break
        time.sleep(2)
    if not streaming:
        print(f"RESULT: no phone stream observed within {a.wait}s (last camera status: {status}).")
        print("        Either the permission was not granted, the page did not load, or the tunnel is blocked.")
        for line in auto:
            print("  [auto ok]", line)
        return 1
    auto.append(f"Server observed the phone stream: fps={status.get('fps')}, last_frame_age_s={status.get('last_frame_age_s')}")

    # --- Events
    print(f"Streaming confirmed. Listening to /ws/events for {a.listen}s ...")
    counts, err = listen_events(a.base, a.listen)
    print("Event counts:", counts or "none", f"(note: {err})" if err else "")
    if err and not counts:
        human.append(f"/ws/events was NOT checked ({err}). Run this script with the project venv Python "
                     "(.venv\\Scripts\\python.exe scripts\\android_check.py) or watch detections on the dashboard yourself.")
    elif counts.get("detection"):
        auto.append(f"{counts['detection']} per-frame detection events received (event pipeline works; boxes only exist if trained weights are loaded - check them by eye)")
    else:
        human.append("No 'detection' events seen: with no trained weights (MODEL_UNAVAILABLE) or a plain scene this is expected. "
                     "Show a fire image to the phone camera and rerun to check bounding boxes.")
    if err and not counts:
        pass  # already reported above
    elif counts.get("alert"):
        auto.append(f"{counts['alert']} alert events received")
    else:
        human.append("No alert events seen. Alerts need a CONFIRMED fire/smoke event (needs trained weights + a fire image in view).")

    human += [
        "Rear camera really used (look at the phone preview)",
        "Annotated video and bounding boxes visible and correct on the laptop dashboard",
        "Siren audible after clicking Enable Sound; exactly one siren per event",
        "Airplane-mode toggle for 10 s -> phone shows Reconnecting, then Streaming again",
        "Screen-lock behaviour and 2-3 minute stability",
    ]
    print("\nAUTOMATICALLY VERIFIED:")
    for x in auto:
        print("  [ok]", x)
    print("\nREQUIRES HUMAN CONFIRMATION:")
    for x in human:
        print("  [ ]", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
