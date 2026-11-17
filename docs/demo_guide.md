# ForestGuard AI — Academic Demo Presentation Guide

This guide provides a step-by-step presentation script and troubleshooting instructions for demonstrating ForestGuard AI during project evaluation.

---

## 1. Demo Walkthrough Script (5–7 Minutes)

### Step 1: System Launch (30 seconds)
1. Open PowerShell and run:
   ```powershell
   .\scripts\run_demo.ps1
   ```
2. Explain to evaluators:
   > *"ForestGuard AI runs FastAPI and the compiled React dashboard on a single unified port (8000). For remote phone camera streaming, it automatically provisions a free Cloudflare HTTPS tunnel so mobile devices can access WebRTC/getUserMedia without self-signed certificate warnings."*
3. Open `http://localhost:8000` in the laptop browser.
4. Click **Enable Sound** on the top header once (unmutes the HTML5 Web Audio API siren).

### Step 2: Overview & Real Architecture (1 minute)
1. Show the **Overview** page (`/`):
   - Highlight the **KPI cards**: Frames Processed, Fire Detections, Smoke Detections, No-Fire Classifications.
   - Point out that **all numbers are genuine** and increment only from real processed frames.
   - Show the system status indicators: Backend Online, CNN Ready, YOLO Ready.

### Step 3: Connect Android Phone Camera (2 minutes)
1. Navigate to **Live Camera** (`/live`).
2. Point your Android phone camera at the QR code displayed on the right sidebar.
3. Open the scanned URL (or manually navigate to `<Tunnel_URL>/camera`).
4. Tap **Start Camera** on the phone.
5. Grant camera permission.
6. Hold up the phone and point it around the room:
   - Evaluators will see the live annotated video stream on the laptop screen.
   - Point to the **CNN panel** (fire vs no-fire probability bars) and **YOLO panel** (detected boxes).
   - Point to the **Status banner**: shows `NORMAL (NO FIRE)` with green indicator.

### Step 4: Threat Simulation & Decision Fusion (2 minutes)
1. Point the phone camera at a controlled test stimulus (e.g. an image of a forest fire displayed on another screen or tablet):
   - Watch YOLO immediately draw red bounding boxes around the flame fronts.
   - Watch the CNN fire probability spike above 80%.
   - Point to the Decision Engine status: turns red **`FIRE CONFIRMED`**.
   - The **Web Audio siren** will immediately sound an alternating 880/660 Hz alarm!
   - Show the red emergency banner at the top of the dashboard.
2. Show **Alert Suppression**:
   - Point out that although frames continue streaming, the siren does not trigger repeatedly; the decision engine groups the outbreak into a single continuous event.
3. Remove the fire image:
   - Notice that after 10 seconds of clear frames, the event clears.

### Step 5: History, Analytics & Transparent Disclosures (1 minute)
1. Click **History & Logs** (`/history`):
   - Show the newly recorded fire detection with its exact timestamp, confidence, and snapshot thumbnail.
   - Click **View** to inspect the evidence frame in the modal dialog.
   - Click **Export CSV** to demonstrate audit logging.
2. Click **Model Info** (`/models`):
   - Point out the honest documentation of limitations (sunsets, orange objects, CPU latency).
   - Show that datasets and models are transparently documented.

---

## 2. Troubleshooting Guide

| Problem Encountered | Root Cause | Solution |
|---|---|---|
| Phone shows *"Camera permission was denied"* | Android Chrome permission prompt was dismissed or blocked | Tap the lock icon in the Chrome URL bar → Site Settings → Reset permissions → Reload page. |
| Phone says *"Insecure Context"* | Opened via plain `http://<LAN_IP>:8000` | Android requires HTTPS for camera access. Connect using the Cloudflare URL or run `.\scripts\run_demo.ps1 -SelfSigned`. |
| Cloudflare Tunnel fails on College Wi-Fi | Campus firewall blocking outbound tunnel connections | **Switch laptop and phone to a personal mobile hotspot**, then re-run `run_demo.ps1`. |
| No siren sound plays on threat detection | Browser audio autoplay blocked | Click the **Enable Sound** button in the top header once to unlock the Web Audio API context. |
| Video stream appears laggy / low FPS | Heavy background apps competing for CPU | Close other browser tabs and heavy apps. The pipeline is designed to maintain ~3–5 FPS on CPU. |
| Status shows *"Model Unavailable"* | Weights missing from `model_weights/` | Place `cnn_mobilenetv2.onnx` and `yolo_best.pt` in `model_weights/` and click **Reload Model Files** in Settings. |
