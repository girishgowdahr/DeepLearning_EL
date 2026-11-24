import cv2
import uuid
import time
import base64
import threading
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.config import settings
from backend.services.pipeline import pipeline

router = APIRouter()

# In-memory video job store
video_jobs = {}

@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Analyzes an uploaded still image with both CNN and YOLO.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Process through pipeline
    result = await pipeline.process_frame(bgr, source="image_upload")

    # Encode annotated image to base64
    _, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "filename": file.filename,
        "decision": result["decision"],
        "cnn": result["cnn"],
        "yolo": result["yolo"],
        "latency_ms": result["latency_ms"],
        "fps": result["fps"],
        "snapshot_path": result["snapshot_path"],
        "annotated_image_b64": f"data:image/jpeg;base64,{b64_str}"
    }

def _process_video_job(job_id: str, video_path: Path):
    """Worker function for processing uploaded video clips at ~2 FPS."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            video_jobs[job_id]["status"] = "failed"
            video_jobs[job_id]["error"] = "Failed to open video file"
            return

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # Sample every Nth frame to achieve ~2 FPS
        sample_interval = max(1, int(orig_fps / 2))

        frame_idx = 0
        sampled_count = 0
        frames_summary = []
        flagged_previews = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                sampled_count += 1
                
                # Resize
                resized = pipeline.resize_for_inference(frame, max_dim=640)
                
                # YOLO & CNN
                yolo_dets = []
                if pipeline.decision_engine and pipeline.executor:
                    from backend.models.model_registry import registry
                    if registry.yolo.available:
                        yolo_dets = registry.yolo.predict(resized, conf=pipeline.decision_engine.yolo_conf_fire, imgsz=640)
                    cnn_res = None
                    if registry.cnn.available:
                        cnn_res = registry.cnn.predict(resized)

                    decision = pipeline.decision_engine.update(
                        cnn_result=cnn_res,
                        yolo_result=yolo_dets,
                        cnn_available=registry.cnn.available,
                        yolo_available=registry.yolo.available
                    )

                    state = decision["status"]
                    timestamp_sec = round(frame_idx / orig_fps, 2)
                    
                    frame_record = {
                        "frame_idx": frame_idx,
                        "time_sec": timestamp_sec,
                        "state": state,
                        "candidate": decision["candidate"],
                        "yolo_count": len(yolo_dets),
                        "cnn_fire": cnn_res["probs"].get("fire", 0) if cnn_res else 0
                    }
                    frames_summary.append(frame_record)

                    # Keep a preview snapshot if threat detected
                    if state != "NO_FIRE" and len(flagged_previews) < 10:
                        from backend.services.annotate import annotate_frame
                        ann = annotate_frame(resized, decision, yolo_dets, cnn_res, fps=2.0)
                        _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        flagged_previews.append({
                            "time_sec": timestamp_sec,
                            "state": state,
                            "b64": f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                        })

                # Update progress
                progress = min(100, int((frame_idx / max(1, total_video_frames)) * 100))
                video_jobs[job_id]["progress"] = progress

            frame_idx += 1

        cap.release()
        
        # Summary statistics
        fire_frames = sum(1 for f in frames_summary if "FIRE" in f["state"])
        smoke_frames = sum(1 for f in frames_summary if "SMOKE" in f["state"])

        video_jobs[job_id]["status"] = "completed"
        video_jobs[job_id]["progress"] = 100
        video_jobs[job_id]["total_sampled"] = sampled_count
        video_jobs[job_id]["fire_frames"] = fire_frames
        video_jobs[job_id]["smoke_frames"] = smoke_frames
        video_jobs[job_id]["frames_summary"] = frames_summary
        video_jobs[job_id]["flagged_previews"] = flagged_previews

    except Exception as e:
        video_jobs[job_id]["status"] = "failed"
        video_jobs[job_id]["error"] = str(e)

@router.post("/detect/video")
async def detect_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts video clip (<= 50MB, <= 60s) for background frame-by-frame analysis.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Video exceeds 50 MB limit.")

    job_id = str(uuid.uuid4())
    temp_path = settings.UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    video_jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "processing",
        "progress": 0,
        "started_at": time.time(),
        "error": None
    }

    # Run in background
    threading.Thread(target=_process_video_job, args=(job_id, temp_path), daemon=True).start()

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video queued for analysis at ~2 FPS"
    }

@router.get("/detect/video/{job_id}")
def get_video_job_status(job_id: str):
    if job_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video job not found.")
    return video_jobs[job_id]
