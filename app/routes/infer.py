"""
routes/infer.py — /predict, /progress/{job_id}, /result/{job_id}
"""

import os
import asyncio
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from app.model_service import service, get_job

router = APIRouter()

PREDICTIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "predictions"
)
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads"
)


async def _save_upload(upload: UploadFile, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 1024):  # 1MB chunks
            f.write(chunk)


@router.post("/predict")
async def predict(
    flair: UploadFile = File(...),
    t1:    UploadFile = File(...),
    t1ce:  UploadFile = File(...),
    t2:    UploadFile = File(...),
    patch_size: Optional[str] = Form("128,128,128"),
    fast: bool = Form(True)
):
    """
    Accept 4 NIfTI files (.nii or .nii.gz) and run segmentation.
    Returns a job_id for polling via /progress/{job_id}.
    """
    if service.model is None:
        raise HTTPException(503, "Model not loaded on server.")

    # Parse patch size
    try:
        ps = tuple(int(x) for x in patch_size.split(","))
        if len(ps) != 3:
            raise ValueError
    except (ValueError, AttributeError):
        raise HTTPException(400, "patch_size must be three comma-separated integers, e.g. '128,128,128'")

    # Create a temp dir for this upload
    import uuid
    upload_id = str(uuid.uuid4())
    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    os.makedirs(upload_path, exist_ok=True)

    modality_paths = {}
    for name, upload in [("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2)]:
        ext = ".nii.gz" if str(upload.filename).endswith(".nii.gz") else ".nii"
        dest = os.path.join(upload_path, f"{name}{ext}")
        await _save_upload(upload, dest)
        modality_paths[name] = dest

    job_id = service.predict(modality_paths, PREDICTIONS_DIR, patch_size=ps, fast_mode=fast)
    return {"job_id": job_id}


@router.get("/progress/{job_id}")
async def progress(job_id: str):
    """SSE stream: sends {status, progress, error} updates until done/error."""

    async def event_stream():
        import json
        prev_progress = -1
        while True:
            job = get_job(job_id)
            if not job:
                data = json.dumps({"status": "not_found", "progress": 0})
                yield f"data: {data}\n\n"
                break

            cur_progress = job.get("progress", 0)
            status       = job.get("status", "unknown")

            if cur_progress != prev_progress or status in ("done", "error", "not_found"):
                payload = {
                    "status":   status,
                    "progress": cur_progress,
                    "error":    job.get("error"),
                }
                if status == "done":
                    payload["stats"] = job.get("stats")
                yield f"data: {json.dumps(payload)}\n\n"
                prev_progress = cur_progress

            if status in ("done", "error", "not_found"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/result/{job_id}/segmentation")
def download_segmentation(job_id: str):
    """Download the predicted NIfTI segmentation."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(409, f"Job status: {job.get('status')}")
    out_path = job.get("output_path")
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(500, "Output file missing")

    return FileResponse(
        out_path,
        media_type="application/gzip",
        filename=f"segmentation_{job_id}.nii.gz"
    )


@router.get("/result/{job_id}/stats")
def result_stats(job_id: str):
    """Return volumetric stats for the segmented tumour classes."""
    job = get_job(job_id)
    if not job:
        # Fallback to disk if it's an old job
        results = service.list_saved_jobs(PREDICTIONS_DIR)
        for r in results:
            if r["job_id"] == job_id:
                return r
        raise HTTPException(404, "Job not found")
        
    if job.get("status") != "done":
        raise HTTPException(409, f"Job status: {job.get('status')}")
    return {"job_id": job_id, "stats": job.get("stats", {})}


@router.get("/results")
def list_results():
    """Returns a list of all historical results stored on disk."""
    return service.list_saved_jobs(PREDICTIONS_DIR)
