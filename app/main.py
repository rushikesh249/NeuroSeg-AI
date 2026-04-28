"""
main.py — FastAPI entrypoint for the Brain Tumour 3D Segmenter.

Start with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import zipfile
import io

from app.model_service import service
from app.routes.health import router as health_router
from app.routes.infer  import router as infer_router

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_FRONTEND_DIST = os.path.join(_ROOT, "frontend", "dist")
_MODEL   = os.path.join(_ROOT, "models", "best_model.pth")


# ── Lifespan (load model once on startup) ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    print("Initializing server. Model will load in the background...")
    
    # Run load in a thread so the server starts listening immediately
    thread = threading.Thread(target=service.load, args=(_MODEL,), daemon=True)
    thread.start()
    
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroSeg AI",
    description="3D Brain Tumor Segmentation with React + Tailwind.",
    version="1.2.0",
    lifespan=lifespan,
)

@app.get("/download-sample")
async def download_sample():
    sample_dir = os.path.join(_ROOT, "sample_data_original")
    if not os.path.isdir(sample_dir):
        raise HTTPException(status_code=404, detail="Sample data directory not found")
        
    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in os.listdir(sample_dir):
            if file.lower().endswith(('.nii', '.nii.gz')):
                file_path = os.path.join(sample_dir, file)
                zip_file.write(file_path, arcname=file)
                
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=neuroseg_sample_data.zip"}
    )

# include routers
app.include_router(health_router, tags=["System"])
app.include_router(infer_router,  tags=["Inference"])

# serve built frontend static files
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react(full_path: str):
    """Serve the React app for all non-API routes."""
    # Check if the requested path is an actual file in dist
    file_path = os.path.join(_FRONTEND_DIST, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Fallback to index.html for React Router styling / SPA support
    index_path = os.path.join(_FRONTEND_DIST, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    
    # Final fallback if nothing exists yet (useful during initial dev)
    return {"message": "NeuroSeg AI - Backend Live. Please build the frontend."}
