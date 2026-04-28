"""
routes/health.py — /health and /model-info endpoints
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.model_service import service

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": service.model is not None}


@router.get("/model-info")
def model_info():
    info = service.checkpoint_info.copy()
    if not info:
        return JSONResponse(
            {"error": "Model not loaded"},
            status_code=503
        )

    # Prettify
    best_val = info.get("best_val")
    epoch    = info.get("epoch")

    return {
        "architecture": "3D Residual UNet",
        "in_channels": 4,
        "num_classes": 4,
        "class_names": ["Background", "ET (Enhancing Tumour)", "NET/NCR", "ED (Oedema)"],
        "base_features": 32,
        "num_params": info.get("num_params"),
        "device": info.get("device"),
        "model_path": info.get("model_path"),
        "training": {
            "epoch": (epoch + 1) if isinstance(epoch, int) else epoch,
            "best_val_mean_dice": round(float(best_val), 4) if best_val is not None else None,
        },
    }
