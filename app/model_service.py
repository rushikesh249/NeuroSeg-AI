"""
model_service.py — Singleton wrapper around UNet3D for inference.

Loads the model ONCE at startup; subsequent predict() calls reuse it.
Progress is reported via a shared dict keyed by job_id.
"""

import os
import sys
import uuid
import json
import time
import threading
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Add parent dir and scripts dir to sys.path so we can import from root/scripts
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_ROOT, "scripts")

for p in [_ROOT, _SCRIPTS]:
    if p not in sys.path:
        sys.path.insert(0, p)

from unet3d import UNet3D, count_parameters
from dataset import safe_load_nifti, LABEL_MAP

# ──────────────────────────────────────────────────────────────────────────────
#  Jobs registry  {job_id: {"status": str, "progress": int, "error": str|None}}
# ──────────────────────────────────────────────────────────────────────────────
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {}
        _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


# ──────────────────────────────────────────────────────────────────────────────
#  ModelService singleton
# ──────────────────────────────────────────────────────────────────────────────

class ModelService:
    _instance: Optional["ModelService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            # i7-3740QM is 4-core, 8-thread. 4 threads are often faster for 3D AI than 8.
            torch.set_num_threads(4)
        self.model_path: Optional[str] = None
        self.checkpoint_info: Dict[str, Any] = {}
        self.num_params: int = 0
        self._lock = threading.Lock()

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, model_path: str):
        """Load or reload the model from a .pth checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        state = torch.load(model_path, map_location=self.device)

        model = UNet3D(in_channels=4, base_features=32, num_classes=4)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_path = model_path
        self.num_params = count_parameters(model)

        # Pull training metadata if available
        self.checkpoint_info = {
            "epoch": state.get("epoch", None),
            "best_val": state.get("best_val", None),
            "device": str(self.device),
            "num_params": self.num_params,
            "model_path": model_path,
        }
        print(
            f"[ModelService] Loaded model | device={self.device} "
            f"| params={self.num_params:,} "
            f"| best_val={self.checkpoint_info.get('best_val')}"
        )

    # ── normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Per-channel z-normalisation. x shape: (C, D, H, W)"""
        out = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[0]):
            v = x[c]
            m, s = v.mean(), v.std()
            out[c] = (v - m) / (s if s > 1e-6 else 1.0)
        return out

    # ── padding helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _pad(x: np.ndarray, patch: Tuple[int, int, int]):
        C, D, H, W = x.shape
        pd, ph, pw = (max(0, p - s) for p, s in zip(patch, (D, H, W)))
        pads = ((0, 0), (pd // 2, pd - pd // 2),
                (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
        return np.pad(x, pads, constant_values=0.0), pads

    @staticmethod
    def _unpad(x: np.ndarray, pads):
        _, (d0, d1), (h0, h1), (w0, w1) = pads
        d_sl = slice(d0, None if d1 == 0 else -d1)
        h_sl = slice(h0, None if h1 == 0 else -h1)
        w_sl = slice(w0, None if w1 == 0 else -w1)
        return x[:, d_sl, h_sl, w_sl]

    # ── sliding window inference ──────────────────────────────────────────────

    def _sliding_window(
        self,
        volume: np.ndarray,
        patch_size: Tuple[int, int, int],
        job_id: str,
    ) -> np.ndarray:
        """Returns probability array shape (4, D, H, W)."""
        C, D, H, W = volume.shape
        pd, ph, pw = patch_size

        vol_p, pads = self._pad(volume, patch_size)
        _, Dp, Hp, Wp = vol_p.shape

        # CPU Optimisation: Increase stride to reduce redundant patches
        # 0.8 stride = 20% overlap. Much faster than 50% overlap.
        stride_d = int(pd * 0.8) if self.device.type == "cpu" else pd // 2
        stride_h = int(ph * 0.8) if self.device.type == "cpu" else ph // 2
        stride_w = int(pw * 0.8) if self.device.type == "cpu" else pw // 2

        def _positions(total, size, stride):
            pts = list(range(0, total - size + 1, stride))
            if not pts or pts[-1] != total - size:
                pts.append(total - size)
            return pts

        ds = _positions(Dp, pd, stride_d)
        hs = _positions(Hp, ph, stride_h)
        ws = _positions(Wp, pw, stride_w)

        total_patches = len(ds) * len(hs) * len(ws)
        processed = 0

        probs_sum = np.zeros((4, Dp, Hp, Wp), dtype=np.float32)
        counts    = np.zeros((Dp, Hp, Wp),    dtype=np.float32)

        for d in ds:
            for h in hs:
                for w in ws:
                    patch = vol_p[:, d:d+pd, h:h+ph, w:w+pw]
                    inp = torch.from_numpy(patch[None]).float().to(self.device)
                    with torch.no_grad():
                        out = self.model(inp)
                        if isinstance(out, tuple):
                            out = out[0]
                        prob = torch.softmax(out, dim=1).cpu().numpy()[0]

                    probs_sum[:, d:d+pd, h:h+ph, w:w+pw] += prob
                    counts[d:d+pd, h:h+ph, w:w+pw] += 1.0

                    processed += 1
                    pct = int(processed / total_patches * 90) + 5  # 5..95
                    _set_job(job_id, progress=pct)

        counts[counts == 0] = 1.0
        probs_avg = probs_sum / counts[None]
        return self._unpad(probs_avg, pads)

    # ── public predict API ────────────────────────────────────────────────────

    def predict(
        self,
        modality_paths: Dict[str, str],
        output_dir: str,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        fast_mode: bool = True,
    ) -> str:
        """
        Run segmentation on 4 NIfTI modalities.

        Parameters
        ----------
        modality_paths : dict with keys 'flair','t1','t1ce','t2'
        output_dir     : folder to write prediction .nii.gz
        patch_size     : sliding window patch

        Returns
        -------
        job_id : str
        """
        job_id = str(uuid.uuid4())
        _set_job(job_id, status="queued", progress=0, error=None,
                 output_path=None, stats=None)

        def _run():
            try:
                _set_job(job_id, status="loading", progress=2)

                # Load volumes
                keys = ["flair", "t1", "t1ce", "t2"]
                imgs = []
                affine = header = None
                for k in keys:
                    arr, affine, header = safe_load_nifti(modality_paths[k])
                    imgs.append(np.transpose(arr, (2, 0, 1)))  # H,W,D → D,H,W

                volume = np.stack(imgs, axis=0).astype(np.float32)
                volume_norm = self._normalize(volume)

                _set_job(job_id, status="running", progress=5)

                with self._lock:
                    if fast_mode:
                        print(f"[ModelService] Running in FAST MODE (Resolution Scaling)")
                        # Convert to torch tensor for fast interpolation
                        ten = torch.from_numpy(volume_norm[None]).to(self.device).float()
                        # Downsample to a manageable size (must be divisible by 16 for UNet)
                        target_size = (96, 128, 128) 
                        ten_small = F.interpolate(ten, size=target_size, mode='trilinear', align_corners=False)
                        
                        _set_job(job_id, progress=10)
                        
                        with torch.no_grad():
                            out_small = self.model(ten_small)
                            if isinstance(out_small, tuple):
                                out_small = out_small[0]
                            probs_small = torch.softmax(out_small, dim=1)
                        
                        _set_job(job_id, progress=70)
                        
                        # Upsample back to original H,W,D
                        orig_size = (volume.shape[1], volume.shape[2], volume.shape[3])
                        probs_ten = F.interpolate(probs_small, size=orig_size, mode='trilinear', align_corners=False)
                        probs = probs_ten.cpu().numpy()[0]
                    else:
                        probs = self._sliding_window(volume_norm, patch_size, job_id)

                _set_job(job_id, progress=95)

                # Argmax → label indices → remap to original BraTS values
                pred = np.argmax(probs, axis=0).astype(np.uint8)  # (D,H,W) 0..3
                inv_map = {v: k for k, v in LABEL_MAP.items()}
                pred_brats = np.vectorize(inv_map.get)(pred).astype(np.uint8)

                # Calculate per-class voxel volumes
                voxel_volume_mm3 = float(np.abs(np.linalg.det(affine[:3, :3])))
                stats = {}
                
                # Bounding box and center of mass for OVERALL tumor (labels > 0)
                tumor_indices = np.argwhere(pred_brats > 0)
                wt_count = int(np.sum(pred_brats > 0))
                
                if tumor_indices.size > 0:
                    # Indices are (D, H, W)
                    min_idx = tumor_indices.min(axis=0)
                    max_idx = tumor_indices.max(axis=0)
                    com_indices = tumor_indices.mean(axis=0)
                    
                    # Convert to NIfTI coordinates (H, W, D) for the UI
                    com_hwd = (float(com_indices[1]), float(com_indices[2]), float(com_indices[0]))
                    
                    # Size in mm
                    dx, dy, dz = abs(affine[0,0]), abs(affine[1,1]), abs(affine[2,2])
                    size_mm = [
                        float((max_idx[1] - min_idx[1]) * dx),
                        float((max_idx[2] - min_idx[2]) * dy),
                        float((max_idx[0] - min_idx[0]) * dz)
                    ]
                    
                    # Quadrant Determination (Simple 8-quadrant model)
                    depth_mid, height_mid, width_mid = np.array(pred_brats.shape) // 2
                    q_v = "Superior" if com_indices[0] > depth_mid else "Inferior"
                    q_h = "Posterior" if com_indices[1] > height_mid else "Anterior"
                    q_l = "Left" if com_indices[2] > width_mid else "Right"
                    
                    stats["location"] = {
                        "center_of_mass": com_hwd,
                        "description": f"{q_v} {q_h} {q_l}",
                        "coordinates": f"H:{com_hwd[0]:.0f}, W:{com_hwd[1]:.0f}, D:{com_hwd[2]:.0f}",
                        "bbox_mm": f"{size_mm[0]:.1f} x {size_mm[1]:.1f} x {size_mm[2]:.1f} mm",
                        "dimensions": {"x": size_mm[0], "y": size_mm[1], "z": size_mm[2]}
                    }
                else:
                    stats["location"] = {"center_of_mass": None, "description": "No tumor detected", "bbox_mm": "N/A"}

                # Whole Tumor Aggregate
                stats["WT"] = {
                    "voxels": wt_count,
                    "volume_mm3": wt_count * voxel_volume_mm3
                }

                # Map indices 1,2,3 back to diagnostic names ET, NET, ED for the UI
                name_map = {1: "NET", 2: "ED", 3: "ET"}
                for idx, name in name_map.items():
                    # inv_map.get(idx) gives the original BraTS label (1, 2, 4)
                    brats_label = inv_map.get(idx)
                    count = int(np.sum(pred_brats == brats_label))
                    stats[name] = {
                        "voxels": count,
                        "volume_mm3": count * voxel_volume_mm3,
                        "ratio_to_wt": (count / wt_count * 100) if wt_count > 0 else 0
                    }

                # Enhancement Ratio (specific clinical marker)
                et_vol = stats.get("ET", {}).get("volume_mm3", 0)
                stats["enhancement_index"] = (et_vol / (stats["WT"]["volume_mm3"])) if stats["WT"]["volume_mm3"] > 0 else 0


                # Save the result
                import nibabel as nib
                pred_hwz = np.transpose(pred_brats, (1, 2, 0))
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{job_id}_pred.nii.gz")
                nib.save(nib.Nifti1Image(pred_hwz, affine, header), out_path)

                # Save metadata for the gallery
                meta = {
                    "job_id": job_id,
                    "timestamp": time.time(),
                    "stats": stats,
                    "output_path": out_path
                }
                meta_path = out_path.replace(".nii.gz", ".json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                # Set final status ONLY AFTER file is saved and path is valid
                _set_job(job_id, status="done", progress=100,
                         output_path=out_path, stats=stats)

            except Exception as exc:
                import traceback
                _set_job(job_id, status="error", error=str(exc),
                         traceback=traceback.format_exc())

        threading.Thread(target=_run, daemon=True).start()
        return job_id

    def list_saved_jobs(self, predictions_dir: str) -> List[Dict[str, Any]]:
        """Scan predictions directory for meta.json files and return them sorted by date."""
        if not os.path.exists(predictions_dir):
            return []
        
        results = []
        for f in os.listdir(predictions_dir):
            if f.endswith(".json"):
                path = os.path.join(predictions_dir, f)
                try:
                    with open(path, "r") as jf:
                        data = json.load(jf)
                        results.append(data)
                except:
                    continue
        
        # Sort by timestamp descending
        results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return results


# Module-level singleton
service = ModelService()
