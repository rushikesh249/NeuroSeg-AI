import os
import argparse
import math
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from unet3d import UNet3D
try:
    from monai.inferers import sliding_window_inference as monai_sw_inference
    _have_monai_infer = True
except Exception:
    _have_monai_infer = False
from dataset import safe_load_nifti, LABEL_MAP, INV_LABEL_MAP


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def load_case_volumes(root: str, case_id: str) -> Tuple[np.ndarray, object, object]:
    # expects 4 files: case_id_0000.._0003 under root/imagesTs or imagesTr
    img_dir = os.path.join(root, 'imagesTs')
    if not os.path.isdir(img_dir):
        # try imagesTr fallback
        img_dir = os.path.join(root, 'imagesTr')
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Neither imagesTs nor imagesTr found under {root}")

    imgs = []
    affine = None
    header = None
    for i in range(4):
        p = os.path.join(img_dir, f"{case_id}_000{i}.nii.gz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing modality file: {p}")
        arr, affine, header = safe_load_nifti(p)
        # nibabel returns HxWxD; transpose to D x H x W
        arr = np.transpose(arr, (2, 0, 1))
        imgs.append(arr)

    x = np.stack(imgs, axis=0).astype(np.float32)  # (C, D, H, W)
    return x, affine, header


def normalize_volume(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        v = x[c]
        m = v.mean()
        s = v.std()
        if s < 1e-6:
            out[c] = v - m
        else:
            out[c] = (v - m) / s
    return out


def pad_to_shape(x: np.ndarray, desired: Tuple[int, int, int]):
    # x shape (C,D,H,W)
    C, D, H, W = x.shape
    pd, ph, pw = desired
    pdiff = max(0, pd - D)
    hdiff = max(0, ph - H)
    wdiff = max(0, pw - W)
    pad = ((0, 0), (pdiff // 2, pdiff - pdiff // 2), (hdiff // 2, hdiff - hdiff // 2), (wdiff // 2, wdiff - wdiff // 2))
    if any([s > 0 for s in (pdiff, hdiff, wdiff)]):
        x = np.pad(x, pad_width=pad, constant_values=0.0)
    return x, pad


def unpad_from_shape(x: np.ndarray, pad) -> np.ndarray:
    # remove pad given pad tuple
    if pad is None:
        return x
    if x.ndim == 4:
        _, Dpad, Hpad, Wpad = pad
        d0, d1 = Dpad
        h0, h1 = Hpad
        w0, w1 = Wpad
        if d1 == 0:
            d_slice = slice(d0, None)
        else:
            d_slice = slice(d0, -d1)
        if h1 == 0:
            h_slice = slice(h0, None)
        else:
            h_slice = slice(h0, -h1)
        if w1 == 0:
            w_slice = slice(w0, None)
        else:
            w_slice = slice(w0, -w1)
        return x[:, d_slice, h_slice, w_slice]
    return x


@torch.no_grad()
def sliding_window_inference(model: nn.Module, volume: np.ndarray, patch_size: Tuple[int, int, int], device: torch.device):
    # volume: (C,D,H,W) numpy
    model.eval()
    C, D, H, W = volume.shape
    pd, ph, pw = patch_size

    stride_d = pd // 2
    stride_h = ph // 2
    stride_w = pw // 2

    # prepare accumulation arrays
    probs_sum = np.zeros((4, D, H, W), dtype=np.float32)
    counts = np.zeros((D, H, W), dtype=np.float32)

    # pad if necessary
    vol_p, pad = pad_to_shape(volume, patch_size)
    _, Dp, Hp, Wp = vol_p.shape

    # iterate windows
    ds = list(range(0, Dp - pd + 1, stride_d))
    hs = list(range(0, Hp - ph + 1, stride_h))
    ws = list(range(0, Wp - pw + 1, stride_w))

    # ensure last window touches end
    if len(ds) == 0 or ds[-1] != Dp - pd:
        ds.append(Dp - pd)
    if len(hs) == 0 or hs[-1] != Hp - ph:
        hs.append(Hp - ph)
    if len(ws) == 0 or ws[-1] != Wp - pw:
        ws.append(Wp - pw)

    for d in tqdm(ds, desc='sw-d'):
        for h in hs:
            for w in ws:
                patch = vol_p[:, d:d + pd, h:h + ph, w:w + pw]
                inp = torch.from_numpy(patch[None, ...]).float().to(device)  # shape (1,C,D,H,W)
                # model expects (B,C,D,H,W)
                inp = inp
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out = model(inp)  # (1, num_classes, pd, ph, pw)
                    out_prob = torch.softmax(out, dim=1).cpu().numpy()[0]  # shape (C,pd,ph,pw)
                # accumulate
                probs_sum[:, d:d + pd, h:h + ph, w:w + pw] += out_prob
                counts[d:d + pd, h:h + ph, w:w + pw] += 1.0

    # avoid division by zero
    counts[counts == 0] = 1.0
    probs_averaged = probs_sum / counts[None, :, :, :]

    # unpad
    if any([x > 0 for x in pad[1]]):
        # pad returned as ((0,0),(d0,d1),(h0,h1),(w0,w1))
        # convert to remove outer dims
        d0, d1 = pad[1]
        h0, h1 = pad[2]
        w0, w1 = pad[3]
        if d1 == 0:
            probs_averaged = probs_averaged[:, d0:, h0:, w0:]
        else:
            probs_averaged = probs_averaged[:, d0:-d1, h0:-h1, w0:-w1]

    # final prediction
    pred = np.argmax(probs_averaged, axis=0).astype(np.uint8)  # (D,H,W) values 0..3
    return probs_averaged, pred


def map_pred_back_labels(pred: np.ndarray) -> np.ndarray:
    # pred values 0..3 -> map to original BraTS labels 0,1,2,4
    out = np.zeros_like(pred, dtype=np.uint8)
    reverse_map = {v: k for k, v in LABEL_MAP.items()}  # {0:0,1:1,2:2,3:4}
    for k, v in reverse_map.items():
        out[pred == k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='nnUNet_raw/Dataset501_BraTS2021')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--case', type=str, required=True, help='case id like BraTS2021_00000 (without modality suffix)')
    parser.add_argument('--patch-size', type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument('--output', type=str, default=None, help='path for output NIfTI file')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        try:
            print('GPU:', torch.cuda.get_device_name(0))
        except Exception:
            pass
    else:
        print('Using CPU for inference (very slow)')

    # prepare output folder
    ensure_dir('predictions')

    # load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')

    state = torch.load(args.model_path, map_location=device)
    model = UNet3D(in_channels=4, base_features=32, num_classes=4)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.to(device)
    model.eval()

    # load image volumes
    case_id = args.case
    x, affine, header = load_case_volumes(args.data_root, case_id)
    x_norm = normalize_volume(x)

    # sliding-window inference: use MONAI if available which is optimized and supports batching
    if _have_monai_infer:
        inp = torch.from_numpy(x_norm[None, ...]).float().to(device)  # (1,C,D,H,W)

        def _model_forward(inp_tensor):
            out = model(inp_tensor)
            if isinstance(out, tuple):
                out = out[0]
            return out

        with torch.no_grad():
            pred_tensor = monai_sw_inference(inp, tuple(args.patch_size), sw_batch_size=1, predictor=_model_forward, overlap=0.5)
            probs = torch.softmax(pred_tensor.cpu().numpy()[0], axis=0)
            pred = np.argmax(probs, axis=0).astype(np.uint8)
    else:
        # fallback to local sliding window implementation
        probs, pred = sliding_window_inference(model, x_norm, tuple(args.patch_size), device)

    # map back to labels 0,1,2,4 and transpose back to H,W,D
    pred_labels = map_pred_back_labels(pred)  # (D,H,W)
    pred_labels_hwz = np.transpose(pred_labels, (1, 2, 0))

    out_path = args.output if args.output is not None else os.path.join('predictions', f'{case_id}_pred.nii.gz')
    out_img = nib.Nifti1Image(pred_labels_hwz.astype(np.uint8), affine, header)
    nib.save(out_img, out_path)
    print('Saved prediction to', out_path)


if __name__ == '__main__':
    main()
