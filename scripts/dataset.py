import os
import json
import random
from typing import Tuple, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}  # maps nifti labels -> contiguous class indices 0..3
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def safe_load_nifti(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    return arr, img.affine, img.header


class BraTSPatchDataset(Dataset):
    """
    Patch-based loader for BraTS-style nnUNet_raw structure.

    Expected layout (root_dir argument):
      <root_dir>/imagesTr/  -> <case>_0000.nii.gz, <case>_0001.nii.gz, <case>_0002.nii.gz, <case>_0003.nii.gz
      <root_dir>/labelsTr/  -> <case>.nii.gz
    
    Returns patches in shape: (C=4, D, H, W) as float32 and label (D,H,W) as int64 (class indices 0..3)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        tumor_sample_prob: float = 0.7,
        seed: Optional[int] = None,
        modality_order: Optional[List[str]] = None,
    ) -> None:
        assert split in ("train", "val", "test"), "split must be 'train'|'val'|'test'"
        self.root_dir = root_dir
        self.split = split
        self.patch_size = tuple(patch_size)
        self.tumor_sample_prob = float(tumor_sample_prob)
        if seed is not None:
            random.seed(seed)

        # Primary expected layout: nnUNet_raw style (imagesTr/labelsTr)
        self.images_dir = os.path.join(root_dir, "imagesTr")
        self.labels_dir = os.path.join(root_dir, "labelsTr")

        self.layout = None
        # nnUNet_raw layout
        if os.path.isdir(self.images_dir) and os.path.isdir(self.labels_dir):
            self.layout = 'nnunet'

        # alternative (original) BraTS layout: <root_dir>/BraTS2021_Training_Data/<case>/*_flair.nii.gz etc
        # or root_dir itself is BraTS2021_Training_Data containing patient subfolders
        bra_ts_dir_candidate = None
        if self.layout is None:
            # if given root itself contains patient subfolders (BraTS non-nnunet layout)
            try:
                child_list = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            except Exception:
                child_list = []
            # detect a subfolder starting with 'BraTS' or files like '*_flair.nii.gz' inside
            if any(name.startswith('BraTS') for name in child_list):
                # If the root contains many 'BraTS...' subfolders, treat root as the collection
                brats_subs = [name for name in child_list if name.startswith('BraTS')]
                if len(brats_subs) > 1:
                    bra_ts_dir_candidate = root_dir
                else:
                    # prefer top-level 'BraTS2021_Training_Data' if present or single subfolder
                    bra_ts_dir_candidate = next((name for name in child_list if name.startswith('BraTS')), None)
                    if bra_ts_dir_candidate:
                        bra_ts_dir_candidate = os.path.join(root_dir, bra_ts_dir_candidate)
            else:
                # root_dir might itself be the patient collection
                # check if root contains *_flair.nii.gz entries or folders with flair files
                has_patient_folders = False
                for f in child_list:
                    pth = os.path.join(root_dir, f)
                    # check for flair file inside this folder
                    if os.path.isdir(pth) and any(fn.endswith('_flair.nii.gz') for fn in os.listdir(pth)):
                        has_patient_folders = True
                        break
                if has_patient_folders:
                    bra_ts_dir_candidate = root_dir

        if bra_ts_dir_candidate is not None and os.path.isdir(bra_ts_dir_candidate):
            self.layout = 'brats_folders'
            self.brats_root = bra_ts_dir_candidate

        if self.layout is None:
            # Check for "flat" layout: all modalities in root_dir with suffixes _flair, _t1, etc.
            files = os.listdir(root_dir)
            has_any_flair = any('_flair.nii' in f for f in files)
            if has_any_flair:
                self.layout = 'flat_brats'
                self.flat_root = root_dir

        if self.layout is None:
            # if neither layout found -> raise
            raise FileNotFoundError(
                f"Could not find dataset layout under {root_dir}. Expected nnUNet_raw pattern (imagesTr/labelsTr), BraTS patient folders, or flat NIfTI files with suffixes."
            )

        # discover cases
        self.cases = []
        if self.layout == 'nnunet':
            for f in os.listdir(self.labels_dir) if self.labels_dir else []:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    case_id = f.replace('.nii.gz', '').replace('.nii', '')
                    # verify required 4 modalities exist for this case
                    imgs = [os.path.join(self.images_dir, f"{case_id}_000{i}.nii.gz") for i in range(4)]
                    if all(os.path.exists(p) for p in imgs):
                        self.cases.append(case_id)
                    else:
                        # skip incomplete case
                        continue
        elif self.layout == 'brats_folders':
            # brats_folders layout: each subfolder contains files like <case>_flair.nii.gz, *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_seg.nii.gz
            for d in os.listdir(self.brats_root):
                dpath = os.path.join(self.brats_root, d)
                if not os.path.isdir(dpath):
                    continue
                # collect expected names
                files = os.listdir(dpath)
                has_flair = any(fn.endswith('_flair.nii.gz') for fn in files)
                has_t1 = any(fn.endswith('_t1.nii.gz') for fn in files)
                has_t1ce = any(fn.endswith('_t1ce.nii.gz') for fn in files)
                has_t2 = any(fn.endswith('_t2.nii.gz') for fn in files)
                has_seg = any(fn.endswith('_seg.nii.gz') for fn in files)
                if has_flair and has_t1 and has_t1ce and has_t2 and (has_seg or self.split == 'test'):
                    # case id is folder name (e.g., BraTS2021_00000)
                    self.cases.append(d)
        elif self.layout == 'flat_brats':
            # detect unique case IDs from _flair.nii files
            files = os.listdir(self.flat_root)
            for f in files:
                if '_flair.nii' in f:
                    # e.g., BraTS20_Training_001_flair.nii -> BraTS20_Training_001
                    case_id = f.split('_flair.nii')[0]
                    self.cases.append(case_id)

        if len(self.cases) == 0:
            raise RuntimeError("No cases found under labelsTr (or imagesTr). Are you pointing at nnUNet_raw/Dataset501_BraTS2021?")

        # load optional MONAI augmentations if available
        self.use_monai = False
        try:
            import monai
            from monai.transforms import Compose, RandFlipd, RandGaussianNoised, RandShiftIntensityd, RandScaleIntensityd, RandAdjustContrastd, RandAffined
            self.use_monai = True
            # create dict-style transforms that apply to both image and label keys
            self.train_transform = Compose([
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
                RandAffined(keys=['image', 'label'], prob=0.3, rotate_range=(0.1, 0.1, 0.1), translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1), mode=('bilinear', 'nearest')),
                RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.01),
                RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.15),
                RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.15),
                RandAdjustContrastd(keys=['image'], gamma=(0.9, 1.1), prob=0.15),
            ])
        except Exception:
            # no MONAI; will use numpy-based augmentations already implemented below
            self.use_monai = False

        # Load dataset.json if present for meta (not required)
        dataset_json = os.path.join(root_dir, 'dataset.json')
        self.meta = None
        if os.path.exists(dataset_json):
            with open(dataset_json, 'r') as f:
                self.meta = json.load(f)

    def __len__(self) -> int:
        # we'll present 1 patch per case per epoch; DataLoader should iterate many epochs
        # Sometimes want more patches than cases; you can set an epoch multiplier externally.
        return len(self.cases)

    def _load_case(self, case_id: str):
        # return stacked image channels and label for supported layouts
        imgs = []
        affine = None
        header = None
        if self.layout == 'nnunet':
            for i in range(4):
                p = os.path.join(self.images_dir, f"{case_id}_000{i}.nii.gz")
                arr, affine, header = safe_load_nifti(p)
                # nibabel returns HxWxD; we transpose to D x H x W for network
                arr = np.transpose(arr, (2, 0, 1))
                imgs.append(arr)

            x = np.stack(imgs, axis=0)  # shape (4, D, H, W)

            label_path = os.path.join(self.labels_dir, f"{case_id}.nii.gz")
            y, _, _ = safe_load_nifti(label_path)
            # label returned as HxWxD -> transpose to D x H x W
            y = np.transpose(y.astype(np.int16), (2, 0, 1))

        elif self.layout == 'brats_folders':
            case_folder = os.path.join(self.brats_root, case_id)
            if not os.path.isdir(case_folder):
                raise FileNotFoundError(f"Case folder not found: {case_folder}")
            # expected files: <case>_flair.nii.gz, etc.
            modal_names = [f"{case_id}_flair.nii.gz", f"{case_id}_t1.nii.gz", f"{case_id}_t1ce.nii.gz", f"{case_id}_t2.nii.gz"]
            for fn in modal_names:
                p = os.path.join(case_folder, fn)
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing modality file: {p}")
                arr, affine, header = safe_load_nifti(p)
                arr = np.transpose(arr, (2, 0, 1))
                imgs.append(arr)
            x = np.stack(imgs, axis=0)
            seg_name = f"{case_id}_seg.nii.gz"
            seg_path = os.path.join(case_folder, seg_name)
            if not os.path.exists(seg_path):
                if self.split == 'test':
                    D, H, W = x.shape[1:]
                    y = np.zeros((D, H, W), dtype=np.int16)
                else:
                    raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
            else:
                y, _, _ = safe_load_nifti(seg_path)
                y = np.transpose(y.astype(np.int16), (2, 0, 1))

        elif self.layout == 'flat_brats':
            # filenames: <case_id>_flair.nii, etc.
            modal_suffixes = ["_flair", "_t1", "_t1ce", "_t2"]
            files_in_root = os.listdir(self.flat_root)
            for s in modal_suffixes:
                match = next((f for f in files_in_root if f.startswith(case_id) and s in f), None)
                if not match:
                    raise FileNotFoundError(f"Missing modality {s} for case {case_id}")
                p = os.path.join(self.flat_root, match)
                arr, affine, header = safe_load_nifti(p)
                arr = np.transpose(arr, (2, 0, 1))
                imgs.append(arr)
            x = np.stack(imgs, axis=0)
            seg_match = next((f for f in files_in_root if f.startswith(case_id) and '_seg.nii' in f), None)
            if not seg_match:
                if self.split == 'test':
                    D, H, W = x.shape[1:]
                    y = np.zeros((D, H, W), dtype=np.int16)
                else:
                    raise FileNotFoundError(f"Missing segmentation for case {case_id}")
            else:
                y, _, _ = safe_load_nifti(os.path.join(self.flat_root, seg_match))
                y = np.transpose(y.astype(np.int16), (2, 0, 1))
        else:
            raise RuntimeError(f"Unknown dataset layout: {self.layout}")

        # map 0,1,2,4 -> 0..3
        y_mapped = np.vectorize(LABEL_MAP.get)(y)

        return x, y_mapped.astype(np.int64), affine, header

    def _pad_if_needed(self, arr: np.ndarray, desired: Tuple[int, int, int], pad_value: float = 0.0):
        # arr shape: (C?, D, H, W) or (D,H,W) - support both
        if arr.ndim == 4:
            _, D, H, W = arr.shape
        else:
            D, H, W = arr.shape
        pd = max(desired[0] - D, 0)
        ph = max(desired[1] - H, 0)
        pw = max(desired[2] - W, 0)
        if pd == ph == pw == 0:
            return arr, (0, 0, 0)
        # compute pad widths for np.pad
        pad_width = ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)) if arr.ndim == 4 else ((pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
        arr_p = np.pad(arr, pad_width, constant_values=pad_value)
        return arr_p, (pad_width if arr.ndim == 4 else pad_width)

    def _random_crop(self, img: np.ndarray, label: np.ndarray, size: Tuple[int, int, int]):
        # img shape (C, D, H, W), label shape (D, H, W)
        C, D, H, W = img.shape
        pd, ph, pw = size

        if D <= pd and H <= ph and W <= pw:
            # pad to fit
            img_p, _ = self._pad_if_needed(img, size)
            label_p, _ = self._pad_if_needed(label, size)
            return img_p, label_p

        # choose starting indices
        max_d = max(D - pd, 0)
        max_h = max(H - ph, 0)
        max_w = max(W - pw, 0)
        sd = random.randint(0, max_d)
        sh = random.randint(0, max_h)
        sw = random.randint(0, max_w)
        img_crop = img[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
        label_crop = label[sd:sd + pd, sh:sh + ph, sw:sw + pw]
        return img_crop, label_crop

    def _tumor_centered_crop(self, img: np.ndarray, label: np.ndarray, size: Tuple[int, int, int]):
        # find nonzero indices in label
        nz = np.argwhere(label > 0)
        if nz.size == 0:
            return self._random_crop(img, label, size)
        # pick a random tumor voxel as center
        idx = random.choice(nz)
        cd, ch, cw = idx
        pd, ph, pw = size

        # compute start positions
        sd = cd - pd // 2
        sh = ch - ph // 2
        sw = cw - pw // 2

        D, H, W = label.shape
        # clamp
        sd = max(0, min(sd, D - pd))
        sh = max(0, min(sh, H - ph))
        sw = max(0, min(sw, W - pw))

        img_crop = img[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
        label_crop = label[sd:sd + pd, sh:sh + ph, sw:sw + pw]

        # if any dimension is undersized because image smaller than patch we pad (handled below)
        if img_crop.shape[1:] != tuple(size):
            # pad
            img_p, _ = self._pad_if_needed(img_crop, size)
            label_p, _ = self._pad_if_needed(label_crop, size)
            return img_p, label_p
        return img_crop, label_crop

    def _center_crop(self, img: np.ndarray, label: np.ndarray, size: Tuple[int, int, int]):
        C, D, H, W = img.shape
        pd, ph, pw = size
        sd = max(0, (D - pd) // 2)
        sh = max(0, (H - ph) // 2)
        sw = max(0, (W - pw) // 2)
        img_crop = img[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
        label_crop = label[sd:sd + pd, sh:sh + ph, sw:sw + pw]
        if img_crop.shape[1:] != tuple(size):
            img_p, _ = self._pad_if_needed(img_crop, size)
            label_p, _ = self._pad_if_needed(label_crop, size)
            return img_p, label_p
        return img_crop, label_crop

    def _normalize(self, img: np.ndarray):
        # img shape (C,D,H,W) float32
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[0]):
            v = img[c]
            m = v.mean()
            s = v.std()
            if s < 1e-6:
                out[c] = v - m
            else:
                out[c] = (v - m) / s
        return out

    def __getitem__(self, idx: int):
        case_id = self.cases[idx]
        x, y, affine, header = self._load_case(case_id)

        # optionally pad
        x, _ = self._pad_if_needed(x, self.patch_size, pad_value=0.0)
        y, _ = self._pad_if_needed(y, self.patch_size, pad_value=0)

        if self.split == 'train':
            if random.random() < self.tumor_sample_prob:
                x_patch, y_patch = self._tumor_centered_crop(x, y, self.patch_size)
            else:
                x_patch, y_patch = self._random_crop(x, y, self.patch_size)

            # Apply MONAI dict-style augmentations if available
            if self.use_monai:
                sample = {'image': x_patch.copy(), 'label': y_patch.copy()}
                sample = self.train_transform(sample)
                x_patch = sample['image']
                y_patch = sample['label']
            else:
                # numpy-based augmentations fallback (flips)
                if random.random() < 0.5:
                    x_patch = np.flip(x_patch, axis=1).copy()
                    y_patch = np.flip(y_patch, axis=0).copy()
                if random.random() < 0.5:
                    x_patch = np.flip(x_patch, axis=2).copy()
                    y_patch = np.flip(y_patch, axis=1).copy()
                if random.random() < 0.5:
                    x_patch = np.flip(x_patch, axis=3).copy()
                    y_patch = np.flip(y_patch, axis=2).copy()

        elif self.split == 'val':
            # deterministic center crop
            x_patch, y_patch = self._center_crop(x, y, self.patch_size)
        else:
            x_patch, y_patch = x, y

        # normalize per-channel
        x_patch = self._normalize(x_patch)

        # to torch tensors
        x_t = torch.from_numpy(x_patch.copy()).float()
        y_t = torch.from_numpy(y_patch.copy()).long()

        # avoid returning nibabel header objects via DataLoader collate on training/validation
        item = {
            'image': x_t,  # (C, D, H, W)
            'label': y_t,   # (D, H, W) with class indices 0..3
            'case_id': case_id,
        }
        # only return affine/header for test split (not collated during training)
        if self.split == 'test':
            item['affine'] = affine
            item['header'] = header
        return item


if __name__ == '__main__':
    # quick smoke test using the workspace dataset (if present)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='nnUNet_raw/Dataset501_BraTS2021')
    args = parser.parse_args()

    ds = BraTSPatchDataset(args.root, split='train', patch_size=(128, 128, 128))
    sample = ds[0]
    print('image', sample['image'].shape, sample['image'].dtype)
    print('label', sample['label'].shape, np.unique(sample['label'].numpy()))
