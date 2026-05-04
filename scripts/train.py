import os
import sys
import argparse
import math
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import BraTSPatchDataset
from unet3d import UNet3D, count_parameters
from losses import CombinedLoss


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def dice_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, smooth=1e-6) -> torch.Tensor:
    # probs: (B, C, D, H, W), target: (B, D, H, W) with labels 0..C-1
    C = probs.shape[1]
    target_onehot = torch.nn.functional.one_hot(target.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()

    # compute dice per class
    dims = (2, 3, 4)
    intersect = (probs * target_onehot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice_per_class = (2.0 * intersect + smooth) / (cardinality + smooth)
    # dice loss: 1 - mean dice across classes
    return 1.0 - dice_per_class.mean()


def compute_metrics(probs: torch.Tensor, target: torch.Tensor):
    # returns Dice per class (numpy)
    # probs: (B, C, D, H, W), target: (B, D, H, W)
    with torch.no_grad():
        C = probs.shape[1]
        pred = probs.argmax(dim=1)  # (B,D,H,W)
        dices = []
        for c in range(C):
            pred_c = (pred == c).float()
            tgt_c = (target == c).float()
            intersect = (pred_c * tgt_c).sum(dim=(1, 2, 3))
            union = pred_c.sum(dim=(1, 2, 3)) + tgt_c.sum(dim=(1, 2, 3))
            dice = (2.0 * intersect + 1e-6) / (union + 1e-6)
            dices.append(dice.cpu().numpy())
        # dices: C lists arrays shape (B,)
        dices = np.stack(dices, axis=1)  # (B, C)
        mean_dices = np.mean(dices, axis=0)
        return mean_dices, dices


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    loss_obj: nn.Module,
    epoch: int,
    grad_clip: float,
    deep_supervision: bool = False,
):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train epoch {epoch}")
    for i, batch in pbar:
        images = batch['image'].to(device)  # (B,C,D,H,W)
        labels = batch['label'].to(device)  # (B,D,H,W)

        optimizer.zero_grad()
        # use new torch.amp.autocast API and deep supervision aware loss
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(images)
            # if deep supervision, model returns (main_out, [ds1, ds2, ...])
            if deep_supervision and isinstance(outputs, tuple):
                main_out, ds_list = outputs
                loss_main = loss_obj(main_out, labels.long())
                loss_ds = 0.0
                for ds in ds_list:
                    loss_ds = loss_ds + loss_obj(ds, labels.long())
                loss_ds = loss_ds / max(1, len(ds_list))
                # weight the main output higher
                loss = 0.7 * loss_main + 0.3 * loss_ds
                probs = torch.softmax(main_out, dim=1)
            else:
                main_out = outputs
                probs = torch.softmax(main_out, dim=1)
                loss = loss_obj(main_out, labels.long())

        scaler.scale(loss).backward()
        # gradient clipping
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (i + 1)})

    avg_loss = running_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, deep_supervision: bool = False):
    model.eval()
    per_class_dices = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validate')
    for i, batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        outputs = model(images)
        if deep_supervision and isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.softmax(outputs, dim=1)
        mean_dices, _ = compute_metrics(probs, labels)
        per_class_dices.append(mean_dices)
    per_class_dices = np.stack(per_class_dices, axis=0)
    mean_per_class = np.mean(per_class_dices, axis=0)
    mean_overall = mean_per_class.mean()
    return mean_overall, mean_per_class, mean_per_class


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D U-Net for BraTS')
    parser.add_argument('--data-root', type=str, default='nnUNet_raw/Dataset501_BraTS2021')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--patch-size', type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--deep-supervision', action='store_true')
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.save_dir)
    ensure_dir('predictions')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            print('Using GPU:', torch.cuda.get_device_name(0))
        except Exception:
            pass
    else:
        print('CUDA not available: device CPU will be used (training will be slow)')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # dataset
    try:
        ds = BraTSPatchDataset(args.data_root, split='train', patch_size=tuple(args.patch_size))
    except Exception as e:
        print(f"Failed to open dataset at {args.data_root}: {e}")
        # try fallback to commonly named folder in workspace
        fallback = os.path.join(os.getcwd(), 'BraTS2021_Training_Data')
        if os.path.isdir(fallback):
            print(f"Falling back to detected dataset folder: {fallback}")
            args.data_root = fallback
            ds = BraTSPatchDataset(args.data_root, split='train', patch_size=tuple(args.patch_size))
        else:
            raise

    # split 80/20
    n = len(ds)
    indices = list(range(n))
    # shuffle indices for random 80/20 split (repeatable with seed)
    import random as _rnd
    _rnd.seed(args.seed)
    _rnd.shuffle(indices)
    split_idx = int(n * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    model = UNet3D(in_channels=4, base_features=32, num_classes=4, dropout=0.2, deep_supervision=args.deep_supervision)
    print(f'Model total params: {count_parameters(model):,}')
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warm-up for first args.warmup_epochs then cosine annealing
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # combined loss (Generalized Dice + Focal)
    loss_obj = CombinedLoss(alpha=0.5, gamma=2.0)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    start_epoch = 0
    best_val = -1.0

    if args.resume is not None:
        if os.path.exists(args.resume):
            print('Resuming from', args.resume)
            state = torch.load(args.resume, map_location=device)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            start_epoch = state.get('epoch', 0) + 1
            best_val = state.get('best_val', -1.0)
        else:
            print(f'Warning: resume checkpoint {args.resume} not found, ignoring')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_obj, epoch, args.grad_clip, deep_supervision=args.deep_supervision)
        val_mean, val_per_class, val_per_class_all = validate(model, val_loader, device, deep_supervision=args.deep_supervision)
        scheduler.step()
        t1 = time.time()

        print(f'Epoch {epoch:3d}/{args.epochs} - train_loss: {loss:.4f}  val_mean_dice: {val_mean:.4f}  per_class: {val_per_class}')

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val': best_val,
        }

        # save last every epoch
        last_path = os.path.join(args.save_dir, 'last_model.pth')
        torch.save(ckpt, last_path)

        # save best
        # choose best by average tumor-class dice (classes indices 1,2,3)
        tumor_mean = float(np.mean(val_per_class_all[[1, 2, 3]]))
        if tumor_mean > best_val:
            best_val = tumor_mean
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            ckpt['best_val'] = best_val
            torch.save(ckpt, best_path)
            print('Saved new best model ->', best_path)

    print('Training finished. Best validation mean Dice:', best_val)


if __name__ == '__main__':
    main()
