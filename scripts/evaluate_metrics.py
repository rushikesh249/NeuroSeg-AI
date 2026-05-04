import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import BraTSPatchDataset
from unet3d import UNet3D
from losses import CombinedLoss

def compute_detailed_metrics(probs: torch.Tensor, target: torch.Tensor):
    """
    Computes Dice, Precision, Recall, and Accuracy per class.
    probs: (B, C, D, H, W)
    target: (B, D, H, W)
    """
    with torch.no_grad():
        C = probs.shape[1]
        pred = probs.argmax(dim=1)  # (B,D,H,W)
        
        metrics = {
            'dice': [],
            'precision': [],
            'recall': [],
            'accuracy': []
        }
        
        for c in range(C):
            pred_c = (pred == c).float()
            tgt_c = (target == c).float()
            
            tp = (pred_c * tgt_c).sum()
            fp = (pred_c * (1 - tgt_c)).sum()
            fn = ((1 - pred_c) * tgt_c).sum()
            tn = ((1 - pred_c) * (1 - tgt_c)).sum()
            
            # Dice
            dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
            metrics['dice'].append(dice.item())
            
            # Precision
            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            metrics['precision'].append(precision.item())
            
            # Recall (Sensitivity)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)
            metrics['recall'].append(recall.item())
            
            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            metrics['accuracy'].append(accuracy.item())
            
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D U-Net metrics')
    parser.add_argument('--data-root', type=str, default='nnUNet_raw/Dataset501_BraTS2021', help='Path to dataset')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--patch-size', type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    ds = BraTSPatchDataset(args.data_root, split='train', patch_size=tuple(args.patch_size))
    
    # Re-create validation split (consistent with train.py)
    n = len(ds)
    indices = list(range(n))
    import random as _rnd
    _rnd.seed(args.seed)
    _rnd.shuffle(indices)
    split_idx = int(n * 0.8)
    val_idx = indices[split_idx:]
    val_ds = Subset(ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Load model
    model = UNet3D(in_channels=4, base_features=32, num_classes=4)
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint: {args.model_path}")
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)
    else:
        print(f"Error: {args.model_path} not found.")
        return

    model = model.to(device)
    model.eval()

    all_metrics = {k: [] for k in ['dice', 'precision', 'recall', 'accuracy']}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            
            m = compute_detailed_metrics(probs, labels)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    # Aggregate
    print("\n" + "="*50)
    print(f"{'Class':<10} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'Accuracy':<8}")
    print("-" * 50)
    
    class_names = ["Background", "ET", "NET", "ED"]
    for i in range(4):
        d = np.mean([x[i] for x in all_metrics['dice']])
        p = np.mean([x[i] for x in all_metrics['precision']])
        r = np.mean([x[i] for x in all_metrics['recall']])
        a = np.mean([x[i] for x in all_metrics['accuracy']])
        print(f"{class_names[i]:<10} | {d:.4f}   | {p:.4f}      | {r:.4f}   | {a:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
