import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from dataset import BraTSPatchDataset
from unet3d import UNet3D
from losses import CombinedLoss

def compute_all_metrics(probs: torch.Tensor, target: torch.Tensor, num_classes=4):
    """
    Computes comprehensive metrics: Accuracy, Precision, Recall, F1, Specificity, Dice, Jaccard.
    probs: (B, C, D, H, W)
    target: (B, D, H, W)
    """
    with torch.no_grad():
        pred = probs.argmax(dim=1)  # (B,D,H,W)
        
        # Flatten for pixel-wise metrics
        y_true = target.view(-1).cpu().numpy()
        y_pred = pred.view(-1).cpu().numpy()
        
        metrics = {}
        class_names = ["Background", "ET", "NET", "ED"]
        
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        
        class_metrics = []
        for i in range(num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)
            specificity = (tn + 1e-6) / (tn + fp + 1e-6)
            f1 = (2 * precision * recall) / (precision + recall + 1e-6)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
            jaccard = (tp + 1e-6) / (tp + fp + fn + 1e-6)
            
            class_metrics.append({
                'Class': class_names[i],
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'F1-Score': f1,
                'Accuracy': accuracy,
                'Dice': dice,
                'Jaccard': jaccard
            })
            
        return class_metrics, cm

def plot_confusion_matrix(cm, class_names, output_base_path):
    # 1. Raw counts heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Voxel-wise Confusion Matrix (Counts)')
    plt.tight_layout()
    plt.savefig(output_base_path + ".png", dpi=300)
    plt.close()

    # 2. Normalized heatmap (Accuracy per class)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual (True Class)')
    plt.title('Normalized Confusion Matrix (Accuracy %)')
    plt.tight_layout()
    plt.savefig(output_base_path + "_norm.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive 3D U-Net evaluation')
    parser.add_argument('--data-root', type=str, default='nnUNet_raw/Dataset501_BraTS2021')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth')
    parser.add_argument('--patch-size', type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Try to load dataset
    try:
        ds = BraTSPatchDataset(args.data_root, split='train', patch_size=tuple(args.patch_size))
    except Exception as e:
        print(f"Error loading dataset at {args.data_root}: {e}")
        print("Attempting fallback to 'sample_data'...")
        try:
            # Modify BraTSPatchDataset to handle sample_data if it's just a flat folder?
            # Actually, BraTSPatchDataset expects subfolders or imagesTr/labelsTr.
            # I'll try to find any folder that contains imagesTr
            ds = BraTSPatchDataset('sample_data', split='train', patch_size=tuple(args.patch_size))
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return

    # Split 80/20 to get validation set (same as train.py)
    n = len(ds)
    indices = list(range(n))
    import random as _rnd
    _rnd.seed(args.seed)
    _rnd.shuffle(indices)
    
    # Robust split: if n is very small, use all for evaluation
    if n > 1:
        split_idx = int(n * 0.8)
        val_idx = indices[split_idx:]
        if len(val_idx) == 0:
            val_idx = [indices[-1]]
    else:
        val_idx = indices
        
    print(f"Dataset contains {n} cases. Evaluating on {len(val_idx)} cases.")
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

    total_cm = np.zeros((4, 4), dtype=np.int64)
    class_names = ["Background", "ET", "NET", "ED"]
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            
            _, cm = compute_all_metrics(probs, labels)
            total_cm += cm

    # Calculate final metrics from total confusion matrix
    final_metrics = []
    total_voxel_count = total_cm.sum()
    
    overall_accuracy = np.trace(total_cm) / total_voxel_count
    
    for i in range(4):
        tp = total_cm[i, i]
        fp = total_cm[:, i].sum() - tp
        fn = total_cm[i, :].sum() - tp
        tn = total_voxel_count - (tp + fp + fn)
        
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        specificity = (tn + 1e-6) / (tn + fp + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
        jaccard = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        
        final_metrics.append({
            'Class': class_names[i],
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1-Score': f1,
            'Accuracy': accuracy,
            'Dice': dice,
            'Jaccard': jaccard
        })

    # Save metrics report
    report_path = os.path.join(args.output_dir, 'metrics_report.md')
    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Overall Voxel-wise Accuracy:** {overall_accuracy:.4f}\n\n")
        f.write("## Class-wise Metrics\n\n")
        f.write("| Class | Precision | Recall | Specificity | F1-Score | Accuracy | Dice | Jaccard |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for m in final_metrics:
            f.write(f"| {m['Class']} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['Specificity']:.4f} | {m['F1-Score']:.4f} | {m['Accuracy']:.4f} | {m['Dice']:.4f} | {m['Jaccard']:.4f} |\n")
        
        f.write("\n## Confusion Matrix\n")
        f.write("![Confusion Matrix Heatmap](heatmap.png)\n")

    # Plot and save heatmaps (Raw and Normalized)
    plot_confusion_matrix(total_cm, class_names, os.path.join(args.output_dir, 'heatmap'))
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}/")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

if __name__ == '__main__':
    main()
