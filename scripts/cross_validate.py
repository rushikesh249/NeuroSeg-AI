import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset import BraTSPatchDataset
from unet3d import UNet3D

def compute_metrics_from_cm(cm):
    """Calculate metrics from confusion matrix."""
    num_classes = cm.shape[0]
    total_voxel_count = cm.sum()
    class_metrics = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_voxel_count - (tp + fp + fn)
        
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
        
        class_metrics.append({
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'dice': float(dice)
        })
    return class_metrics

def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Evaluation')
    parser.add_argument('--data-root', type=str, default='sample_data')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--patch-size', type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/cv')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    try:
        ds = BraTSPatchDataset(args.data_root, split='train', patch_size=tuple(args.patch_size))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    n = len(ds)
    if n < args.folds:
        print(f"Warning: Only {n} cases found. Reducing folds to {n}.")
        args.folds = max(1, n)

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

    indices = np.arange(n)
    cv_results = []
    class_names = ["Background", "ET", "NET", "ED"]
    aggregated_cm = np.zeros((4, 4), dtype=np.int64)

    # KFold requires at least 2 folds. If 1, we do it manually.
    if args.folds > 1:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_iterator = kf.split(indices)
    else:
        # Single fold manual (test on the only case we have)
        fold_iterator = [(np.array([]), indices)]

    print(f"Starting {args.folds}-fold cross-validation...")

    for fold, (train_idx, test_idx) in enumerate(fold_iterator):
        print(f"\nFold {fold + 1}/{args.folds} (Testing on {len(test_idx)} cases)")
        
        test_ds = Subset(ds, test_idx)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        
        fold_cm = np.zeros((4, 4), dtype=np.int64)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold+1}"):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = torch.softmax(outputs, dim=1)
                pred = probs.argmax(dim=1).view(-1).cpu().numpy()
                target = labels.view(-1).cpu().numpy()
                
                cm = confusion_matrix(target, pred, labels=[0, 1, 2, 3])
                fold_cm += cm
                aggregated_cm += cm
        
        fold_metrics = compute_metrics_from_cm(fold_cm)
        cv_results.append(fold_metrics)

    # Aggregate and report
    print("\n" + "="*60)
    print(f"{'Class':<12} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'Acc':<8}")
    print("-" * 60)

    report_path = os.path.join(args.output_dir, 'cv_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# {args.folds}-Fold Cross-Validation Report\n\n")
        f.write("| Class | Dice (Mean ± Std) | Precision | Recall | Accuracy |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for c in range(4):
            dices = [res[c]['dice'] for res in cv_results]
            precs = [res[c]['precision'] for res in cv_results]
            recs  = [res[c]['recall'] for res in cv_results]
            accs  = [res[c]['accuracy'] for res in cv_results]
            
            d_m, d_s = np.mean(dices), np.std(dices)
            p_m, r_m, a_m = np.mean(precs), np.mean(recs), np.mean(accs)
            
            print(f"{class_names[c]:<12} | {d_m:.4f}   | {p_m:.4f}      | {r_m:.4f}   | {a_m:.4f}")
            f.write(f"| {class_names[c]} | {d_m:.4f} ± {d_s:.4f} | {p_m:.4f} | {r_m:.4f} | {a_m:.4f} |\n")

    # Plot aggregated heatmaps
    # 1. Raw Counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(aggregated_cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Aggregated Confusion Matrix (Counts, {args.folds}-Fold CV)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'cv_heatmap.png'), dpi=300)
    plt.close()

    # 2. Normalized
    cm_norm = aggregated_cm.astype('float') / (aggregated_cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual (True Class)')
    plt.title(f'Aggregated Normalized Confusion Matrix ({args.folds}-Fold CV)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'cv_heatmap_norm.png'), dpi=300)
    plt.close()

    print("="*60)
    print("Cross-validation complete. Results saved to {args.output_dir}/")

if __name__ == '__main__':
    main()
