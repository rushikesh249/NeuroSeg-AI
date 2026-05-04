import torch
import numpy as np
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.getcwd())

from unet3d import UNet3D, count_parameters

def main():
    checkpoint_path = 'models/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return

    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Model parameters
    model = UNet3D(in_channels=4, base_features=32, num_classes=4)
    params = count_parameters(model)
    
    # Extract metrics from checkpoint
    epoch = checkpoint.get('epoch', 'N/A')
    best_val = checkpoint.get('best_val', 'N/A')
    
    print(f"--- Model Metrics ---")
    print(f"Architecture: 3D UNet with Residual Blocks")
    print(f"Total Parameters: {params:,}")
    print(f"Best Validation Mean Dice (Classes 1,2,3): {best_val}")
    print(f"Training Epochs completed: {epoch + 1 if isinstance(epoch, int) else epoch}")
    
    # If we had more detailed class-wise metrics stored, we'd print them here.
    # From train.py, it seems only best_val (mean of 1,2,3) is explicitly saved as a scalar.

if __name__ == "__main__":
    main()
