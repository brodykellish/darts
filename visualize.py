import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import DartDataset, PosePredictor, get_device
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"visualization_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def quat2rotmat(q):
    # Convert quaternion to rotation matrix
    q = q / torch.norm(q, dim=-1, keepdim=True)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.stack([
        1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
    ], dim=-1).reshape(-1, 3, 3)
    return R

def plot_loss_curves(log_dir):
    """Plot training and validation loss curves from log files."""
    log_files = sorted([f for f in os.listdir(log_dir) if f.startswith('training_')])
    if not log_files:
        print("No training log files found!")
        return
    
    # Read the most recent log file
    latest_log = os.path.join(log_dir, log_files[-1])
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(latest_log, 'r') as f:
        for line in f:
            if "Epoch" in line and "Loss:" in line:
                parts = line.split()
                epoch = int(parts[1].split('/')[0])
                loss = float(parts[3].rstrip(','))
                epochs.append(epoch)
                train_losses.append(loss)
            elif "Validation metrics at epoch" in line:
                parts = line.split()
                val_loss = float(parts[-1])
                val_losses.append(val_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    if val_losses:
        plt.plot(epochs[::5], val_losses, label='Validation Loss')  # Validation every 5 epochs
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/loss_curves.png')
    plt.close()

def visualize_predictions(model, dataloader, device, num_samples=5):
    """Visualize model predictions on test set."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for images, q_gt, t_gt in dataloader:
            images = images.to(device)
            q_gt = q_gt.to(device)
            t_gt = t_gt.to(device)
            
            # Get predictions
            x_recon, q_pred, t_pred = model(images)
            
            # Convert quaternions to rotation matrices
            R_gt = quat2rotmat(q_gt)
            R_pred = quat2rotmat(q_pred)
            
            # Store samples
            for i in range(min(num_samples, len(images))):
                samples.append({
                    'image': images[i].cpu(),
                    'recon': x_recon[i].cpu(),
                    'R_gt': R_gt[i].cpu(),
                    'R_pred': R_pred[i].cpu(),
                    't_gt': t_gt[i].cpu(),
                    't_pred': t_pred[i].cpu()
                })
            
            if len(samples) >= num_samples:
                break
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, sample in enumerate(samples):
        # Original image
        axes[i, 0].imshow(sample['image'].permute(1, 2, 0))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Reconstructed image
        axes[i, 1].imshow(sample['recon'].permute(1, 2, 0))
        axes[i, 1].set_title('Reconstructed Image')
        axes[i, 1].axis('off')
        
        # Pose visualization
        ax = axes[i, 2]
        ax.set_title('Pose Prediction')
        
        # Plot coordinate frames
        origin = np.zeros(3)
        scale = 0.5
        
        # Ground truth pose
        R_gt = sample['R_gt'].numpy()
        t_gt = sample['t_gt'].numpy()
        for j, color in enumerate(['r', 'g', 'b']):
            ax.quiver(origin[0], origin[1], origin[2],
                     R_gt[0, j]*scale, R_gt[1, j]*scale, R_gt[2, j]*scale,
                     color=color, label=f'GT {["X", "Y", "Z"][j]}')
        
        # Predicted pose
        R_pred = sample['R_pred'].numpy()
        t_pred = sample['t_pred'].numpy()
        for j, color in enumerate(['r', 'g', 'b']):
            ax.quiver(origin[0], origin[1], origin[2],
                     R_pred[0, j]*scale, R_pred[1, j]*scale, R_pred[2, j]*scale,
                     color=color, linestyle='--', label=f'Pred {["X", "Y", "Z"][j]}')
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/predictions.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions and training progress')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    device = get_device()
    
    # Load model
    model = PosePredictor().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from {args.model_path}")
    
    # Create dataset and dataloader
    dataset = DartDataset("./dart_dataset")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Plot loss curves
    logger.info("Plotting loss curves...")
    plot_loss_curves("logs")
    
    # Visualize predictions
    logger.info("Visualizing predictions...")
    visualize_predictions(model, dataloader, device)
    
    logger.info("Visualization complete! Check the 'plots' directory for results.")

if __name__ == "__main__":
    main() 