import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import sys
from PIL import Image
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.model.autoencoder import Autoencoder

def euler_to_rotation_matrix(euler_angles):
    """Convert Euler angles to rotation matrix."""
    # Extract angles
    x, y, z = euler_angles
    
    # Calculate rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations
    R = Rz @ Ry @ Rx
    return R

def visualize_prediction(input_img, output_img, true_rot, pred_rot, epoch, batch_idx, output_dir):
    """Visualize input, output, and rotation vectors."""
    # Convert tensors to numpy arrays
    input_img = input_img.cpu().numpy().transpose(1, 2, 0)
    output_img = output_img.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = std * input_img + mean
    output_img = std * output_img + mean
    
    # Clip to valid range
    input_img = np.clip(input_img, 0, 1)
    output_img = np.clip(output_img, 0, 1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot images
    ax1.imshow(input_img)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    ax2.imshow(output_img)
    ax2.set_title('Reconstructed Image')
    ax2.axis('off')
    
    # Add rotation vectors
    def draw_rotation_vector(ax, rot, color, label):
        # Convert Euler angles to rotation matrix
        rot_matrix = euler_to_rotation_matrix(rot)
        
        # Get transformed Z-axis
        z_axis = np.array([0, 0, 1])
        transformed_z = rot_matrix @ z_axis
        
        # Draw vector
        ax.arrow(128, 128,  # Start at center
                 transformed_z[0] * 50, transformed_z[1] * 50,  # Direction
                 head_width=5, head_length=10, fc=color, ec=color,
                 label=label)
    
    # Draw true rotation vector on input image
    draw_rotation_vector(ax1, true_rot, 'red', 'True Rotation')
    
    # Draw predicted rotation vector on output image
    draw_rotation_vector(ax2, pred_rot, 'green', 'Predicted Rotation')
    
    # Add legends
    ax1.legend()
    ax2.legend()
    
    # Add title with epoch and batch info
    plt.suptitle(f'Epoch {epoch}, Batch {batch_idx}')
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, f'vis_epoch_{epoch}_batch_{batch_idx}.png')
    plt.savefig(vis_path)
    
    # Show plot and wait for user to close
    plt.show()
    plt.close()

class DartPoseDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = Path(data_dir)
        self.example_dirs = sorted([d for d in os.listdir(data_dir) 
                                  if os.path.isdir(os.path.join(data_dir, d))])
        self.image_size = image_size
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.example_dirs)
    
    def __getitem__(self, idx):
        ex_dir = self.example_dirs[idx]
        ex_path = self.data_dir / ex_dir
        
        # Load image
        img_path = ex_path / 'image.png'
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load rotation data
        rot_path = ex_path / 'rotation.txt'
        rotation = torch.tensor(np.loadtxt(rot_path), dtype=torch.float32)
        
        return image, rotation

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, latent = model(data)
        loss = criterion(output, data)  # Reconstruction loss
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Print batch progress
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, data)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    
    return total_loss / total_samples

def main():
    parser = argparse.ArgumentParser(description='Train autoencoder model')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = DartPoseDataset(args.data_dir, args.image_size)
    
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with persistent workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    model = Autoencoder(args.latent_dim).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * args.epochs
    
    # Create OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1e4  # Final lr = initial_lr/1e4
    )
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Initialize loss history
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    print("=" * 50)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch stats
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        print(f'Time: {epoch_time:.2f}s')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Saved best model to {best_model_path}')
    
    total_time = time.time() - start_time
    
    # Print final training summary
    print("\nTraining Summary:")
    print("=" * 50)
    print(f'Total training time: {total_time/60:.1f} minutes')
    print(f'Best validation loss: {best_val_loss:.6f}')
    print("\nLoss History:")
    print("=" * 50)
    print("Epoch | Train Loss | Val Loss")
    print("-" * 50)
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        print(f"{epoch+1:5d} | {train_loss:10.6f} | {val_loss:9.6f}")
    
    # Save loss history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time
    }
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    main() 