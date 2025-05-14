#!/usr/bin/env python3
"""
Training script for the ViTPosePredictor model.
This script:
- Loads a dataset of rendered images with rotation information
- Trains a ViTPosePredictor model to predict rotation from images
- Saves checkpoints and logs training/validation loss
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import random

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.model.vit_pose_predictor import ViTPosePredictor
from src.train.trainer import PoseTrainer
from src.train.rotation_losses import RotationLoss, CombinedRotationTranslationLoss


class ViTPoseDataset(Dataset):
    """Dataset for training the ViTPosePredictor model with enhanced augmentation."""

    def __init__(self, data_dir, image_size=224, augment=True):
        """
        Args:
            data_dir (str): Directory containing the dataset
            image_size (int): Size to resize images to (ViT models typically use 224x224)
            augment (bool): Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.example_dirs = sorted([d for d in os.listdir(data_dir)
                                  if os.path.isdir(os.path.join(data_dir, d))])
        self.image_size = image_size
        self.augment = augment

        # Base transforms that are always applied
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])  # ViT normalization
        ])

        # Augmentation transforms that are only applied during training
        self.augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

    def __len__(self):
        return len(self.example_dirs)

    def __getitem__(self, idx):
        ex_dir = self.example_dirs[idx]
        ex_path = self.data_dir / ex_dir

        # Load image
        img_path = ex_path / 'image.png'
        image = Image.open(img_path).convert('RGB')

        # Apply base transform
        image = self.base_transform(image)

        # Apply augmentation if enabled
        if self.augment:
            image = self.augmentation(image)

        # Try to load quaternion data first, fall back to Euler angles if not available
        quat_path = ex_path / 'quaternion.txt'
        if os.path.exists(quat_path):
            # Load quaternion (w, x, y, z)
            rotation = torch.tensor(np.loadtxt(quat_path), dtype=torch.float32)
        else:
            # Load Euler angles and convert to quaternion
            rot_path = ex_path / 'rotation.txt'
            euler_angles = np.loadtxt(rot_path)

            # Convert to quaternion using scipy
            from scipy.spatial.transform import Rotation
            rotation_obj = Rotation.from_euler('xyz', euler_angles)
            quat = rotation_obj.as_quat()  # Returns (x, y, z, w)

            # Reorder to (w, x, y, z) which is more common
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            rotation = torch.tensor(quat, dtype=torch.float32)

        # For now, we'll use a dummy translation (all zeros)
        # since our dataset only contains rotation information
        translation = torch.zeros(3, dtype=torch.float32)

        return {
            'image': image,
            'rotation': rotation,
            'translation': translation
        }


class ViTPoseTrainer:
    """Enhanced training pipeline for ViTPosePredictor with custom loss and LR scheduling."""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate=1e-4, weight_decay=1e-4, rotation_mode='euler'):
        self.model = model.to(device)
        self.device = device
        self.rotation_mode = rotation_mode

        # Custom loss functions
        self.combined_loss = CombinedRotationTranslationLoss(
            rotation_mode=rotation_mode,
            rotation_weight=1.0,
            translation_weight=0.1
        )

        # Optimizer with increased weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=30,  # Number of epochs before resetting the cycle
            eta_min=learning_rate / 100  # Minimum learning rate
        )

        # For logging
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_rot_loss = 0
        total_trans_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            # Move data to device
            images = batch['image'].to(self.device)
            target_rot = batch['rotation'].to(self.device)
            target_trans = batch['translation'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred_rot, pred_trans = self.model(images)

            # Compute combined loss
            loss, loss_components = self.combined_loss(pred_rot, pred_trans, target_rot, target_trans)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss_components['combined_loss']
            total_rot_loss += loss_components['rotation_loss']
            total_trans_loss += loss_components['translation_loss']

        # Update learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        avg_loss = total_loss / len(dataloader)
        avg_rot_loss = total_rot_loss / len(dataloader)
        avg_trans_loss = total_trans_loss / len(dataloader)

        return avg_loss, avg_rot_loss, avg_trans_loss

    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_rot_loss = 0
        total_trans_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                target_rot = batch['rotation'].to(self.device)
                target_trans = batch['translation'].to(self.device)

                # Forward pass
                pred_rot, pred_trans = self.model(images)

                # Compute combined loss
                loss, loss_components = self.combined_loss(pred_rot, pred_trans, target_rot, target_trans)

                total_loss += loss_components['combined_loss']
                total_rot_loss += loss_components['rotation_loss']
                total_trans_loss += loss_components['translation_loss']

        avg_loss = total_loss / len(dataloader)
        avg_rot_loss = total_rot_loss / len(dataloader)
        avg_trans_loss = total_trans_loss / len(dataloader)

        return avg_loss, avg_rot_loss, avg_trans_loss

    def train(self, train_loader, val_loader, num_epochs, save_dir, log_interval=1, patience=10):
        """
        Train the model for multiple epochs with early stopping and learning rate scheduling.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs to train for
            save_dir: Directory to save checkpoints and logs
            log_interval: Interval (in epochs) to save checkpoints and plots
            patience: Number of epochs to wait for validation improvement before early stopping
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Create log file
        log_file = os.path.join(save_dir, 'training_log.json')
        log_data = {
            'epochs': [],
            'train_loss': [],
            'train_rot_loss': [],
            'train_trans_loss': [],
            'val_loss': [],
            'val_rot_loss': [],
            'val_trans_loss': [],
            'learning_rates': [],
            'best_epoch': 0
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss, train_rot_loss, train_trans_loss = self.train_epoch(train_loader)

            # Step the scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Validate
            val_loss, val_rot_loss, val_trans_loss = self.validate(val_loader)

            # Log losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Update log data
            log_data['epochs'].append(epoch + 1)
            log_data['train_loss'].append(train_loss)
            log_data['train_rot_loss'].append(train_rot_loss)
            log_data['train_trans_loss'].append(train_trans_loss)
            log_data['val_loss'].append(val_loss)
            log_data['val_rot_loss'].append(val_rot_loss)
            log_data['val_trans_loss'].append(val_trans_loss)
            log_data['learning_rates'].append(current_lr)

            # Print progress
            print(f"Train Loss: {train_loss:.6f} (Rot: {train_rot_loss:.6f}, Trans: {train_trans_loss:.6f})")
            print(f"Val Loss: {val_loss:.6f} (Rot: {val_rot_loss:.6f}, Trans: {val_trans_loss:.6f})")
            print(f"Learning Rate: {current_lr:.8f}")

            # Check for improvement
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                epochs_without_improvement = 0
                log_data['best_epoch'] = epoch + 1

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! (Val Loss: {val_loss:.6f}, Improvement: {improvement:.6f})")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs. Best val loss: {best_val_loss:.6f}")

                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Save checkpoint at log interval
            if (epoch + 1) % log_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

                # Save log data
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)

                # Plot and save loss curves
                self._plot_losses(save_dir)

        # Final save
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(save_dir, 'final_model.pth'))

        # Save final log data
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        # Plot final loss curves
        self._plot_losses(save_dir)

        return log_data

    def _plot_losses(self, save_dir):
        """Plot and save loss curves and learning rate."""
        # Plot loss curves
        plt.figure(figsize=(12, 10))

        # Loss subplot
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Learning rate subplot
        if self.learning_rates:
            plt.subplot(2, 1, 2)
            plt.plot(self.learning_rates, label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')  # Log scale for better visualization

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ViTPosePredictor model with improved regularization')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio (increased)')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (ViT typically uses 224)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization (increased)')
    parser.add_argument('--pretrained', type=str, default="google/vit-base-patch16-224",
                      help='Pretrained ViT model to use')
    parser.add_argument('--log_interval', type=int, default=5,
                      help='Interval (in epochs) to save checkpoints and plots')
    parser.add_argument('--patience', type=int, default=15,
                      help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                      help='Dropout rate for model regularization')
    parser.add_argument('--freeze_layers', type=int, default=8,
                      help='Number of backbone layers to freeze')
    parser.add_argument('--rotation_mode', type=str, default='euler', choices=['euler', 'quaternion'],
                      help='Rotation representation mode')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    # Create training dataset with augmentation
    train_dataset = ViTPoseDataset(args.data_dir, args.image_size, augment=True)
    # Create validation dataset without augmentation
    val_dataset = ViTPoseDataset(args.data_dir, args.image_size, augment=False)

    # Get total dataset size
    total_size = len(train_dataset)
    print(f"Dataset size: {total_size} samples")

    # Create indices for train/val split
    indices = list(range(total_size))
    np.random.shuffle(indices)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    train_indices = indices[val_size:]  # Use the rest for training
    val_indices = indices[:val_size]    # Use first portion for validation

    # Create subset datasets
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    # Create data loaders with persistent workers for efficiency
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

    # Create model with improved regularization
    print(f"Creating ViTPosePredictor with pretrained model: {args.pretrained}")
    model = ViTPosePredictor(
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate,
        freeze_backbone_layers=args.freeze_layers
    )

    # Create trainer with custom loss and learning rate scheduling
    trainer = ViTPoseTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        rotation_mode=args.rotation_mode
    )

    # Train model with early stopping
    print(f"Starting training for up to {args.epochs} epochs (with early stopping)...")
    start_time = time.time()
    log_data = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.output_dir,
        log_interval=args.log_interval,
        patience=args.patience
    )

    # Print training summary
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best epoch: {log_data['best_epoch']}")
    print(f"Best validation loss: {min(log_data['val_loss']):.6f}")
    print(f"Final training loss: {log_data['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {log_data['val_loss'][-1]:.6f}")
    print(f"Model checkpoints and logs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
