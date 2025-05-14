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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.model.vit_pose_predictor import ViTPosePredictor
from src.train.trainer import PoseTrainer


class ViTPoseDataset(Dataset):
    """Dataset for training the ViTPosePredictor model."""
    
    def __init__(self, data_dir, image_size=224):
        """
        Args:
            data_dir (str): Directory containing the dataset
            image_size (int): Size to resize images to (ViT models typically use 224x224)
        """
        self.data_dir = Path(data_dir)
        self.example_dirs = sorted([d for d in os.listdir(data_dir) 
                                  if os.path.isdir(os.path.join(data_dir, d))])
        
        # Define image transforms for ViT
        # ViT models typically expect 224x224 images with specific normalization
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add some color augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])  # ViT normalization
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
        
        # For now, we'll use a dummy translation (all zeros)
        # since our dataset only contains rotation information
        translation = torch.zeros(3, dtype=torch.float32)
        
        return {
            'image': image,
            'rotation': rotation,
            'translation': translation
        }


class ViTPoseTrainer:
    """Training pipeline for ViTPosePredictor."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 learning_rate=1e-4, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.rotation_loss = nn.MSELoss()
        self.translation_loss = nn.MSELoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # For logging
        self.train_losses = []
        self.val_losses = []
        
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
            
            # Compute losses
            rot_loss = self.rotation_loss(pred_rot, target_rot)
            trans_loss = self.translation_loss(pred_trans, target_trans)
            
            # For now, we focus more on rotation prediction
            loss = rot_loss + 0.1 * trans_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            total_trans_loss += trans_loss.item()
            
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
                
                # Compute losses
                rot_loss = self.rotation_loss(pred_rot, target_rot)
                trans_loss = self.translation_loss(pred_trans, target_trans)
                
                # For now, we focus more on rotation prediction
                loss = rot_loss + 0.1 * trans_loss
                
                total_loss += loss.item()
                total_rot_loss += rot_loss.item()
                total_trans_loss += trans_loss.item()
                
        avg_loss = total_loss / len(dataloader)
        avg_rot_loss = total_rot_loss / len(dataloader)
        avg_trans_loss = total_trans_loss / len(dataloader)
        
        return avg_loss, avg_rot_loss, avg_trans_loss
    
    def train(self, train_loader, val_loader, num_epochs, save_dir, log_interval=1):
        """Train the model for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
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
            'best_epoch': 0
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_rot_loss, train_trans_loss = self.train_epoch(train_loader)
            
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
            
            # Print progress
            print(f"Train Loss: {train_loss:.6f} (Rot: {train_rot_loss:.6f}, Trans: {train_trans_loss:.6f})")
            print(f"Val Loss: {val_loss:.6f} (Rot: {val_rot_loss:.6f}, Trans: {val_trans_loss:.6f})")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log_data['best_epoch'] = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! (Val Loss: {val_loss:.6f})")
            
            # Save checkpoint at log interval
            if (epoch + 1) % log_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
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
        """Plot and save loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ViTPosePredictor model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (ViT typically uses 224)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--pretrained', type=str, default="google/vit-base-patch16-224", 
                      help='Pretrained ViT model to use')
    parser.add_argument('--log_interval', type=int, default=5, 
                      help='Interval (in epochs) to save checkpoints and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = ViTPoseDataset(args.data_dir, args.image_size)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set: {train_size} samples")
    print(f"Validation set: {val_size} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"Creating ViTPosePredictor with pretrained model: {args.pretrained}")
    model = ViTPosePredictor(pretrained=args.pretrained)
    
    # Create trainer
    trainer = ViTPoseTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    log_data = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.output_dir,
        log_interval=args.log_interval
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
