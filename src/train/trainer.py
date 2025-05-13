import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..model.pose_predictor import PosePredictor
from ..data.dataset import PoseDataset
import os
from tqdm import tqdm

class PoseTrainer:
    """Training pipeline for pose prediction."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.rotation_loss = nn.MSELoss()
        self.translation_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
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
            loss = rot_loss + trans_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
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
                loss = rot_loss + trans_loss
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs, save_dir):
        """Train the model for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')) 