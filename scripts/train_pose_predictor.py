import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
from PIL import Image
import torchvision.transforms as transforms

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.model.pose_predictor import PosePredictor

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

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        rotation_pred, _ = model(data)  # We only care about rotation for now
        loss = criterion(rotation_pred, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            rotation_pred, _ = model(data)
            loss = criterion(rotation_pred, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description='Train pose predictor model')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to')
    
    args = parser.parse_args()
    
    # Create output directory
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    
    # Create model
    model = PosePredictor().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Saved best model to {best_model_path}')

if __name__ == '__main__':
    main() 