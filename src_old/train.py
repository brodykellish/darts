import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.autoencoder import BasicAutoEncoder
import logging
from datetime import datetime
import argparse

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device")
        # Enable memory efficient attention if available
        if hasattr(torch.backends.mps, 'is_mem_efficient_attention_enabled'):
            torch.backends.mps.enable_mem_efficient_attention()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device")
    return device

# Set up logging
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class DartDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        pose_path = os.path.join(self.data_dir, img_name.replace('.png', '.txt'))
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Load pose (3x4 matrix)
        pose = np.loadtxt(pose_path)
        R = pose[:, :3]  # 3x3 rotation matrix
        t = pose[:, 3]   # 3x1 translation vector
        
        # Convert rotation matrix to quaternion for easier learning
        q = self.rotmat2quat(R)
        
        return image, torch.FloatTensor(q), torch.FloatTensor(t)

    @staticmethod
    def rotmat2quat(R):
        # Convert rotation matrix to quaternion
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / S
                qx = 0.25 * S
                qy = (R[0,1] + R[1,0]) / S
                qz = (R[0,2] + R[2,0]) / S
            elif R[1,1] > R[2,2]:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / S
                qx = (R[0,1] + R[1,0]) / S
                qy = 0.25 * S
                qz = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / S
                qx = (R[0,2] + R[2,0]) / S
                qy = (R[1,2] + R[2,1]) / S
                qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

class PosePredictor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.autoencoder = BasicAutoEncoder(latent_dim)
        
        # Pose prediction from latent space
        self.rotation_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # quaternion
        )
        
        self.translation_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # translation vector
        )
        
    def forward(self, x):
        # Get reconstruction and latent code
        x_recon, z = self.autoencoder(x)
        
        # Predict pose
        q = self.rotation_head(z)
        t = self.translation_head(z)
        
        return x_recon, q, t

def compute_metrics(x_recon, images, q_pred, q_gt, t_pred, t_gt):
    # Reconstruction loss
    recon_loss = nn.MSELoss()(x_recon, images)
    
    # Normalize quaternions
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    q_gt = q_gt / torch.norm(q_gt, dim=1, keepdim=True)
    
    # Quaternion distance (1 - dot product)
    rot_loss = 1 - torch.abs(torch.sum(q_pred * q_gt, dim=1)).mean()
    
    # Translation loss
    trans_loss = nn.MSELoss()(t_pred, t_gt)
    
    # Total loss
    total_loss = recon_loss + 0.1 * rot_loss + 0.1 * trans_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss.item(),
        'rot_loss': rot_loss.item(),
        'trans_loss': trans_loss.item()
    }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, q_gt, t_gt in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        q_gt = q_gt.to(device)
        t_gt = t_gt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, q_pred, t_pred = model(images)
        
        # Compute metrics
        metrics = compute_metrics(x_recon, images, q_pred, q_gt, t_pred, t_gt)
        
        # Backward pass
        loss = metrics['total_loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device, logger):
    model.eval()
    epoch_metrics = {
        'total_loss': 0,
        'recon_loss': 0,
        'rot_loss': 0,
        'trans_loss': 0
    }
    
    with torch.no_grad():
        for batch_idx, (images, q_gt, t_gt) in enumerate(tqdm(dataloader, desc="Validation")):
            images = images.to(device)
            q_gt = q_gt.to(device)
            t_gt = t_gt.to(device)
            
            # Forward pass
            x_recon, q_pred, t_pred = model(images)
            
            # Compute metrics
            metrics = compute_metrics(x_recon, images, q_pred, q_gt, t_pred, t_gt)
            
            # Update epoch metrics
            for k, v in metrics.items():
                if k == 'total_loss':
                    epoch_metrics[k] += v.item()
                else:
                    epoch_metrics[k] += v
    
    # Average metrics
    for k in epoch_metrics:
        epoch_metrics[k] /= len(dataloader)
    
    return epoch_metrics

def save_model(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)

def load_model(model, optimizer, path, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'] + 1
    return 0

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the pose prediction model')
    parser.add_argument('--input-model', type=str,
                      help='Path to an existing model to continue training from')
    parser.add_argument('--output-model', type=str,
                      help='Path to save the trained model. If not specified, a new path will be created.')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting training...")
    
    # Handle model paths
    if args.input_model:
        if not os.path.exists(args.input_model):
            raise FileNotFoundError(f"Input model not found: {args.input_model}")
        input_model_path = args.input_model
        logger.info(f"Loading existing model from: {input_model_path}")
    else:
        input_model_path = None
        logger.info("No input model specified. Starting fresh training.")
    
    if args.output_model:
        output_model_path = args.output_model
    else:
        # Create a new model path with timestamp
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_model_path = f'models/model_{timestamp}.pth'
    logger.info(f"Model will be saved to: {output_model_path}")
    
    # Get device and set up
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Hyperparameters
    data_dir = "./dart_dataset"
    batch_size = 128  # Increased for M2 Max
    num_epochs = 200
    initial_lr = 1e-3
    min_lr = 1e-5  # Minimum learning rate
    val_split = 0.2
    save_interval = 5  # Save every 5 epochs
    
    # Create dataset and dataloader
    dataset = DartDataset(data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Optimize dataloader for MPS
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=False  # Disabled for MPS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False  # Disabled for MPS
    )
    
    logger.info(f"Dataset split - Train: {train_size}, Validation: {val_size}")
    
    # Initialize model and optimizer
    model = PosePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_lr
    )
    
    # Load model if specified
    start_epoch = 0
    if input_model_path:
        checkpoint = torch.load(input_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
        # Adjust scheduler to match loaded epoch
        for _ in range(start_epoch):
            scheduler.step()
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, LR: {current_lr:.2e}")
        
        # Update learning rate after optimizer step
        scheduler.step()
        
        # Save model periodically
        if (epoch + 1) % save_interval == 0:
            # Run validation before saving
            val_metrics = validate(model, val_loader, device, logger)
            logger.info(f"Validation metrics at epoch {epoch+1} - " + 
                       " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
            
            # Save checkpoint
            checkpoint_path = output_model_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            save_model(model, optimizer, epoch, val_metrics, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final validation
    logger.info("Running final validation...")
    val_metrics = validate(model, val_loader, device, logger)
    logger.info("Final validation metrics - " + " - ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
    
    # Save final model
    save_model(model, optimizer, num_epochs-1, val_metrics, output_model_path)
    logger.info(f"Saved final model to {output_model_path}")
