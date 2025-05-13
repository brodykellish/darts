import argparse
import os
from src.render.renderer import ModelRenderer
from src.model.pose_predictor import PosePredictor
from src.data.dataset import PoseDataset
from src.train.trainer import PoseTrainer
from torch.utils.data import DataLoader
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='3D Object Pose Estimation Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Render command
    render_parser = subparsers.add_parser('render', help='Render dataset from STL file')
    render_parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    render_parser.add_argument('--output_dir', type=str, required=True, help='Output directory for renders')
    render_parser.add_argument('--num_images', type=int, default=1000, help='Number of images to render')
    render_parser.add_argument('--image_res', type=int, default=256, help='Image resolution')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train pose prediction model')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Directory with rendered images and poses')
    train_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    return parser.parse_args()

def render_dataset(args):
    """Render dataset from STL file."""
    renderer = ModelRenderer(args.stl_path, args.output_dir, args.image_res)
    renderer.render_dataset(args.num_images)

def train_model(args):
    """Train pose prediction model."""
    # Create dataset
    dataset = PoseDataset(args.data_dir)
    
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model and trainer
    model = PosePredictor()
    trainer = PoseTrainer(model)
    
    # Train
    trainer.train(train_loader, val_loader, args.num_epochs, args.output_dir)

def main():
    args = parse_args()
    
    if args.command == 'render':
        render_dataset(args)
    elif args.command == 'train':
        train_model(args)
    else:
        print("Please specify a command: render or train")

if __name__ == '__main__':
    main() 